# DistributedDataParallel

## Reference

- https://pytorch.org/docs/stable/notes/ddp.html#ddp
- https://pytorch.org/tutorials/beginner/dist_overview.html
- https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
- https://towardsdatascience.com/how-to-convert-a-pytorch-dataparallel-project-to-use-distributeddataparallel-b84632eed0f6

---

## Introduction

[DistributedDataParallel](https://pytorch.org/docs/stable/nn.html#torch.nn.parallel.DistributedDataParallel) (DDP) implements data parallelism at the module level which can run across multiple machines. Applications using DDP should spawn multiple processes and create a single DDP instance per process. DDP uses collective communications in the [torch.distributed](https://pytorch.org/tutorials/intermediate/dist_tuto.html) package to synchronize gradients and buffers. 

More specifically, DDP registers an autograd hook for each parameter given by `model.parameters()` and the hook will fire when the corresponding gradient is computed in the backward pass. Then DDP uses that signal to trigger gradient synchronization across processes. 

The recommended way to use DDP is to spawn one process for each model replica, where a model replica can span multiple devices. DDP processes can be placed on the same machine or across machines, but GPU devices cannot be shared across processes. 

Compared to [DataParallel](https://pytorch.org/docs/master/generated/torch.nn.DataParallel.html), [DistributedDataParallel](https://pytorch.org/docs/master/generated/torch.nn.parallel.DistributedDataParallel.html) requires one more step to set up, i.e., calling [init_process_group](https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group). DDP uses multi-process parallelism, and hence there is no GIL contention across model replicas. Moreover, the model is broadcast at DDP construction time instead of in every forward pass, which also helps to speed up training. DDP is shipped with several performance optimization technologies.

---

## Basic Use Case

Please note, as DDP broadcasts model states from rank 0 process to all other processes in the DDP constructor, you don’t need to worry about different DDP processes start from different model parameter initial values.

```python
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP


def example(rank, world_size):
    # create default process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    
    # create local model
    model = nn.Linear(10, 10).to(rank)
    # construct DDP model
    ddp_model = DDP(model, device_ids=[rank])
    # define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    
    # save the model in only one process and then load it to all processes
    # reducing write overhead.
    CHECKPOINT_PATH = tempfile.gettempdir() + "/model.checkpoint"
    if rank == 0:
        # All processes should see same parameters as they all start from same
        # random parameters and gradients are synchronized in backward passes.
        # Therefore, saving it in one process is sufficient.
        torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)
        
    # Use a barrier() to make sure that process 1 loads the model after process
    # 0 saves it.
    dist.barrier()
    # configure map_location properly
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    ddp_model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=map_location))

    # forward pass
    # put the data to the correct device
    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10).to(rank))
    labels = torch.randn(20, 10).to(rank)
    # backward pass
    loss_fn(outputs, labels).backward()
    # update parameters
    optimizer.step()
    
	# Not necessary to use a dist.barrier() to guard the file deletion below
    # as the AllReduce ops in the backward pass of DDP already served as
    # a synchronization.
    if rank == 0:
        os.remove(CHECKPOINT_PATH)

def main():
    world_size = 2
    mp.spawn(example,
        args=(world_size,),
        nprocs=world_size,
        join=True)

if __name__=="__main__":
    main()
```

DDP wraps lower-level distributed communication details and provides a clean API as if it is a local model. Gradient synchronization communications take place during the backward pass and overlap with the backward computation. 

When the `backward()` returns, `param.grad` already contains the synchronized gradient tensor. For basic use cases, DDP only requires a few more LoCs to set up the process group. When applying DDP to more advanced use cases, some caveats require caution.

---

## Internal Design

This section reveals how it works under the hood of [`torch.nn.parallel.DistributedDataParallel`](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel) by diving into details of every step in one iteration.

### Prerequisite

DDP relies on c10d `ProcessGroup` for communications. Hence, applications must create `ProcessGroup` instances before constructing DDP.

### Construction

- The DDP constructor takes a reference to the local module, and broadcasts `state_dict()` from the process with rank 0 to all other processes in the group to make sure that all model replicas start from the exact same state. 

- Then, each DDP process creates a local `Reducer`, which later will take care of the gradients synchronization during the backward pass. 

- To improve communication efficiency, the `Reducer` organizes parameter gradients into buckets, and reduces one bucket at a time. 

  - Bucket size can be configured by setting the bucket_cap_mb argument in DDP constructor. 

  - The mapping from parameter gradients to buckets is determined at the construction time, based on the bucket size limit and parameter sizes. 

  - Model parameters are allocated into buckets in (roughly) the reverse order of `Model.parameters()` from the given model. 

  - The reason for using the reverse order is because DDP expects gradients to become ready during the backward pass in approximately that order. 

  - The figure below shows an example. Note that, the `grad0` and `grad1` are in `bucket1`, and the other two gradients are in `bucket0`. 

    ![ddp_grad_sync.png](https://user-images.githubusercontent.com/16999635/72401724-d296d880-371a-11ea-90ab-737f86543df9.png)

  - Of course, this assumption might not always be true, and when that happens it could hurt DDP backward speed as the `Reducer` cannot kick off the communication at the earliest possible time. 

- Besides bucketing, the `Reducer` also registers autograd hooks during construction, one hook per parameter. These hooks will be triggered during the backward pass when the gradient becomes ready.

### Forward Pass

- The DDP takes the input and passes it to the local model, and then analyzes the output from the local model if `find_unused_parameters` is set to `True`. 
  - This mode allows running backward on a subgraph of the model, and DDP finds out which parameters are involved in the backward pass by traversing the autograd graph from the model output and marking all unused parameters as ready for reduction. 
- During the backward pass, the `Reducer` would only wait for unready parameters, but it would still reduce all buckets. 
- Marking a parameter gradient as ready does not help DDP skip buckets as for now, but it will prevent DDP from waiting for absent gradients forever during the backward pass. 
- Note that traversing the autograd graph introduces extra overheads, so applications should only set `find_unused_parameters` to `True` when necessary.

### Backward Pass

- The `backward()` function is directly invoked on the loss `Tensor`, which is out of DDP’s control, and DDP uses autograd hooks registered at construction time to trigger gradients synchronizations. 
- When one gradient becomes ready, its corresponding DDP hook on that grad accumulator will fire, and DDP will then mark that parameter gradient as ready for reduction. 
- When gradients in one bucket are all ready, the `Reducer` kicks off an asynchronous `allreduce` on that bucket to calculate mean of gradients across all processes. 
- When all buckets are ready, the `Reducer` will block waiting for all `allreduce` operations to finish. 
- When this is done, averaged gradients are written to the `param.grad` field of all parameters. So after the backward pass, the grad field on the same corresponding parameter across different DDP processes should be the same.

### Optimizer Step

From the optimizer’s perspective, it is optimizing a local model. Model replicas on all DDP processes can keep in sync because they all start from the same state and they have the same averaged gradients in every iteration.

---

## Comparison between other methods

### multiprocessing

There are significant caveats to using CUDA models with [`multiprocessing`](https://pytorch.org/docs/stable/multiprocessing.html#module-torch.multiprocessing); unless care is taken to meet the data handling requirements exactly, it is likely that your program will have incorrect or undefined behavior.

### nn.DataParallel

Before we dive in, let’s clarify why, despite the added complexity, you would consider using `DistributedDataParallel` over `DataParallel`:

- First, `DataParallel` is single-process, multi-thread, and only works on a single machine, while `DistributedDataParallel` is multi-process and works for both single- and multi- machine training. `DataParallel` is usually slower than `DistributedDataParallel` even on a single machine due to GIL contention across threads, per-iteration replicated model, and additional overhead introduced by scattering inputs and gathering outputs.
- Recall from the [prior tutorial](https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html) that if your model is too large to fit on a single GPU, you must use **model parallel** to split it across multiple GPUs. `DistributedDataParallel` works with **model parallel**; `DataParallel` does not at this time. When DDP is combined with model parallel, each DDP process would use model parallel, and all processes collectively would use data parallel.

### Primitives on which DataParallel is implemented upon

In general, pytorch’s `nn.parallel` primitives can be used independently. We have implemented simple MPI-like primitives:

- **replicate:** replicate a Module on multiple devices
- **scatter:** distribute the input in the first-dimension
- **gather:** gather and concatenate the input in the first-dimension
- **parallel_apply:** apply a set of already-distributed inputs to a set of already-distributed models.

To give a better clarity, here function `data_parallel` composed using these collectives

```python
def data_parallel(module, input, device_ids, output_device=None):
    if not device_ids:
        return module(input)

    if output_device is None:
        output_device = device_ids[0]

    replicas = nn.parallel.replicate(module, device_ids)
    inputs = nn.parallel.scatter(input, device_ids)
    replicas = replicas[:len(inputs)]
    outputs = nn.parallel.parallel_apply(replicas, inputs)
    return nn.parallel.gather(outputs, output_device)
```