# train

## datasets

Details:

- [SSDAugmentation](../data/augmentation.md)

```python
# COCO
args.dataset_root = COCO_ROOT = osp.join(HOME, 'data/coco/')
cfg = coco = {
    'num_classes': 201,
    'lr_steps': (280000, 360000, 400000),
    'max_iter': 400000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [21, 45, 99, 153, 207, 261],
    'max_sizes': [45, 99, 153, 207, 261, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'COCO',
}
dataset = COCODetection(
    root=args.dataset_root,
    transform=SSDAugmentation(
        cfg['min_dim'], MEANS
    )
)

# VOC
args.dataset_root = VOC_ROOT = osp.join(HOME, "data/VOCdevkit/")
cfg = voc = {
    'num_classes': 21,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}
dataset = VOCDetection(
    root=args.dataset_root,
    transform=SSDAugmentation(
        cfg['min_dim'], MEANS
    )
)
```

## training visualization

```python
if args.visdom:
    import visdom
    viz = visdom.Visdom()
```

## model

```python
net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
net = torch.nn.DataParallel(ssd_net).cuda()
cudnn.benchmark = True
net.train()
```

## optimizer

```python
optimizer = optim.SGD(
    net.parameters(), 
    lr=args.lr, 
    momentum=args.momentum,
    weight_decay=args.weight_decay
)
```

## loss

Details: [multibox_loss](multibox_loss.md)

```python
criterion = MultiBoxLoss(
    num_classes=cfg['num_classes'], 
    overlap_thresh=0.5, 
    prior_for_matching=True, 
    bkg_label=0, 
    neg_mining=True, 
    neg_pos=3, 
    neg_overlap=0.5,
    encode_target=False, 
    use_gpu=args.cuda
)
```

## dataloader

```python
data_loader = data.DataLoader(
    dataset, 
    args.batch_size,
    num_workers=args.num_workers,
    shuffle=True, 
    collate_fn=detection_collate,
    pin_memory=True
)

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).
    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations
    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets
```

## training loop

```python
# loss counters
loc_loss = 0
conf_loss = 0
epoch = 0
print('Loading the dataset...')

epoch_size = len(dataset) // args.batch_size
print('Training SSD on:', dataset.name)
print('Using the specified args:')
print(args)

step_index = 0

if args.visdom:
    vis_title = 'SSD.PyTorch on ' + dataset.name
    vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
    iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
    epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)
    
# create batch iterator
batch_iterator = iter(data_loader)
for iteration in range(args.start_iter, cfg['max_iter']):
    if args.visdom and iteration != 0 and (iteration % epoch_size == 0):
        update_vis_plot(
            epoch, loc_loss, conf_loss, epoch_plot, None,
            'append', epoch_size
        )
        # reset epoch loss counters
        loc_loss = 0
        conf_loss = 0
        epoch += 1

    if iteration in cfg['lr_steps']:
        step_index += 1
        adjust_learning_rate(optimizer, args.gamma, step_index)

    # load train data
    images, targets = next(batch_iterator)

    if args.cuda:
        images = Variable(images.cuda())
        targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
    else:
        images = Variable(images)
        targets = [Variable(ann, volatile=True) for ann in targets]
        
    # forward
    t0 = time.time()
    out = net(images)
    
    # backprop
    optimizer.zero_grad()
    loss_l, loss_c = criterion(out, targets)
    loss = loss_l + loss_c
    loss.backward()
    optimizer.step()
    
    t1 = time.time()
    loc_loss += loss_l.data[0]
    conf_loss += loss_c.data[0]

    if iteration % 10 == 0:
        print('timer: %.4f sec.' % (t1 - t0))
        print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.data[0]), end=' ')

    if args.visdom:
        update_vis_plot(
            iteration, loss_l.data[0], loss_c.data[0],
            iter_plot, epoch_plot, 'append'
        )

    if iteration != 0 and iteration % 5000 == 0:
        print('Saving state, iter:', iteration)
        torch.save(
            ssd_net.state_dict(), 
            'weights/ssd300_COCO_' + repr(iteration) + '.pth'
        )
        
torch.save(
    ssd_net.state_dict(),
    args.save_folder + '' + args.dataset + '.pth'
)
```
