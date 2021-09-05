# Corner Pooling

![img](images/corner_pooling.png)

```python
class corner_pool(nn.Module):
    def __init__(self, dim, pool1, pool2):
        super(corner_pool, self).__init__()
        self._init_layers(dim, pool1, pool2)

    def _init_layers(self, dim, pool1, pool2):
        self.p1_conv1 = convolution(k=3, inp_dim=dim, out_dim=128)
        self.p2_conv1 = convolution(k=3, inp_dim=dim, out_dim=128)

        self.p_conv1 = nn.Conv2d(
            in_channels=128, 
            out_channels=dim, 
            kernel_size=(3, 3), 
            padding=(1, 1), 
            bias=False
        )
        self.p_bn1 = nn.BatchNorm2d(dim)

        self.conv1 = nn.Conv2d(
            in_channels=dim, 
            out_channels=dim, 
            kernel_size=(1, 1), 
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = convolution(k=3, inp_dim=dim, out_dim=dim)

        self.pool1 = pool1()
        self.pool2 = pool2()

    def forward(self, x):
        # x.shape: torch.Size([1, 256, 128, 128])

        # pool 1
        p1_conv1 = self.p1_conv1(x)
        # p1_conv1.shape: torch.Size([1, 128, 128, 128])
        pool1 = self.pool1(p1_conv1)
        # pool1.shape: torch.Size([1, 128, 128, 128])

        # pool 2
        p2_conv1 = self.p2_conv1(x)
        # p2_conv1.shape: torch.Size([1, 128, 128, 128])
        pool2 = self.pool2(p2_conv1)
        # pool2.shape: torch.Size([1, 128, 128, 128])

        # (pool1 + pool2).shape: torch.Size([1, 128, 128, 128]) 
        p_conv1 = self.p_conv1(pool1 + pool2)
        # p_conv1.shape: torch.Size([1, 256, 128, 128])
        p_bn1 = self.p_bn1(p_conv1)
        # p_bn1.shape: torch.Size([1, 256, 128, 128])

        # residual branch
        conv1 = self.conv1(x)
        # conv1.shape: torch.Size([1, 256, 128, 128])
        bn1 = self.bn1(conv1)
        # bn1.shape: torch.Size([1, 256, 128, 128])
        relu1 = self.relu1(p_bn1 + bn1)
        # relu1.shape: torch.Size([1, 256, 128, 128])

        conv2 = self.conv2(relu1)
        # conv2.shape: torch.Size([1, 256, 128, 128])

        return conv2
```

## Initialization

```python
tl_modules = nn.ModuleList([
    corner_pool(dim=256, pool1=TopPool, pool2=LeftPool) 
    for _ in range(stacks)
])
br_modules = nn.ModuleList([
    corner_pool(dim=256, pool1=BottomPool, pool2=RightPool) 
    for _ in range(stacks)
])
```

## Training Forward

- **input**: cnvs
- **fn**: [self.tl_modules, self.br_modules] 
- **output**: [tl_modules, br_modules]

```python
# convs.shape: 
# CornerNet: [torch.Size([4, 256, 128, 128])] * stacks
# CornerNet Squeeze: [torch.Size([13, 256, 64, 64])] * stacks
# CornerNet Saccade: [torch.Size([4, 256, 64, 64])] * stacks

tl_modules = [tl_mod_(cnv) for tl_mod_, cnv in zip(self.tl_modules, cnvs)]
br_modules = [br_mod_(cnv) for br_mod_, cnv in zip(self.br_modules, cnvs)]

# tl_modules.shape: 
# CornerNet: [torch.Size([4, 256, 128, 128])] * stacks
# CornerNet Squeeze: [torch.Size([13, 256, 64, 64])] * stacks
# CornerNet Saccade: [torch.Size([4, 256, 64, 64])] * stacks
```

## Testing Forward

```python
# cnvs.shape:
# CornerNet: [torch.Size([2, 256, 128, 192])] * stacks
# CornerNet Squeeze: [torch.Size([13, 256, 64, 96])] * stacks
# CornerNet Saccade: [torch.Size([2, 256, 64, 64])] * stacks

tl_mod = self.tl_modules[-1](cnvs[-1])
br_mod = self.br_modules[-1](cnvs[-1])

# tl_mod.shape: 
# CornerNet: torch.Size([2, 256, 128, 192])
# CornerNet Squeeze: torch.Size([1, 256, 64, 96])
# CornerNet Saccade: torch.Size([2, 256, 64, 64])
```