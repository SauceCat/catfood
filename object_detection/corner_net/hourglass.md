# Hourglass backbone

## Initialization

```python
# CornerNet, CornerNet Squeeze
hgs = hg(pre, hg_mods, cnvs, inters, cnvs_, inters_)

# CornerNet Saccade
hgs = saccade(pre, hg_mods, cnvs, inters, cnvs_, inters_)
```

## Forward

### CornerNet

```python
def forward(self, x):
    # x.shape: torch.Size([4, 3, 511, 511])
    inter = self.pre(x)
    # inter.shape: torch.Size([4, 256, 128, 128])

    cnvs  = []
    for ind, (hg_, cnv_) in enumerate(zip(self.hgs, self.cnvs)):
        # inter.shape: torch.Size([4, 256, 128, 128])
        hg = hg_(inter)
        # hg.shape: torch.Size([4, 256, 128, 128])
        cnv = cnv_(hg)
        # cnv.shape: torch.Size([4, 256, 128, 128])
        cnvs.append(cnv)

        if ind < len(self.hgs) - 1:
            # inter.shape: torch.Size([4, 256, 128, 128])
            # self.inters_[ind](inter).shape: torch.Size([4, 256, 128, 128])
            # cnv.shape: torch.Size([4, 256, 128, 128])
            # self.cnvs_[ind](cnv).shape: torch.Size([4, 256, 128, 128])
            inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
            # inter.shape: torch.Size([4, 256, 128, 128])
            
            inter = nn.functional.relu_(inter)
            inter = self.inters[ind](inter)
            # inter.shape: torch.Size([4, 256, 128, 128])
            
    # cnvs.shape: [torch.Size([4, 256, 128, 128])] * stacks
    return cnvs
```

### CornerNet Squeeze

```python
def forward(self, x):
    # x.shape: torch.Size([13, 3, 511, 511])
    inter = self.pre(x)
    # inter.shape: torch.Size([13, 256, 64, 64])

    cnvs  = []
    for ind, (hg_, cnv_) in enumerate(zip(self.hgs, self.cnvs)):
        # inter.shape: torch.Size([13, 256, 64, 64])
        hg  = hg_(inter)
        # hg.shape: torch.Size([13, 256, 64, 64])
        
        cnv = cnv_(hg)
        # cnv.shape: torch.Size([13, 256, 64, 64])
        cnvs.append(cnv)
        
        if ind < len(self.hgs) - 1:
            # inter.shape: torch.Size([13, 256, 64, 64]) 
            # self.inters_[ind](inter).shape: torch.Size([13, 256, 64, 64]) 
            # cnv.shape: torch.Size([13, 256, 64, 64])
            # self.cnvs_[ind](cnv).shape: torch.Size([13, 256, 64, 64]) 
            inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
            inter = nn.functional.relu_(inter)
            inter = self.inters[ind](inter)
            # inter.shape: torch.Size([13, 256, 64, 64])

    # cnvs.shape: [torch.Size([13, 256, 64, 64])] * stacks
    return cnvs
```

### CornerNet Saccade

```python
def forward(self, x):
    # x.shape: torch.Size([4, 3, 255, 255])
    inter = self.pre(x)
    # inter.shape: torch.Size([4, 256, 64, 64])

    cnvs  = []
    # attentions output
    atts  = []

    for ind, (hg_, cnv_) in enumerate(zip(self.hgs, self.cnvs)):
        # inter.shape: torch.Size([4, 256, 64, 64])
        hg, ups = hg_(inter)
        # hg.shape: torch.Size([4, 256, 64, 64])
        # ups.shape: [
        #     torch.Size([4, 384, 16, 16]), 
        #     torch.Size([4, 384, 32, 32]), 
        #     torch.Size([4, 256, 64, 64])
        # ]
        cnv = cnv_(hg)
        # cnv: torch.Size([4, 256, 64, 64])
        cnvs.append(cnv)
        atts.append(ups)

        if ind < len(self.hgs) - 1:
            # inter.shape: torch.Size([4, 256, 64, 64])
            # self.inters_[ind](inter).shape: torch.Size([4, 256, 64, 64])
            # cnv.shape: torch.Size([4, 256, 64, 64])
            # self.cnvs_[ind](cnv).shape: torch.Size([4, 256, 64, 64])
            inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
            inter = nn.functional.relu_(inter)
            inter = self.inters[ind](inter)
            # inter.shape: torch.Size([4, 256, 64, 64])

    # cnvs.shape: [torch.Size([4, 256, 64, 64])] * stacks
    # atts.shape: [
    #     torch.Size([4, 384, 16, 16]), 
    #     torch.Size([4, 384, 32, 32]), 
    #     torch.Size([4, 256, 64, 64])
    # ] * stacks
    return cnvs, atts
```

## pre

### CornerNet, CornerNet Saccade

```python
pre = nn.Sequential(
    convolution(k=7, inp_dim=3, out_dim=128, stride=2, with_bn=True),
    residual(inp_dim=128, out_dim=256, k=3, stride=2),
)

# forward
# x.shape: torch.Size([4, 3, 511, 511])
inter = self.pre(x)
# inter.shape: torch.Size([4, 256, 128, 128])
```

### CornerNet Squeeze

```python
pre = nn.Sequential(
    convolution(k=7, inp_dim=3, out_dim=128, stride=2, with_bn=True),
    residual(inp_dim=128, out_dim=256, k=3, stride=2),
    residual(inp_dim=256, out_dim=256, k=3, stride=2),
)

# forward
# x.shape: torch.Size([13, 3, 511, 511])
inter = self.pre(x)
# inter.shape: torch.Size([13, 256, 64, 64])
```

## hg_mods

### CornerNet

```python
hg_mods = nn.ModuleList([
    hg_module(
        n=5, 
        dims=[256, 256, 384, 384, 384, 512], 
        modules=[2, 2, 2, 2, 2, 4],
        make_pool_layer=make_pool_layer,
        make_hg_layer=make_hg_layer
    ) for _ in range(2)
])
```

### CornerNet Saccade

```python
hg_mods = nn.ModuleList([
    saccade_module(
        n=3, 
        dims=[256, 384, 384, 512], 
        modules=[1, 1, 1, 1],
        make_pool_layer=make_pool_layer,
        make_hg_layer=make_hg_layer
    ) for _ in range(3)
])
```

### CornerNet Squeeze

```python
hg_mods = nn.ModuleList(
    [
        hg_module(
            n=4,
            dims=[256, 256, 384, 384, 512],
            modules=[2, 2, 2, 2, 4],
            make_pool_layer=make_pool_layer,
            make_unpool_layer=make_unpool_layer,
            make_up_layer=make_layer,
            make_low_layer=make_layer,
            make_hg_layer_revr=make_layer_revr,
            make_hg_layer=make_hg_layer,
        )
        for _ in range(2)
    ]
)
```

## cnvs, inters, cnvs\_, inters\_

### CornerNet, CornerNet Saccade, CornerNet Squeeze

```python
cnvs = nn.ModuleList([
    convolution(k=3, inp_dim=256, out_dim=256) 
    for _ in range(stacks)
])
inters = nn.ModuleList([
    residual(inp_dim=256, out_dim=256, k=3, stride=1) 
    for _ in range(stacks - 1)
])
cnvs_ = nn.ModuleList([
    self._merge_mod() for _ in range(stacks - 1)
])
inters_ = nn.ModuleList([
    self._merge_mod() for _ in range(stacks - 1)
])

def _merge_mod(self):
    return nn.Sequential(
        nn.Conv2d(
        	in_channels=256, out_channels=256, 
        	kernel_size=(1, 1), bias=False
        ),
        nn.BatchNorm2d(256)
    )
```

## Build blocks

### convolution

Simple `conv-bn-relu` layer.

```python
class convolution(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(convolution, self).__init__()

        pad = (k - 1) // 2
        self.conv = nn.Conv2d(
            in_channels=inp_dim,
            out_channels=out_dim,
            kernel_size=(k, k),
            padding=(pad, pad),
            stride=(stride, stride),
            bias=not with_bn,
        )
        self.bn = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.conv(x)
        bn = self.bn(conv)
        relu = self.relu(bn)
        return relu
```

### residual

Simple residual layer: `conv1-bn1-relu1-conv2-bn2 + conv-bn`

```python
class residual(nn.Module):
    def __init__(self, inp_dim, out_dim, k=3, stride=1):
        super(residual, self).__init__()
        p = (k - 1) // 2

        self.conv1 = nn.Conv2d(
            in_channels=inp_dim,
            out_channels=out_dim,
            kernel_size=(k, k),
            padding=(p, p),
            stride=(stride, stride),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            in_channels=out_dim, 
            out_channels=out_dim, 
            kernel_size=(k, k), 
            padding=(p, p), 
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_dim)

        self.skip = (
            nn.Sequential(
                nn.Conv2d(
                    in_channels=inp_dim, 
                    out_channels=out_dim, 
                    kernel_size=(1, 1), 
                    stride=(stride, stride), 
                    bias=False
                ),
                nn.BatchNorm2d(out_dim),
            )
            if stride != 1 or inp_dim != out_dim
            else nn.Sequential()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        relu1 = self.relu1(bn1)

        conv2 = self.conv2(relu1)
        bn2 = self.bn2(conv2)
        
        skip = self.skip(x)
        return self.relu(bn2 + skip)
```

### fire_module

```python
class fire_module(nn.Module):
    def __init__(self, inp_dim, out_dim, sr=2, stride=1):
        super(fire_module, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=inp_dim,
            out_channels=out_dim // sr,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_dim // sr)
        self.conv_1x1 = nn.Conv2d(
            in_channels=out_dim // sr,
            out_channels=out_dim // 2,
            kernel_size=1,
            stride=stride,
            bias=False,
        )
        self.conv_3x3 = nn.Conv2d(
            in_channels=out_dim // sr,
            out_channels=out_dim // 2,
            kernel_size=3,
            padding=1,
            stride=stride,
            groups=out_dim // sr,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_dim)
        self.skip = stride == 1 and inp_dim == out_dim
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)

        conv2 = torch.cat((self.conv_1x1(bn1), self.conv_3x3(bn1)), 1)
        bn2 = self.bn2(conv2)

        if self.skip:
            return self.relu(bn2 + x)
        else:
            return self.relu(bn2)
```

### residual vs. fire_module

#### conv1

residual: `conv1-bn1-relu1`

```python
# x.shape: torch.Size([4, 128, 256, 256])
conv1 = self.conv1(x)
bn1 = self.bn1(conv1)
relu1 = self.relu1(bn1)
# relu1.shape: torch.Size([4, 256, 128, 128])

self.conv1 = nn.Conv2d(
    in_channels=128,
    out_channels=256,
    kernel_size=(3, 3),
    padding=(1, 1),
    stride=(2, 2),
    bias=False,
)
self.bn1 = nn.BatchNorm2d(256)
self.relu1 = nn.ReLU(inplace=True)
```

fire_module: `conv1-bn1`

```python
# x.shape: torch.Size([13, 256, 64, 64])
conv1 = self.conv1(x)
bn1 = self.bn1(conv1)
# bn1.shape: torch.Size([13, 128, 64, 64]) 

self.conv1 = nn.Conv2d(
    in_channels=256, 
    out_channels=128, 
    kernel_size=(1, 1), 
    stride=(1, 1), 
    bias=False
)
self.bn1 = nn.BatchNorm2d(128)
```

#### conv2

residual: `conv2-bn2`

```python
# relu1.shape: torch.Size([4, 256, 128, 128])
conv2 = self.conv2(relu1)
bn2 = self.bn2(conv2)
# bn2.shape: torch.Size([4, 256, 128, 128])

self.conv2 = nn.Conv2d(
    in_channels=256,
    out_channels=256,
    kernel_size=(3, 3),
    padding=(1, 1),
    stride=(1, 1),
    bias=False
)
self.bn2 = nn.BatchNorm2d(256)
```

fire_module: `conv2-bn2`

```python
# bn1.shape: torch.Size([13, 128, 64, 64]) 
# self.conv_1x1(bn1).shape: torch.Size([13, 128, 64, 64])
# self.conv_3x3(bn1).shape: torch.Size([13, 128, 64, 64])
conv2 = torch.cat(
    (
        self.conv_1x1(bn1), 
        self.conv_3x3(bn1)
    ), 1
)
bn2 = self.bn2(conv2)
# bn2.shape: torch.Size([13, 256, 64, 64])

self.conv_1x1 = nn.Conv2d(
    in_channels=128, 
    out_channels=128, 
    kernel_size=(1, 1), 
    stride=(1, 1), 
    bias=False
)
self.conv_3x3 = nn.Conv2d(
    in_channels=128,
    out_channels=128,
    kernel_size=(3, 3),
    padding=(1, 1),
    stride=(1, 1),
    groups=128,
    bias=False,
)
self.bn2 = nn.BatchNorm2d(256)
```

#### skip connection

residual: `conv1-bn1-relu1-conv2-bn2 + conv-bn`

```python
# x.shape: torch.Size([4, 128, 256, 256])
skip = self.skip(x)
# skip.shape: torch.Size([4, 256, 128, 128])

# bn2.shape: torch.Size([4, 256, 128, 128])
# self.relu(bn2 + skip): torch.Size([4, 256, 128, 128])
return self.relu(bn2 + skip)

self.skip = (
    nn.Sequential(
        nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=(1, 1),
            stride=(2, 2),
            bias=False
        ),
        nn.BatchNorm2d(256),
    )
    if stride != 1 or inp_dim != out_dim
    else nn.Sequential()
)
self.relu = nn.ReLU(inplace=True)
```

fire_module: `conv1-bn1-conv2-bn2 + x`

```python
# fire_module
# x.shape: torch.Size([13, 256, 64, 64])
# bn2.shape: torch.Size([13, 256, 64, 64])
if self.skip:
    return self.relu(bn2 + x)
else:
    return self.relu(bn2)

self.skip = stride == 1 and inp_dim == out_dim
self.relu = nn.ReLU(inplace=True)
```
