# Build blocks

## convolution

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

## residual

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

## fire_module

We replace the residual block with the **fire module**, the building block of **SqueezeNet**. 

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

## residual vs. fire_module

![](/home/saucecat/Desktop/develop/catfood/object_detection/corner_net/images/fire_module.png)

### conv1

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

The fire module first reduces the number of input channels with a squeeze layer consisting of 1×1 filters.

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

### conv2

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

Then, it feeds the result through an expand layer consisting of a mixture of 1×1 and 3×3 filters. Furthermore, inspired by the success of MobileNets, we replace the 3×3 standard convolution in the second layer with a 3×3 depth-wise separable convolution, which further improves inference time.

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

### skip connection

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

