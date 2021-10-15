# VGG16

![](../images/vgg16.png)

```python
vgg_layers = vgg(
    cfg=[64, 64, 
         'M', 128, 128, 
         'M', 256, 256, 256, 
         'C', 512, 512, 512, 
         'M', 512, 512, 512
        ], 
    i=3, 
    batch_norm=False
)

# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i # 3
    
    # layer 0-29
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
            
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1) 
    layers += [
        # layer 30-34
        pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)
    ]
    return layers
```

## layers

```python
# 0-3: 3->64->64
Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
ReLU(inplace=True)
Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
ReLU(inplace=True)

# 4: M, downsample
MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

# 5-8: 64->128->128
Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
ReLU(inplace=True)
Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
ReLU(inplace=True)

# 9: M, downsample
9 5 MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

# 10-15: 128->256->256->256
Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
ReLU(inplace=True)
Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
ReLU(inplace=True)
Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
ReLU(inplace=True)

# 16: C, downsample
MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)

# 17-22: 256->512->512->512
Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
ReLU(inplace=True)
Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
ReLU(inplace=True)
Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
ReLU(inplace=True)

# 23: M, downsample
MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

# 24-29: 512->512->512->512
Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
ReLU(inplace=True)
Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
ReLU(inplace=True)
Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
ReLU(inplace=True)

# 30: pool5, downsample
MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)

# 31-32: conv6, 512->1024
Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(6, 6), dilation=(6, 6))
ReLU(inplace=True)

# 33-34: conv7, 1024->1024
Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
ReLU(inplace=True)
```

