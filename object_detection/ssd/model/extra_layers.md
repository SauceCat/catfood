# Extra layers

![](../images/ssd_extra_layers.png)

```python
extra_layers = add_extras(
    cfg=[
        256, 
        'S', 512, 
        128, 
        'S', 256, 
        128, 256, 128, 256
    ], 
    i=1024, 
    batch_norm=False
)

def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i # 1024
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [
                    nn.Conv2d(
                        in_channels, 
                        cfg[k + 1],
                        kernel_size=(1, 3)[flag], 
                        stride=2, 
                        padding=1
                    )
                ]
            else:
                layers += [
                    nn.Conv2d(
                        in_channels, 
                        v, 
                        kernel_size=(1, 3)[flag]
                    )
                ]
            flag = not flag
        in_channels = v
    return layers
```

### extra_layers

```python
extra_layers = [
    # 0: 1024->256
    Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1)),
    
    # 1: S, 256->512
    Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
    
    # 2: 512->128
    Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1)),
    
    # 3: S, 128->256
    Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
    
    # 4: 256->128
    Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1)),
    
    # 5: 128->256
    Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1)),
    
    # 6: 256->128
    Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1)),
    
    # 7: 128->256
    Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1))
]
```