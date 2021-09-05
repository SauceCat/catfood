# hg_mods

![img](https://miro.medium.com/max/700/1*WW20CTpg_ipNwtT8kofpYQ.png)



*Fig. 3. An illustration of a single “hourglass” module. Each box in the figure corresponds to a residual module as seen in Figure 4. The number of features is consistent across the whole hourglass.*

![img](https://miro.medium.com/max/508/1*tjy99K7waBpTkDhw3AMTQg.png)

*Fig. 4. Left: Residual Module that we use throughout our network.*

## Initialization

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

## hg_module and saccade_module

### initialization

```python
# hg_module, saccade_module: same
class module(nn.Module):
    def __init__(
        self, 
        n, dims, modules, 
        make_up_layer, make_pool_layer, make_hg_layer, 
        make_low_layer,
        make_hg_layer_revr, make_unpool_layer, 
        make_merge_layer
    ):
        super(module, self).__init__()

        curr_mod, next_mod = modules[0], modules[1]
        curr_dim, next_dim = dims[0], dims[1]

        self.n = n
        self.up1 = make_up_layer(curr_dim, curr_dim, curr_mod)
        self.max1 = make_pool_layer(curr_dim)
        self.low1 = make_hg_layer(curr_dim, next_dim, curr_mod)
        self.low2 = (
            module(
                n=n - 1,
                dims=dims[1:],
                modules=modules[1:],
                ...
            )
            if n > 1
            else make_low_layer(next_dim, next_dim, next_mod)
        )
        self.low3 = make_hg_layer_revr(next_dim, curr_dim, curr_mod)
        self.up2 = make_unpool_layer(curr_dim)
        self.merg = make_merge_layer(curr_dim)
```

### forward

#### CornerNet, CornerNet Squeeze

```python
def forward(self, x):
	# CornerNet
	# x.shape: torch.Size([4, 256, 128, 128])
    
    # CornerNet Squeeze
	# x.shape: torch.Size([13, 256, 64, 64])
                    
    max1 = self.max1(x)
    low1 = self.low1(max1)
    low2 = self.low2(low1)
    low3 = self.low3(low2)
    up2 = self.up2(low3)
        
    # CornerNet
	# max1.shape: torch.Size([4, 256, 128, 128])
	# low1.shape: torch.Size([4, 256, 64, 64])
	# low2.shape: torch.Size([4, 256, 64, 64])
	# low3.shape: torch.Size([4, 256, 64, 64])
	# up2.shape: torch.Size([4, 256, 128, 128])
    
    # CornerNet Squeeze
	# max1.shape: torch.Size([13, 256, 64, 64])
	# low1.shape: torch.Size([13, 256, 32, 32])
	# low2.shape: torch.Size([13, 256, 32, 32])
	# low3.shape: torch.Size([13, 256, 32, 32])
	# up2.shape: torch.Size([13, 256, 64, 64])                    
    
    up1 = self.up1(x)
    merg = self.merg(up1, up2)
    
	# CornerNet
	# x.shape: torch.Size([4, 256, 128, 128])
	# up1.shape: torch.Size([4, 256, 128, 128])
	# merg.shape: torch.Size([4, 256, 128, 128])

	# CornerNet Squeeze
	# x.shape: torch.Size([13, 256, 64, 64])
	# up1.shape: torch.Size([13, 256, 64, 64])
	# merg.shape: torch.Size([13, 256, 64, 64])
    
    return merg
```

#### CornerNet Saccade

```python
def forward(self, x):
    max1 = self.max1(x)
    low1 = self.low1(max1)

    if self.n > 1:
        low2, mergs = self.low2(low1)
    else:
        low2, mergs = self.low2(low1), []

    low3 = self.low3(low2)
    up2 = self.up2(low3)
    
    # x.shape: torch.Size([4, 256, 64, 64])
    # max1.shape: torch.Size([4, 256, 64, 64])
    # low1.shape: torch.Size([4, 384, 32, 32])
    # low2.shape: torch.Size([4, 384, 32, 32])
	# mergs.shape = [torch.Size([4, 384, 16, 16]), torch.Size([4, 384, 32, 32])]
    # low3.shape: torch.Size([4, 256, 32, 32])
    # up2.shape: torch.Size([4, 256, 64, 64])
    
    up1 = self.up1(x)
    merg = self.merg(up1, up2)
    # x.shape: torch.Size([4, 256, 64, 64])
    # merg.shape: torch.Size([4, 256, 64, 64])
    
    mergs.append(merg)
	# mergs.shape = [
	# 	torch.Size([4, 384, 16, 16]), 
	# 	torch.Size([4, 384, 32, 32]), 
	# 	torch.Size([4, 256, 64, 64])
	# ]
    
    return merg, mergs
```

## Compare make layers

- make_up_layer, make_pool_layer, make_hg_layer
- make_low_layer
- make_hg_layer_revr, make_unpool_layer
- make_merge_layer

### `self.up1 = make_up_layer(curr_dim, curr_dim, curr_mod)`

```python
# CornerNet, CornerNet Saccade
# x.shape: torch.Size([4, 256, 128, 128])
up1 = self.up1(x)
# up1.shape: torch.Size([4, 256, 128, 128])

make_up_layer = _make_layer
def _make_layer(inp_dim=256, out_dim=256, modules=2):
    layers = [residual(inp_dim=256, out_dim=256, k=3, stride=1)]
    layers += [residual(out_dim=256, out_dim=256, k=3, stride=1) for _ in range(1, 2)]
    return nn.Sequential(*layers)

# CornerNet Squeeze
# x.shape: torch.Size([13, 256, 64, 64])
up1 = self.up1(x)
# up1.shape: torch.Size([13, 256, 64, 64])

# replace residual with fire_module
make_up_layer = make_layer
def make_layer(inp_dim=256, out_dim=256, modules=2):
    layers = [fire_module(inp_dim=256, out_dim=256, sr=2, stride=1)]
    layers += [fire_module(out_dim=256, out_dim=256, sr=2, stride=1) for _ in range(1, 2)]
    return nn.Sequential(*layers)
```

### `self.max1 = make_pool_layer(curr_dim)`

```python
# CornerNet, CornerNet Saccade
# x.shape: torch.Size([4, 256, 128, 128])
max1 = self.max1(x)
# max1.shape: torch.Size([4, 256, 128, 128])
make_pool_layer = make_pool_layer

# CornerNet Squeeze
# x.shape: torch.Size([13, 256, 64, 64])
max1 = self.max1(x)
# max1.shape: torch.Size([13, 256, 64, 64])
make_pool_layer = make_pool_layer

def make_pool_layer(dim=256):
    return nn.Sequential()
```

### `self.low1 = make_hg_layer(curr_dim, next_dim, curr_mod)`

Downsampling and prepare input for recursive module: `curr_dim -> next_dim`

```python
# CornerNet, CornerNet Saccade
# max1.shape: torch.Size([4, 256, 128, 128])
low1 = self.low1(max1)
# low1.shape: torch.Size([4, 256, 64, 64])

make_hg_layer = make_hg_layer
def make_hg_layer(inp_dim=256, out_dim=256, modules=2):
    layers  = [residual(inp_dim=256, out_dim=256, k=3, stride=2)]
    layers += [residual(out_dim=256, out_dim=256, k=3, stride=1) for _ in range(1, 2)]
    return nn.Sequential(*layers)

# CornerNet Squeeze
# max1.shape: torch.Size([13, 256, 64, 64])
low1 = self.low1(max1)
# low1.shape: torch.Size([13, 256, 32, 32])

# replace residual with fire_module
make_hg_layer = make_hg_layer
def make_hg_layer(inp_dim=256, out_dim=256, modules=2):
    layers = [fire_module(inp_dim=256, out_dim=256, stride=2, sr=2)]
    layers += [fire_module(out_dim=256, out_dim=256, stride=1, sr=2) for _ in range(1, 2)]
    return nn.Sequential(*layers)
```

### `self.low2 = hg_module(self.n - 1)`

```python
# CornerNet
# low1.shape: torch.Size([4, 256, 64, 64])
low2 = self.low2(low1)
# low2.shape: torch.Size([4, 256, 64, 64])

# CornerNet Saccade
low2, mergs = self.low2(low1)
# low2.shape: torch.Size([4, 384, 32, 32])
# mergs.shape = [torch.Size([4, 384, 16, 16]), torch.Size([4, 384, 32, 32])]

# CornerNet Squeeze
# low1.shape: torch.Size([13, 256, 32, 32])
low2 = self.low2(low1)
# low2.shape: torch.Size([13, 256, 32, 32])
```

### `self.low3 = make_hg_layer_revr(next_dim, curr_dim, curr_mod)`

Convert dim back to current dim: `next_dim -> curr_dim`

```python
# CornerNet, CornerNet Saccade
# low2.shape: torch.Size([4, 256, 64, 64])
low3 = self.low3(low2)
# low3.shape: torch.Size([4, 256, 64, 64])

make_hg_layer_revr = _make_layer_revr
def _make_layer_revr(inp_dim=256, out_dim=256, modules=2):
    layers  = [residual(inp_dim=256, inp_dim=256, k=3, stride=1) for _ in range(modules - 1)]
    layers += [residual(inp_dim=256, out_dim=256, k=3, stride=1)]
    return nn.Sequential(*layers)

# CornerNet Squeeze
# low2.shape: torch.Size([13, 256, 32, 32])
low3 = self.low3(low2)
# low3.shape: torch.Size([13, 256, 32, 32])

# replace residual with fire_module
make_hg_layer_revr = make_layer_revr
def make_layer_revr(inp_dim=256, out_dim=256, modules=2):
    layers = [fire_module(inp_dim=256, inp_dim=256, stride=1, sr=2) for _ in range(2 - 1)]
    layers += [fire_module(inp_dim=256, out_dim=256, stride=1, sr=2)]
    return nn.Sequential(*layers)
```

### `self.up2 = make_unpool_layer(curr_dim)`

Upsampling to match the resolution of the current layer.

```python
# CornerNet, CornerNet Saccade
# low3.shape: torch.Size([4, 256, 64, 64])
up2 = self.up2(low3)
# up2.shape: torch.Size([4, 256, 128, 128]) 

make_unpool_layer = _make_unpool_layer
def _make_unpool_layer(dim=256):
    return upsample(scale_factor=2)

class upsample(nn.Module):
    def __init__(self, scale_factor):
        super(upsample, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=self.scale_factor)

# CornerNet Squeeze
# low3.shape: torch.Size([13, 256, 32, 32])
up2 = self.up2(low3)
# up2.shape: torch.Size([13, 256, 64, 64])

make_unpool_layer = make_unpool_layer
def make_unpool_layer(dim):
    return nn.ConvTranspose2d(
        dim=256, dim=256, kernel_size=4, stride=2, padding=1
    )
```

### `self.merg = make_merge_layer(curr_dim)`

Residual connection.

```python
# CornerNet, CornerNet Saccade
# up1.shape: torch.Size([4, 256, 128, 128])
# up2.shape: torch.Size([13, 256, 64, 64])
merg = self.merg(up1, up2)
# merg.shape: torch.Size([4, 256, 128, 128]) 
make_merge_layer = _make_merge_layer

# CornerNet Squeeze
# up1.shape: torch.Size([13, 256, 64, 64])
# up2.shape: torch.Size([13, 256, 64, 64])
merg = self.merg(up1, up2)
# merg.shape: torch.Size([13, 256, 64, 64]) 

make_merge_layer = _make_merge_layer
def _make_merge_layer(dim=256):
    return merge()
```

## Shape changes through forward pass

### CornerNet: `x-max1-low1-low2-low3-up2 + x-up1 = merg`

#### n = 5

```python
# x.shape: torch.Size([1, 256, 128, 128])
# max1.shape: torch.Size([1, 256, 128, 128])
# low1.shape: torch.Size([1, 256, 64, 64])
# low2.shape: torch.Size([1, 256, 64, 64])
# low3.shape: torch.Size([1, 256, 64, 64])
# up2.shape: torch.Size([1, 256, 128, 128])

# x.shape: torch.Size([1, 256, 128, 128])
# up1.shape: torch.Size([1, 256, 128, 128])

# merg.shape: torch.Size([1, 256, 128, 128]) 
```

#### n = 4

```python
# x = hg_module(n=5).low1

# x.shape: torch.Size([1, 256, 64, 64])
# max1.shape: torch.Size([1, 256, 64, 64])
# low1.shape: torch.Size([1, 384, 32, 32])
# low2.shape: torch.Size([1, 384, 32, 32])
# low3.shape: torch.Size([1, 256, 32, 32])
# up2.shape: torch.Size([1, 256, 64, 64])

# x.shape: torch.Size([1, 256, 64, 64])
# up1.shape: torch.Size([1, 256, 64, 64])

# merg.shape: torch.Size([1, 256, 64, 64])
```

#### n = 3

```python
# x = hg_module(n=4).low1

# x.shape: torch.Size([1, 384, 32, 32])
# max1.shape: torch.Size([1, 384, 32, 32]) 
# low1.shape: torch.Size([1, 384, 16, 16])
# low2.shape: torch.Size([1, 384, 16, 16])
# low3.shape: torch.Size([1, 384, 16, 16])
# up2.shape: torch.Size([1, 384, 32, 32])

# x.shape: torch.Size([1, 384, 32, 32])
# up1.shape: torch.Size([1, 384, 32, 32]) 

# merg.shape: torch.Size([1, 384, 32, 32])
```

#### n = 2

```python
# x = hg_module(n=3).low1

# x.shape: torch.Size([1, 384, 16, 16])
# low1.shape: torch.Size([1, 384, 8, 8])
# low2.shape: torch.Size([1, 384, 8, 8])
# low3.shape: torch.Size([1, 384, 8, 8])
# up2.shape: torch.Size([1, 384, 16, 16])

# x.shape: torch.Size([1, 384, 16, 16])
# up1.shape: torch.Size([1, 384, 16, 16]) 

# merg.shape: torch.Size([1, 384, 16, 16])
```

#### n = 1

```python
# x = hg_module(n=2).low1

# x.shape: torch.Size([1, 384, 8, 8])
# max1.shape: torch.Size([1, 384, 8, 8])
# low1.shape: torch.Size([1, 512, 4, 4])
# low2.shape: torch.Size([1, 512, 4, 4]) 
# low3.shape: torch.Size([1, 384, 4, 4])
# up2.shape: torch.Size([1, 384, 8, 8])

# x.shape: torch.Size([1, 384, 8, 8])
# up1.shape: torch.Size([1, 384, 8, 8])

# merg.shape: torch.Size([1, 384, 8, 8])
```

### CornerNet Saccade: `x-max1-low1-low2-low3-up2 + x-up1 = merg`

#### n = 3

```python
# x.shape: torch.Size([4, 256, 64, 64])
# max1.shape: torch.Size([4, 256, 64, 64])
# low1.shape: torch.Size([4, 384, 32, 32])
# low2.shape: torch.Size([4, 384, 32, 32])
# mergs.shape = [torch.Size([4, 384, 16, 16]), torch.Size([4, 384, 32, 32])]
# low3.shape: torch.Size([4, 256, 32, 32])
# up2.shape: torch.Size([4, 256, 64, 64])

# x.shape: torch.Size([4, 256, 64, 64])
# up1.shape: torch.Size([4, 256, 64, 64])

# merg.shape: torch.Size([4, 256, 64, 64])
# mergs.shape = [torch.Size([4, 384, 16, 16]), torch.Size([4, 384, 32, 32]), torch.Size([4, 256, 64, 64])]
```

#### n = 2

```python
# x = saccade_module(n=3).low1

# x.shape: torch.Size([4, 384, 32, 32])
# max1.shape: torch.Size([4, 384, 32, 32]) 
# low1.shape: torch.Size([4, 384, 16, 16])
# low2.shape: torch.Size([4, 384, 16, 16])
# mergs.shape = [torch.Size([4, 384, 16, 16])]
# low3.shape: torch.Size([4, 384, 16, 16])
# up2.shape: torch.Size([4, 384, 32, 32])

# x.shape: torch.Size([4, 384, 32, 32])
# up1.shape: torch.Size([4, 384, 32, 32]) 

# merg.shape: torch.Size([4, 384, 32, 32])
# mergs.shape = [torch.Size([4, 384, 16, 16]), torch.Size([4, 384, 32, 32])]
```

#### n = 1

```python
# x = saccade_module(n=2).low1

# x.shape: torch.Size([4, 384, 16, 16])
# max1.shape: torch.Size([4, 384, 16, 16]) 
# low1.shape: torch.Size([4, 512, 8, 8])
# low2.shape: torch.Size([4, 512, 8, 8])
# mergs.shape = []
# low3.shape: torch.Size([4, 384, 8, 8])
# up2.shape: torch.Size([4, 384, 16, 16])

# x.shape: torch.Size([4, 384, 16, 16])
# up1.shape: torch.Size([4, 384, 16, 16]) 

# merg.shape: torch.Size([4, 384, 16, 16])
# mergs.shape = [torch.Size([4, 384, 16, 16])]
```

