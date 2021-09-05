# CornerNet Models

## Model overview

### CornerNet and CornerNet Squeeze

```python
class model(hg_net):
    def _pred_mod(self, dim):
        pass

    def _merge_mod(self):
		pass
    
    def __init__(self):
        stacks = 2
        hg_mods = nn.ModuleList([hg_module(...) for _ in range(stacks)])
        hgs = hg(pre, hg_mods, cnvs, inters, cnvs_, inters_)
        
        # heatmaps initialization
        for tl_heat, br_heat in zip(tl_heats, br_heats):
    		torch.nn.init.constant_(tl_heat[-1].bias, -2.19)
    		torch.nn.init.constant_(br_heat[-1].bias, -2.19)
		
        # hg_net
        super(model, self).__init__(
            # hourglass backbone
            hgs,
            # corner pooling
            tl_modules, br_modules,
            # heatmaps
            tl_heats, br_heats,
            # embeddings
            tl_tags, br_tags,
            # offsets
            tl_offs, br_offs,
        )
        self.loss = CornerNet_Loss(...)
```

### CornerNet Saccade

```python
class model(saccade_net):
    def _pred_mod(self, dim):
        pass

    def _merge_mod(self):
		pass
    
    def __init__(self):
        stacks = 3
        hg_mods = nn.ModuleList([saccade_module(...) for _ in range(stacks)])
        hgs = saccade(pre, hg_mods, cnvs, inters, cnvs_, inters_)
        
        # heatmaps initialization
        for tl_heat, br_heat in zip(tl_heats, br_heats):
    		torch.nn.init.constant_(tl_heat[-1].bias, -2.19)
    		torch.nn.init.constant_(br_heat[-1].bias, -2.19)
        
        # attentions modules
        att_mods = nn.ModuleList(
            [
                nn.ModuleList([...])
                for _ in range(stacks)
            ]
        )
        
        # attentions initialization
        for att_mod in att_mods:
            for att in att_mod:
                torch.nn.init.constant_(att[-1].bias, -2.19)
		
        # saccade_net
        super(model, self).__init__(
            # hourglass backbone
            hgs,
            # corner pooling
            tl_modules, br_modules,
            # heatmaps
            tl_heats, br_heats,
            # embeddings
            tl_tags, br_tags,
            # offsets
            tl_offs, br_offs,
            # attentions
            att_mods,
        )
        self.loss = CornerNet_Saccade_Loss(...)
```

## `_pred_mod`

### CornerNet and CornerNet Saccade

```python
def _pred_mod(self, dim=80):
    return nn.Sequential(
        convolution(
            k=3, 
            inp_dim=256, 
            out_dim=256, 
            stride=1, 
            with_bn=False
        ),
        nn.Conv2d(
            in_channels=256, 
            out_channels=80, 
            kernel_size=(1, 1)
        )
    )
```

### CornerNet Squeeze

```python
def _pred_mod(self, dim=80):
    return nn.Sequential(
        convolution(
            k=1, 
            inp_dim=256, 
            out_dim=256, 
            stride=1, 
            with_bn=False
        ),
        nn.Conv2d(
            in_channels=256, 
            out_channels=80, 
            kernel_size=(1, 1)
        )
    )
```

## `_merge_mod`

```python
def _merge_mod(self):
    return nn.Sequential(
        nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=(1, 1), bias=False
        ),
        nn.BatchNorm2d(256),
    )
```

## Base Networks

### CornerNet and CornerNet Squeeze: `hg_net`

```python
class hg_net(nn.Module):
    def __init__(
        self,
        # hourglass backbone
        hgs,
        # corner pooling
        tl_modules, br_modules,
        # heatmaps
        tl_heats, br_heats,
        # embeddings
        tl_tags, br_tags,
        # offsets
        tl_offs, br_offs,
    ):
        pass

    def _train(self, *xs):
        # input
        image = xs[0]
        
        # hourglass backbone
        cnvs = self.hg(image)
        
        # corner pooling
        tl_modules = [tl_mod_(cnv) for tl_mod_, cnv in zip(self.tl_modules, cnvs)]
        br_modules = [br_mod_(cnv) for br_mod_, cnv in zip(self.br_modules, cnvs)]

        # heatmaps
        tl_heats = [
            tl_heat_(tl_mod) for tl_heat_, tl_mod in zip(self.tl_heats, tl_modules)
        ]
        br_heats = [
            br_heat_(br_mod) for br_heat_, br_mod in zip(self.br_heats, br_modules)
        ]

        # embeddings
        tl_tags = [tl_tag_(tl_mod) for tl_tag_, tl_mod in zip(self.tl_tags, tl_modules)]
        br_tags = [br_tag_(br_mod) for br_tag_, br_mod in zip(self.br_tags, br_modules)]

        # offsets
        tl_offs = [tl_off_(tl_mod) for tl_off_, tl_mod in zip(self.tl_offs, tl_modules)]
        br_offs = [br_off_(br_mod) for br_off_, br_mod in zip(self.br_offs, br_modules)]
        return [tl_heats, br_heats, tl_tags, br_tags, tl_offs, br_offs]

    def _test(self, *xs, **kwargs):
        # input
        image = xs[0]
        
        # hourglass backbone
        cnvs = self.hg(image)

        # corner pooling
        tl_mod = self.tl_modules[-1](cnvs[-1])
        br_mod = self.br_modules[-1](cnvs[-1])
		
        # heatmaps
        tl_heat, br_heat = self.tl_heats[-1](tl_mod), self.br_heats[-1](br_mod)
        
        # embeddings
        tl_tag, br_tag = self.tl_tags[-1](tl_mod), self.br_tags[-1](br_mod)
        
        # offsets
        tl_off, br_off = self.tl_offs[-1](tl_mod), self.br_offs[-1](br_mod)

        outs = [tl_heat, br_heat, tl_tag, br_tag, tl_off, br_off]
        return self._decode(*outs, **kwargs), tl_heat, br_heat, tl_tag, br_tag

    def forward(self, *xs, test=False, **kwargs):
        if not test:
            return self._train(*xs, **kwargs)
        return self._test(*xs, **kwargs)
```

### CornerNet Saccade: `saccade_net`

```python
class saccade_net(nn.Module):
    def __init__(
        self,
        # hourglass backbone
        hgs,
        # corner pooling
        tl_modules, br_modules,
        # heatmaps
        tl_heats, br_heats,
        # embeddings
        tl_tags, br_tags,
        # offsets
        tl_offs, br_offs,
        # attentions
        att_modules,
        up_start=0,
    ):
        pass

    def _train(self, *xs):
        # input
        image = xs[0]
        
        # hourglass backbone
        cnvs, ups = self.hg(image)
        ups = [up[self.up_start :] for up in ups]
        
        # corner pooling
        tl_modules = [tl_mod_(cnv) for tl_mod_, cnv in zip(self.tl_modules, cnvs)]
        br_modules = [br_mod_(cnv) for br_mod_, cnv in zip(self.br_modules, cnvs)]

        # heatmaps
        tl_heats = [
            tl_heat_(tl_mod) for tl_heat_, tl_mod in zip(self.tl_heats, tl_modules)
        ]
        br_heats = [
            br_heat_(br_mod) for br_heat_, br_mod in zip(self.br_heats, br_modules)
        ]

        # embeddings
        tl_tags = [tl_tag_(tl_mod) for tl_tag_, tl_mod in zip(self.tl_tags, tl_modules)]
        br_tags = [br_tag_(br_mod) for br_tag_, br_mod in zip(self.br_tags, br_modules)]

        # offsets
        tl_offs = [tl_off_(tl_mod) for tl_off_, tl_mod in zip(self.tl_offs, tl_modules)]
        br_offs = [br_off_(br_mod) for br_off_, br_mod in zip(self.br_offs, br_modules)]
        
        # attentions
        atts = [
            [att_mod_(u) for att_mod_, u in zip(att_mods, up)]
            for att_mods, up in zip(self.att_modules, ups)
        ]
        return [tl_heats, br_heats, tl_tags, br_tags, tl_offs, br_offs, atts]

    def _test(self, *xs, no_att=False, **kwargs):
        # input
        image = xs[0]
        
        # hourglass backbone
        cnvs, ups = self.hg(image)
        ups = [up[self.up_start :] for up in ups]
        
        # attentions
        if not no_att:
            atts = [att_mod_(up) for att_mod_, up in zip(self.att_modules[-1], ups[-1])]
            atts = [torch.sigmoid(att) for att in atts]

        # corner pooling
        tl_mod = self.tl_modules[-1](cnvs[-1])
        br_mod = self.br_modules[-1](cnvs[-1])
		
        # heatmaps
        tl_heat, br_heat = self.tl_heats[-1](tl_mod), self.br_heats[-1](br_mod)
        
        # embeddings
        tl_tag, br_tag = self.tl_tags[-1](tl_mod), self.br_tags[-1](br_mod)
        
        # offsets
        tl_off, br_off = self.tl_offs[-1](tl_mod), self.br_offs[-1](br_mod)

        outs = [tl_heat, br_heat, tl_tag, br_tag, tl_off, br_off]
        if not no_att:
            return self._decode(*outs, **kwargs), atts
        else:
            return self._decode(*outs, **kwargs)

    def forward(self, *xs, test=False, **kwargs):
        if not test:
            return self._train(*xs, **kwargs)
        return self._test(*xs, **kwargs)
```

## Forward

### Input

```python
image = xs[0]
```

#### train

```python
# image.shape:
# CornerNet: torch.Size([4, 3, 511, 511])
# CornerNet Squeeze: torch.Size([13, 3, 511, 511])
# CornerNet Saccade: torch.Size([4, 3, 255, 255])
```

#### test

```python
# image.shape:
# CornerNet: torch.Size([2, 3, 511, 767])
# CornerNet Squeeze: torch.Size([2, 3, 511, 767])
# CornerNet Saccade: torch.Size([4, 3, 255, 255])
```

---

### Hourglass backbone

```python
# CornerNet, CornerNet Squeeze
cnvs = self.hg(image)

# CornerNet Saccade
cnvs, ups = self.hg(image)
ups = [up[self.up_start :] for up in ups]
```

#### train

```python
# cnvs.shape:
# CornerNet: [torch.Size([4, 256, 128, 128])] * stacks
# CornerNet Squeeze: [torch.Size([13, 256, 64, 64])] * stacks
# CornerNet Saccade: [torch.Size([4, 256, 64, 64])] * stacks

# CornerNet Saccade:
# ups.shape: [
#     torch.Size([4, 384, 16, 16]),
#     torch.Size([4, 384, 32, 32]),
#     torch.Size([4, 256, 64, 64])
# ] * stacks
```

#### test

```python
# cnvs.shape:
# CornerNet: [torch.Size([2, 256, 128, 192])] * stacks
# CornerNet Squeeze: [torch.Size([13, 256, 64, 96])] * stacks
# CornerNet Saccade: [torch.Size([2, 256, 64, 64])] * stacks

# CornerNet Saccade:
# ups.shape: [
#     torch.Size([4, 384, 16, 16]),
#     torch.Size([4, 384, 32, 32]),
#     torch.Size([4, 256, 64, 64])
# ] * stacks
```

---

### Corner pooling

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

#### train

```python
tl_modules = [tl_mod_(cnv) for tl_mod_, cnv in zip(self.tl_modules, cnvs)]
br_modules = [br_mod_(cnv) for br_mod_, cnv in zip(self.br_modules, cnvs)]

# tl_modules.shape: 
# CornerNet: [torch.Size([4, 256, 128, 128])] * stacks
# CornerNet Squeeze: [torch.Size([13, 256, 64, 64])] * stacks
# CornerNet Saccade: [torch.Size([4, 256, 64, 64])] * stacks
```

#### test

```python
tl_mod = self.tl_modules[-1](cnvs[-1])
br_mod = self.br_modules[-1](cnvs[-1])

# tl_mod.shape: 
# CornerNet: torch.Size([2, 256, 128, 192])
# CornerNet Squeeze: torch.Size([1, 256, 64, 96])
# CornerNet Saccade: torch.Size([2, 256, 64, 64])
```

---

### Heatmaps

```python
tl_heats = nn.ModuleList([self._pred_mod(80) for _ in range(stacks)])
br_heats = nn.ModuleList([self._pred_mod(80) for _ in range(stacks)])
```

#### train

```python
tl_heats = [
    tl_heat_(tl_mod) for tl_heat_, tl_mod in zip(self.tl_heats, tl_modules)
]
br_heats = [
    br_heat_(br_mod) for br_heat_, br_mod in zip(self.br_heats, br_modules)
]

# tl_heats.shape: 
# CornerNet: [torch.Size([4, 80, 128, 128])] * stacks
# CornerNet Squeeze: [torch.Size([13, 80, 64, 64])] * stacks
# CornerNet Saccade: [torch.Size([4, 80, 64, 64])] * stacks
```

#### test

```python
tl_heat, br_heat = self.tl_heats[-1](tl_mod), self.br_heats[-1](br_mod)

# tl_heat.shape: 
# CornerNet: torch.Size([2, 80, 128, 192])
# CornerNet Squeeze: torch.Size([1, 80, 64, 96])
# CornerNet Saccade: torch.Size([2, 80, 64, 64])      
```

---

### Embeddings

```python
tl_tags = nn.ModuleList([self._pred_mod(1) for _ in range(stacks)])
br_tags = nn.ModuleList([self._pred_mod(1) for _ in range(stacks)])
```

#### train

```python
tl_tags = [tl_tag_(tl_mod) for tl_tag_, tl_mod in zip(self.tl_tags, tl_modules)]
br_tags = [br_tag_(br_mod) for br_tag_, br_mod in zip(self.br_tags, br_modules)]

# tl_tags.shape: 
# CornerNet: [torch.Size([4, 1, 128, 128])] * stacks
# CornerNet Squeeze: [torch.Size([13, 1, 64, 64])] * stacks
# CornerNet Saccade: [torch.Size([4, 1, 64, 64])] * stacks
```

#### test

```python
tl_tag, br_tag = self.tl_tags[-1](tl_mod), self.br_tags[-1](br_mod)

# tl_heat.shape: 
# CornerNet: torch.Size([2, 1, 128, 192])
# CornerNet Squeeze: torch.Size([1, 1, 64, 96])
# CornerNet Saccade: torch.Size([2, 1, 64, 64])      
```

------

### Offsets

```python
tl_offs = nn.ModuleList([self._pred_mod(2) for _ in range(stacks)])
br_offs = nn.ModuleList([self._pred_mod(2) for _ in range(stacks)])
```

#### train

```python
tl_offs = [tl_off_(tl_mod) for tl_off_, tl_mod in zip(self.tl_offs, tl_modules)]
br_offs = [br_off_(br_mod) for br_off_, br_mod in zip(self.br_offs, br_modules)]

# tl_offs.shape: 
# CornerNet: [torch.Size([4, 2, 128, 128])] * stacks
# CornerNet Squeeze: [torch.Size([13, 2, 64, 64])] * stacks
# CornerNet Saccade: [torch.Size([4, 2, 64, 64])] * stacks
```

#### test

```python
tl_off, br_off = self.tl_offs[-1](tl_mod), self.br_offs[-1](br_mod)

# tl_heat.shape: 
# CornerNet: torch.Size([2, 2, 128, 192])
# CornerNet Squeeze: torch.Size([1, 2, 64, 96])
# CornerNet Saccade: torch.Size([2, 2, 64, 64])      
```

------

### Attentions (CornerNet Saccade)

```python
att_mods = nn.ModuleList(
    [
        nn.ModuleList(
            [
                nn.Sequential(
                    convolution(
                        k=3, inp_dim=384, out_dim=256, stride=1, with_bn=False
                    ),
                    nn.Conv2d(
                        in_channels=256, out_channels=1, kernel_size=(1, 1)
                    ),
                ),
                nn.Sequential(
                    convolution(
                        k=3, inp_dim=384, out_dim=256, stride=1, with_bn=False
                    ),
                    nn.Conv2d(
                        in_channels=256, out_channels=1, kernel_size=(1, 1)
                    ),
                ),
                nn.Sequential(
                    convolution(
                        k=3, inp_dim=256, out_dim=256, stride=1, with_bn=False
                    ),
                    nn.Conv2d(
                        in_channels=256, out_channels=1, kernel_size=(1, 1)
                    ),
                ),
            ]
        )
        for _ in range(stacks)
    ]
)
```

#### train

```python
# ups.shape: [
#     torch.Size([4, 384, 16, 16]),
#     torch.Size([4, 384, 32, 32]),
#     torch.Size([4, 256, 64, 64])
# ] * stacks

atts = [
    [att_mod_(u) for att_mod_, u in zip(att_mods, up)]
    for att_mods, up in zip(self.att_modules, ups)
]

# atts.shape: [
#     torch.Size([4, 1, 16, 16]),
#     torch.Size([4, 1, 32, 32]),
#     torch.Size([4, 1, 64, 64])
# ] * stacks
```

#### test

```python
atts = [att_mod_(up) for att_mod_, up in zip(self.att_modules[-1], ups[-1])]
atts = [torch.sigmoid(att) for att in atts]

# atts.shape: [
#     torch.Size([4, 1, 16, 16]),
#     torch.Size([4, 1, 32, 32]),
#     torch.Size([4, 1, 64, 64])
# ]  
```



