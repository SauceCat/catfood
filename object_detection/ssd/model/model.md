# model

```python
# COCO: cfg['min_dim']=300, cfg['num_classes']=201
# VOC: cfg['min_dim']=300, cfg['num_classes']=21
net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
```

## build_ssd

```python
base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
}

def build_ssd(phase, size=300, num_classes=21):
    base_, extras_, head_ = multibox(
        vgg=vgg(base[str(size)], 3),
        extra_layers=add_extras(extras[str(size)], 1024),
        cfg=mbox[str(size)], 
        num_classes=num_classes
    )
    return SSD(phase, size, base_, extras_, head_, num_classes)
```

# SSD

```python
SSD(
    phase="train", 
    size=300, 
    base_=vgg_layers, 
    extras_=extra_layers, 
    head_=(loc_layers, conf_layers), 
    num_classes=201
)
```

## initialize

```python
class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = (coco, voc)[num_classes == 21]
        self.size = size
```

### PriorBox

Details: [priorbox](priorbox.md)

```python
        self.priorbox = PriorBox(self.cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        # COCO
        # self.priors.shape: torch.Size([8732, 4])
        # (38 ** 2 + 3 ** 2 + 1 ** 2) * 4 + (19 ** 2 + 10 ** 2 + 5 ** 2) * 6 = 8732
```

### VGG base

Details: [vgg16](vgg16.md)

```python
        self.vgg = nn.ModuleList(base)
```

### Extra layers

Details: [extra_layers](extra_layers.md)

```python
        self.extras = nn.ModuleList(extras)
```

### Prediction heads

Details: [multibox](multibox.md)

```python
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
```

### L2Norm

```python
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)

class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight,self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        x = torch.div(x, norm)
        # learn the weight
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out
```

### Inference

Details: [detect](../test/detect.md)

```python
        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(
                num_classes, 
                bkg_label=0, 
                top_k=200, 
                conf_thresh=0.01, 
                nms_thresh=0.45
            )
```

## forward

```python
    def forward(self, x):
        """Applies network layers and ops on input image(s) x.
        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].
        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]
            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()
        
        # apply vgg up to conv4_3 relu
        # vgg 17-22: 256->512->512->512
        # Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # ReLU(inplace=True)
        # Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # ReLU(inplace=True)
        # vgg[21]: Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # ReLU(inplace=True)
        for k in range(23):
            x = self.vgg[k](x)

        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        # 33-34: conv7, 1024->1024
        # vgg[-2]: Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
        # ReLU(inplace=True)
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            # only keep heads, extra_layers[1::2]
            if k % 2 == 1:
                sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
            
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        
        # inference
        if self.phase == "test":
            output = self.detect(
                # loc preds
                loc.view(loc.size(0), -1, 4),
                # conf preds
                self.softmax(
                    conf.view(conf.size(0), -1, self.num_classes)
                ),
                # default boxes
                self.priors.type(type(x.data))                  
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output
```

