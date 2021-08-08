# Model

### initialize

#### backbone

```python
self.inplanes = 64
self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
self.bn1 = nn.BatchNorm2d(64)
self.relu = nn.ReLU(inplace=True)
self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
self.layer1 = self._make_layer(block, 64, layers[0])
self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
```

#### fpn

```python
if block == BasicBlock:
    fpn_sizes = [
        self.layer2[layers[1] - 1].conv2.out_channels,
        self.layer3[layers[2] - 1].conv2.out_channels,
        self.layer4[layers[3] - 1].conv2.out_channels,
    ]
elif block == Bottleneck:
    fpn_sizes = [
        self.layer2[layers[1] - 1].conv3.out_channels,
        self.layer3[layers[2] - 1].conv3.out_channels,
        self.layer4[layers[3] - 1].conv3.out_channels,
    ]
else:
    raise ValueError(f"Block type {block} not understood")

# fpn_sizes: [512, 1024, 2048]
self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])
```

#### others

```python
# classification and regression use separate parameters
self.regressionModel = RegressionModel(256)
self.classificationModel = ClassificationModel(256, num_classes=num_classes)

self.anchors = Anchors()
self.regressBoxes = BBoxTransform()
self.clipBoxes = ClipBoxes()
self.focalLoss = losses.FocalLoss()
```

#### weights initialization

```python
# initialization
# All new conv layers except the final one in the RetinaNet subnets 
# are initialized with bias b = 0 and a Gaussian weight fill with sigma = 0.01. 
for m in self.modules():
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2.0 / n))
    # won't use batchnorm because of small batch size
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

# For the final conv layer of the classification subnet, 
# we set the bias initialization to b = −log((1 − π)/π), 
# where π specifies that at the start of training every anchor 
# should be labeled as foreground with confidence of ∼π.
# sigmoid = 1 / (1 + e(-x)) = 1 / (1 + e(log((1 − π)/π)))
# = 1 / (1 + (1 − π)/π) = π
prior = 0.01
self.classificationModel.output.weight.data.fill_(0)
self.classificationModel.output.bias.data.fill_(
    -math.log((1.0 - prior) / prior)
)

self.regressionModel.output.weight.data.fill_(0)
self.regressionModel.output.bias.data.fill_(0)

# freeze batchnorm because of small batch size
self.freeze_bn()
```

---

### forward

#### backbone

```python
# img_batch: [2, 3, 640, 832]
img_batch, annotations = inputs

# self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
x = self.conv1(img_batch)
x = self.bn1(x)
x = self.relu(x)
# x: [2, 64, 320, 416]
x = self.maxpool(x)
# x: [2, 64, 160, 208]

# resnet50
# out_channels: [256, 512, 1024, 2048]
# strides: [1, 2, 2, 2]
x1 = self.layer1(x)
# x1: [2, 256, 160, 208]
x2 = self.layer2(x1)
# x2: [2, 512, 80, 104]
x3 = self.layer3(x2)
# x3: [2, 1024, 40, 52]
x4 = self.layer4(x3)
# x4: [2, 2048, 20, 26]
```

#### fpn

**Detail:** [feature_pyramid.md](feature_pyramid.md)

```python
# x2: [2, 512, 80, 104]
# x3: [2, 1024, 40, 52]
# x4: [2, 2048, 20, 26]
features = self.fpn([x2, x3, x4])
# features: [P3_x, P4_x, P5_x, P6_x, P7_x]
# P3_x: [2, 256, 80, 104]
# P4_x: [2, 256, 40, 52]
# P5_x: [2, 256, 20, 26]
# P6_x: [2, 256, 10, 13]
# P7_x: [2, 256, 5, 7]
```

#### regressionModel, classificationModel 

**Detail:** [regression_and_classification.md](regression_and_classification.md)

```python
# regression for per feature level
regression_features = [self.regressionModel(feature) for feature in features]
# regression_features: 
# P3_x: [2, 256, 80, 104] -> [2, 74880, 4], 80 * 104 * 9 = 74880
# P4_x: [2, 256, 40, 52] -> [2, 18720, 4], 40 * 52 * 9 = 18720
# P5_x: [2, 256, 20, 26] -> [2, 4680, 4], 20 * 26 * 9 = 4680
# P6_x: [2, 256, 10, 13] -> [2, 1170, 4], 10 * 13 * 9 = 1170
# P7_x: [2, 256, 5, 7] -> [2, 315, 4], 5 * 7 * 9 = 315
regression = torch.cat(regression_features, dim=1)

# classification for per feature level
classification_features = [
    self.classificationModel(feature) for feature in features
]
classification = torch.cat(classification_features, dim=1)
# classification_features: 
# P3_x: [2, 256, 80, 104] -> [2, 74880, 20], 80 * 104 * 9 = 74880
# P4_x: [2, 256, 40, 52] -> [2, 18720, 20], 40 * 52 * 9 = 18720
# P5_x: [2, 256, 20, 26] -> [2, 4680, 20], 20 * 26 * 9 = 4680
# P6_x: [2, 256, 10, 13] -> [2, 1170, 20], 10 * 13 * 9 = 1170
# P7_x: [2, 256, 5, 7] -> [2, 315, 20], 5 * 7 * 9 = 315
```

#### anchors

**Detail:** [anchors.md](anchors.md)

```python
anchors = self.anchors(img_batch)
```

#### focalLoss

**Detail:** [focal_loss.md](focal_loss.md)

```python
self.focalLoss(classification, regression, anchors, annotations)
```