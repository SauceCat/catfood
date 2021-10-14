# Model

## initialize

```python
class ResNet(nn.Module):
    def __init__(self, num_classes, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
```

### ResNet backbone

Details: [resnet](resnet.md)

```python
        # ResNet backbone
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if block == BasicBlock:
            fpn_sizes = [
                self.layer2[layers[1] - 1].conv2.out_channels, 
                self.layer3[layers[2] - 1].conv2.out_channels,
                self.layer4[layers[3] - 1].conv2.out_channels
            ]
        elif block == Bottleneck:
            fpn_sizes = [
                self.layer2[layers[1] - 1].conv3.out_channels, 
                self.layer3[layers[2] - 1].conv3.out_channels,
                self.layer4[layers[3] - 1].conv3.out_channels
            ]
        else:
            raise ValueError(f"Block type {block} not understood")
```

```python
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # match channel dimension
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, 
                    planes * block.expansion,
                    kernel_size=1, 
                    stride=stride, 
                    bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        # update input shape to next layer
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
```

### Feature Pyramid

Details: [feature_pyramid](feature_pyramid.md)

```python
        # fpn
        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])
```

### Box classification and regression

Details: [classification_and_regression](classification_and_regression.md)

```python
        # classification and regression use separate parameters
        self.regressionModel = RegressionModel(256)
        self.classificationModel = ClassificationModel(256, num_classes=num_classes)
```

### Anchors

Details: [anchors](anchors.md)

```python
        self.anchors = Anchors()
```

### regressBoxes and clipBoxes

Details: [regress_and_clip_boxes](regress_and_clip_boxes.md)

```python
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()
```

### FocalLoss

Details: [focal_loss](focal_loss.md)

```python
        # loss
        self.focalLoss = losses.FocalLoss()
```

### Module initialization

All new conv layers except the final one in the RetinaNet subnets are initialized with bias `b = 0` and a Gaussian weight fill with `sigma = 0.01`. 

```python
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            # won't use batchnorm because of small batch size
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
```

For the final conv layer of the classification subnet, we set the bias initialization to `b = −log((1 − π)/π)`, where `π` specifies that at the start of training every anchor should be labeled as foreground with confidence of `∼π`.

```python
        # sigmoid = 1 / (1 + e(-x)) = 1 / (1 + e(log((1 − π)/π)))
        # = 1 / (1 + (1 − π)/π) = π
        prior = 0.01
        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))

        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)
```

### freeze batchnorm because of small batch size

```python
        self.freeze_bn()
```

```python
    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
```

## forward

### input

```python
    def forward(self, inputs):
        if self.training:
            # img_batch: [2, 3, 640, 832]
            img_batch, annotations = inputs
        else:
            img_batch = inputs
```

### downsample twice

```python
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        x = self.conv1(img_batch)
        x = self.bn1(x)
        x = self.relu(x)
        # x: [2, 64, 320, 416]
        x = self.maxpool(x)
        # x: [2, 64, 160, 208]
```

### backbone

```python
        # example: resnet50
        # out_channels: [256, 512, 1024, 2048]
        # strides: [1, 2, 2, 2]
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        # x1: [2, 256, 160, 208]
        # x2: [2, 512, 80, 104]
        # x3: [2, 1024, 40, 52]
        # x4: [2, 2048, 20, 26]
```

### feature pyramid

```python
        features = self.fpn([x2, x3, x4])
        # features: [P3_x, P4_x, P5_x, P6_x, P7_x]
        # P3_x: [2, 256, 80, 104]
        # P4_x: [2, 256, 40, 52]                #for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # P5_x: [2, 256, 20, 26]
        # P6_x: [2, 256, 10, 13]
        # P7_x: [2, 256, 5, 7]
```

### regression

```python
        regression = torch.cat([
            self.regressionModel(feature) 
            for feature in features
        ], dim=1)
        # regression_features: 
        # P3_x: [2, 256, 80, 104] -> [2, 74880, 4], 80 * 104 * 9 = 74880
        # P4_x: [2, 256, 40, 52] -> [2, 18720, 4], 40 * 52 * 9 = 18720
        # P5_x: [2, 256, 20, 26] -> [2, 4680, 4], 20 * 26 * 9 = 4680
        # P6_x: [2, 256, 10, 13] -> [2, 1170, 4], 10 * 13 * 9 = 1170
        # P7_x: [2, 256, 5, 7] -> [2, 315, 4], 5 * 7 * 9 = 315
```

### classification

```python
        classification = torch.cat([
            self.classificationModel(feature) 
            for feature in features
        ], dim=1)
        # classification_features: 
        # P3_x: [2, 256, 80, 104] -> [2, 74880, 20], 80 * 104 * 9 = 74880
        # P4_x: [2, 256, 40, 52] -> [2, 18720, 20], 40 * 52 * 9 = 18720
        # P5_x: [2, 256, 20, 26] -> [2, 4680, 20], 20 * 26 * 9 = 4680
        # P6_x: [2, 256, 10, 13] -> [2, 1170, 20], 10 * 13 * 9 = 1170
        # P7_x: [2, 256, 5, 7] -> [2, 315, 20], 5 * 7 * 9 = 315
```

### anchors

```python
        anchors = self.anchors(img_batch)
```

### focal loss (training)

```python
        if self.training:
            return self.focalLoss(
                classification, regression, anchors, annotations
            )
```

### inference (test)

```python
        else:
            # transformed anchors = anchors + regression prediction
            transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

            finalScores = torch.Tensor([]).cuda()
            finalAnchorBoxesIndexes = torch.Tensor([]).long().cuda()
            finalAnchorBoxesCoordinates = torch.Tensor([]).cuda()
            
            # loop through each category
            for i in range(classification.shape[2]):
                # classification score > 0.05
                # scores: [1, 99765] -> [99765 (total number of anchor boxes)]
                scores = torch.squeeze(classification[:, :, i])
                scores_over_thresh = (scores > 0.05)
                if scores_over_thresh.sum() == 0:
                    # no boxes to NMS, just continue
                    continue

                scores = scores[scores_over_thresh]
                anchorBoxes = torch.squeeze(transformed_anchors)
                # anchors with classification score over threshold
                anchorBoxes = anchorBoxes[scores_over_thresh]
                
                # nms on selected anchors
                # from torchvision.ops import nms
                anchors_nms_idx = nms(anchorBoxes, scores, 0.5)
                
                # classification scores
                finalScores = torch.cat(
                    (finalScores, scores[anchors_nms_idx])
                )
                # image index
                finalAnchorBoxesIndexesValue = torch.tensor(
                    [i] * anchors_nms_idx.shape[0]
                ).cuda()
                finalAnchorBoxesIndexes = torch.cat(
                    (finalAnchorBoxesIndexes, finalAnchorBoxesIndexesValue)
                )
                # anchors coordinates
                finalAnchorBoxesCoordinates = torch.cat(
                    (finalAnchorBoxesCoordinates, anchorBoxes[anchors_nms_idx])
                )

            return [finalScores, finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates]
            # finalScores.shape: [882]
            # finalAnchorBoxesIndexes.shape: [882]
            # finalAnchorBoxesCoordinates.shape: [882, 4]
```
