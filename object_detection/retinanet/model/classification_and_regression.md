# Box classification and regression

```python
# classification and regression use separate parameters
self.regressionModel = RegressionModel(256)
self.classificationModel = ClassificationModel(256, num_classes=num_classes)
```

## Box regression

```python
class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, feature_size=256):
        super(RegressionModel, self).__init__()
        
        # common part: conv1 -> conv4
        self.conv1 = nn.Conv2d(
            num_features_in, feature_size, kernel_size=3, padding=1
        )
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(
            feature_size, feature_size, kernel_size=3, padding=1
        )
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(
            feature_size, feature_size, kernel_size=3, padding=1
        )
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(
            feature_size, feature_size, kernel_size=3, padding=1
        )
        self.act4 = nn.ReLU()
        
        # diff: output 4 coordinates for each box
        self.output = nn.Conv2d(
            feature_size, num_anchors * 4, kernel_size=3, padding=1
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)
        # common part:
        # x: [2, 256, 80, 104] 
        # out: [2, 256, 80, 104]

        out = self.output(out)
        # out: [2, 36, 80, 104] (36 = 4 * num_anchors)

        # out is B x C x W x H, with C = 4 * num_anchors
        out = out.permute(0, 2, 3, 1)
        # out: [2, 80, 104, 36]

        return out.contiguous().view(out.shape[0], -1, 4)
        # return: [2, 74880, 4], 74880 = 80 * 104 * num_anchors
```

## Box classification

```python
class ClassificationModel(nn.Module):
    def __init__(
        self, num_features_in, num_anchors=9, 
        num_classes=80, prior=0.01, feature_size=256
    ):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # common part: conv1 -> conv4
        self.conv1 = nn.Conv2d(
            num_features_in, feature_size, kernel_size=3, padding=1
        )
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(
            feature_size, feature_size, kernel_size=3, padding=1
        )
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(
            feature_size, feature_size, kernel_size=3, padding=1
        )
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(
            feature_size, feature_size, kernel_size=3, padding=1
        )
        self.act4 = nn.ReLU()
        
        # diff: output class probabilities
        self.output = nn.Conv2d(
            feature_size, num_anchors * num_classes, kernel_size=3, padding=1
        )
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)
        # common part:
        # x: [2, 256, 80, 104] 
        # out: [2, 256, 80, 104]

        out = self.output(out)
        out = self.output_act(out)
        # out: [2, 180, 80, 104], 180 = num_classes * num_anchors

        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)
        # out1: [2, 80, 104, 180]
        
        batch_size, width, height, channels = out1.shape
        out2 = out1.view(
            batch_size, width, height, self.num_anchors, self.num_classes
        )
        # out2: [2, 80, 104, 9, 20]

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)
        # return: [2, 74880, 20], 74880 = 80 * 104 * num_anchors
```
