# Regression and Classification

## initialize

### common part

```python
# num_features_in = 256
# num_anchors=9
# feature_size = 256
self.conv1 = nn.Conv2d(
    num_features_in, feature_size, 
    kernel_size=3, padding=1
)
self.act1 = nn.ReLU()

self.conv2 = nn.Conv2d(
    feature_size, feature_size, 
    kernel_size=3, padding=1
)
self.act2 = nn.ReLU()

self.conv3 = nn.Conv2d(
    feature_size, feature_size, 
    kernel_size=3, padding=1
)
self.act3 = nn.ReLU()

self.conv4 = nn.Conv2d(
    feature_size, feature_size, 
    kernel_size=3, padding=1
)
self.act4 = nn.ReLU()
```

### RegressionModel

```python
self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)
```

### ClassificationModel

```python
self.output = nn.Conv2d(
    feature_size, num_anchors * num_classes, kernel_size=3, padding=1
)
# class probability
self.output_act = nn.Sigmoid()
```

---

## forward

### common part

```python
# x: [2, 256, 80, 104]
out = self.conv1(x)
out = self.act1(out)
# out: [2, 256, 80, 104]

out = self.conv2(out)
out = self.act2(out)
# out: [2, 256, 80, 104]

out = self.conv3(out)
out = self.act3(out)
# out: [2, 256, 80, 104]

out = self.conv4(out)
out = self.act4(out)
# out: [2, 256, 80, 104]
```

### RegressionModel

```python
# out: [2, 256, 80, 104]
out = self.output(out)
# out: [2, 36, 80, 104] (36 = 4 * num_anchors)

out = out.permute(0, 2, 3, 1)
# out: [2, 80, 104, 36]

return out.contiguous().view(out.shape[0], -1, 4)
# return: [2, 74880, 4], 74880 = 80 * 104 * num_anchors
```

### ClassificationModel

```python
# out: [2, 256, 80, 104]
out = self.output(out)
out = self.output_act(out)
# out: [2, 180, 80, 104], 180 = num_classes * num_anchors

out1 = out.permute(0, 2, 3, 1)
# out1: [2, 80, 104, 180]

batch_size, height, width, channels = out1.shape
out2 = out1.view(batch_size, height, width, self.num_anchors, self.num_classes)
# out2: [2, 80, 104, 9, 20]

return out2.contiguous().view(x.shape[0], -1, self.num_classes)
# return: [2, 74880, 20], 74880 = 80 * 104 * num_anchors
```

