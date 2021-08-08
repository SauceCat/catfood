# Feature Pyramid Networks

### From ResNets

```python
# fpn_sizes: [layer2, layer3, layer4]
# BasicBlock (ResNet-18/34): [128, 256, 512]
# Bottleneck (ResNet-50/101/152): [512, 1024, 2048]
self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])
```

---

### PyramidFeatures

RetinaNet uses feature pyramid levels P3 to P7:

- P3 to P5 are computed from the output of the corresponding ResNet residual stage (C3 to C5, layer2 to layer4) using top-down and lateral connections just as in FPN
- P6 is obtained via a 3×3 stride-2 conv on C5, and P7 is computed by applying ReLU followed by a 3×3 stride-2 conv on P6. 

This differs slightly from FPN: 

- we don’t use the high-resolution pyramid level P2 for computational reasons
- P6 is computed by strided convolution instead of downsampling
- we include P7 to improve large object detection

These minor modifications improve speed while maintaining accuracy.

#### initialize

```python
C3_size = 512 
C4_size = 1024
C5_size = 2048
feature_size = 256

self.P5_1 = nn.Conv2d(
    C5_size, feature_size, 
    kernel_size=1, stride=1, padding=0
)
self.P5_upsampled = nn.Upsample(scale_factor=2, mode="nearest")
self.P5_2 = nn.Conv2d(
    feature_size, feature_size, 
    kernel_size=3, stride=1, padding=1
)

self.P4_1 = nn.Conv2d(
    C4_size, feature_size, 
    kernel_size=1, stride=1, padding=0
)
self.P4_upsampled = nn.Upsample(scale_factor=2, mode="nearest")
self.P4_2 = nn.Conv2d(
    feature_size, feature_size, 
    kernel_size=3, stride=1, padding=1
)

self.P3_1 = nn.Conv2d(
    C3_size, feature_size, 
    kernel_size=1, stride=1, padding=0
)
self.P3_2 = nn.Conv2d(
    feature_size, feature_size, 
    kernel_size=3, stride=1, padding=1
)

self.P6 = nn.Conv2d(
    C5_size, feature_size, 
    kernel_size=3, stride=2, padding=1
)

self.P7_1 = nn.ReLU()
self.P7_2 = nn.Conv2d(
    feature_size, feature_size, 
    kernel_size=3, stride=2, padding=1
)
```

#### forward

```python
# B, C, H, W
# C3: [2, 512, 80, 104]
# C4: [2, 1024, 40, 52]
# C5: [2, 2048, 20, 26]
C3, C4, C5 = inputs

# C5: [2, 2048, 20, 26]
P5_x = self.P5_1(C5)
# P5_x: [2, 256, 20, 26]

# upsample C5 to get P5 from the FPN paper
# P5_x: [2, 256, 20, 26]
P5_upsampled_x = self.P5_upsampled(P5_x)
# P5_upsampled_x: [2, 256, 40, 52]

# P5_x: [2, 256, 20, 26]
P5_x = self.P5_2(P5_x)
# P5_x: [2, 256, 20, 26]

# C4: [2, 1024, 40, 52]
P4_x = self.P4_1(C4)
# P4_x: [2, 256, 40, 52]

# add P5 elementwise to C4
# P4_x: [2, 256, 40, 52]
# P5_upsampled_x: [2, 256, 40, 52]
P4_x = P5_upsampled_x + P4_x
# P4_x: [2, 256, 40, 52]

# P4_x: [2, 256, 40, 52]
P4_upsampled_x = self.P4_upsampled(P4_x)
# P4_upsampled_x: [2, 256, 80, 104]

# P4_x: [2, 256, 40, 52]
P4_x = self.P4_2(P4_x)
# P4_x: [2, 256, 40, 52]

# C3: [2, 512, 80, 104]
P3_x = self.P3_1(C3)
# P3_x: [2, 256, 80, 104]

# P4_upsampled_x: [2, 256, 80, 104]
# P3_x: [2, 256, 80, 104]
P3_x = P3_x + P4_upsampled_x
# P3_x: [2, 256, 80, 104]

# P3_x: [2, 256, 80, 104]
P3_x = self.P3_2(P3_x)
# P3_x: [2, 256, 80, 104]

# P6 is obtained via a 3x3 stride-2 conv on C5
# C5: [2, 2048, 20, 26]
P6_x = self.P6(C5)
# P6_x: [2, 256, 10, 13]

# P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6
# P6_x: [2, 256, 10, 13]
P7_x = self.P7_1(P6_x)
# P7_x: [2, 256, 10, 13]
P7_x = self.P7_2(P7_x)
# P7_x: [2, 256, 5, 7]
```

#### return

```python
return [P3_x, P4_x, P5_x, P6_x, P7_x]
# P3_x: [2, 256, 80, 104]
# P4_x: [2, 256, 40, 52]
# P5_x: [2, 256, 20, 26]
# P6_x: [2, 256, 10, 13]
# P7_x: [2, 256, 5, 7]
```

