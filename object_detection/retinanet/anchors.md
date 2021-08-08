# Anchors

### initialize

```python
# [layer2, layer3, layer4, ...]
self.pyramid_levels = [3, 4, 5, 6, 7]

# 2 ** 1 (conv1), 2 ** 2 (layer1)
# [2 ** 3 (layer2), 2 ** 4 (layer3), 2 ** 5 (layer4)]
# [8, 16, 32, 64, 128]
self.strides = [2 ** x for x in self.pyramid_levels]
        
# [2 ** (3 + 2), 2 ** (4 + 2), 2 ** (5 + 2), 2 ** (6 + 2), 2 ** (7 + 2)]
# [32, 64, 128, 256, 512]
# (x + 2) because we start from layer2
self.sizes = [2 ** (x + 2) for x in self.pyramid_levels]

self.ratios = np.array([0.5, 1, 2])
self.scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
```

---

### forward

```python
# image: [2, 3, 640, 832]
image_shape = image.shape[2:]
image_shape = np.array(image_shape)
image_shapes = [
    (image_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels
]
# image_shapes: [80, 104], [40, 52], [20, 26], [10, 13], [5, 7]
# fpn features: [P3_x, P4_x, P5_x, P6_x, P7_x]
# P3_x: [2, 256, 80, 104]
# P4_x: [2, 256, 40, 52]
# P5_x: [2, 256, 20, 26]
# P6_x: [2, 256, 10, 13]
# P7_x: [2, 256, 5, 7]

# compute anchors over all pyramid levels
all_anchors = np.zeros((0, 4)).astype(np.float32)
        
# self.pyramid_levels = [3, 4, 5, 6, 7]
# self.sizes: [32, 64, 128, 256, 512]
# self.strides: [8, 16, 32, 64, 128]
# image_shapes: [80, 104], [40, 52], [20, 26], [10, 13], [5, 7]
for idx, p in enumerate(self.pyramid_levels):
    anchors = generate_anchors(
        base_size=self.sizes[idx], ratios=self.ratios, scales=self.scales
    )
    # shifted_anchors: (74880, 4), (18720, 4), (4680, 4), (1170, 4), (315, 4)
    shifted_anchors = shift(image_shapes[idx], self.strides[idx], anchors)
    all_anchors = np.append(all_anchors, shifted_anchors, axis=0)
        
# all_anchors: (99765, 4) -> (1, 99765, 4)
all_anchors = np.expand_dims(all_anchors, axis=0)

return torch.from_numpy(all_anchors.astype(np.float32)).cuda()
```

---

### `generate_anchors`

```python
anchors = generate_anchors(
    base_size=self.sizes[idx], ratios=self.ratios, scales=self.scales
)
# base_size = 32
# ratios = [0.5, 1, 2]
# scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
```

#### initialize

```python
# num_anchors = 9 
num_anchors = len(ratios) * len(scales)

# initialize output anchors
# anchors: [9, 4]
anchors = np.zeros((num_anchors, 4))
```

`anchors`: (9, 4)

```
array([[0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.]])
```

#### scale base_size

```python
# scale base_size
anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T
```

`np.tile(scales, (2, len(ratios)))`: (2, 9)

```
array([[1.        , 1.25992105, 1.58740105, 1.        , 1.25992105,
        1.58740105, 1.        , 1.25992105, 1.58740105],
       [1.        , 1.25992105, 1.58740105, 1.        , 1.25992105,
        1.58740105, 1.        , 1.25992105, 1.58740105]])
```

`anchors`: (4, 9)

```
array([[ 0.        ,  0.        , 32.        , 32.        ],
       [ 0.        ,  0.        , 40.3174736 , 40.3174736 ],
       [ 0.        ,  0.        , 50.79683366, 50.79683366],
       [ 0.        ,  0.        , 32.        , 32.        ],
       [ 0.        ,  0.        , 40.3174736 , 40.3174736 ],
       [ 0.        ,  0.        , 50.79683366, 50.79683366],
       [ 0.        ,  0.        , 32.        , 32.        ],
       [ 0.        ,  0.        , 40.3174736 , 40.3174736 ],
       [ 0.        ,  0.        , 50.79683366, 50.79683366]])
```

#### correct for ratios

```python
# compute areas of anchors
areas = anchors[:, 2] * anchors[:, 3]

# correct for ratios
# ratio = width / height
# area = width * height = ratio * height * height
# height = sqrt(area / ratio)
# width = height * ratio
anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))
```

`areas`: (9,)

```
array([1024.        , 1625.49867722, 2580.31831018, 1024.        ,
       1625.49867722, 2580.31831018, 1024.        , 1625.49867722,
       2580.31831018])
```

`np.repeat(ratios, len(scales))`:

```
array([0.5, 0.5, 0.5, 1. , 1. , 1. , 2. , 2. , 2. ])
```

`anchors`:

```
array([[ 0.        ,  0.        , 45.254834  , 22.627417  ],
       [ 0.        ,  0.        , 57.01751796, 28.50875898],
       [ 0.        ,  0.        , 71.83757109, 35.91878555],
       [ 0.        ,  0.        , 32.        , 32.        ],
       [ 0.        ,  0.        , 40.3174736 , 40.3174736 ],
       [ 0.        ,  0.        , 50.79683366, 50.79683366],
       [ 0.        ,  0.        , 22.627417  , 45.254834  ],
       [ 0.        ,  0.        , 28.50875898, 57.01751796],
       [ 0.        ,  0.        , 35.91878555, 71.83757109]])
```

#### transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)

```python
anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T
```

`np.tile(anchors[:, 2] * 0.5, (2, 1)).T`:

```
array([[22.627417  , 22.627417  ],
       [28.50875898, 28.50875898],
       [35.91878555, 35.91878555],
       [16.        , 16.        ],
       [20.1587368 , 20.1587368 ],
       [25.39841683, 25.39841683],
       [11.3137085 , 11.3137085 ],
       [14.25437949, 14.25437949],
       [17.95939277, 17.95939277]])
```

`anchors`:

```
array([[-22.627417  , -11.3137085 ,  22.627417  ,  11.3137085 ],
       [-28.50875898, -14.25437949,  28.50875898,  14.25437949],
       [-35.91878555, -17.95939277,  35.91878555,  17.95939277],
       [-16.        , -16.        ,  16.        ,  16.        ],
       [-20.1587368 , -20.1587368 ,  20.1587368 ,  20.1587368 ],
       [-25.39841683, -25.39841683,  25.39841683,  25.39841683],
       [-11.3137085 , -22.627417  ,  11.3137085 ,  22.627417  ],
       [-14.25437949, -28.50875898,  14.25437949,  28.50875898],
       [-17.95939277, -35.91878555,  17.95939277,  35.91878555]])
```

---

### `shift`

```python
shifted_anchors = shift(image_shapes[idx], self.strides[idx], anchors)
# shape = [80, 104]
# stride = 8
# anchors (9, 4)
```

#### prepare shifts

```python
shift_x = (np.arange(0, shape[1]) + 0.5) * stride
shift_y = (np.arange(0, shape[0]) + 0.5) * stride
# shift_x.shape: (104,)
# shift_y.shape: (80,)

shift_x, shift_y = np.meshgrid(shift_x, shift_y)
# shift_x.shape: (80, 104)
# shift_y.shape: (80, 104)

shifts = np.vstack(
    (shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())
).transpose()
# shifts.shape: (80 * 104 = 8320, 4)
```

`np.arange(0, shape[0]) + 0.5`:

```
array([ 0.5,  1.5,  2.5,  3.5,  4.5,  5.5,  6.5,  7.5,  8.5,  9.5, 10.5,
       11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5,
       22.5, 23.5, 24.5, 25.5, 26.5, 27.5, 28.5, 29.5, 30.5, 31.5, 32.5,
       33.5, 34.5, 35.5, 36.5, 37.5, 38.5, 39.5, 40.5, 41.5, 42.5, 43.5,
       44.5, 45.5, 46.5, 47.5, 48.5, 49.5, 50.5, 51.5, 52.5, 53.5, 54.5,
       55.5, 56.5, 57.5, 58.5, 59.5, 60.5, 61.5, 62.5, 63.5, 64.5, 65.5,
       66.5, 67.5, 68.5, 69.5, 70.5, 71.5, 72.5, 73.5, 74.5, 75.5, 76.5,
       77.5, 78.5, 79.5])
```

`(np.arange(0, shape[0]) + 0.5) * stride`: (`image: [2, 3, 640, 832]`)

```
array([  4.,  12.,  20.,  28.,  36.,  44.,  52.,  60.,  68.,  76.,  84.,
        92., 100., 108., 116., 124., 132., 140., 148., 156., 164., 172.,
       180., 188., 196., 204., 212., 220., 228., 236., 244., 252., 260.,
       268., 276., 284., 292., 300., 308., 316., 324., 332., 340., 348.,
       356., 364., 372., 380., 388., 396., 404., 412., 420., 428., 436.,
       444., 452., 460., 468., 476., 484., 492., 500., 508., 516., 524.,
       532., 540., 548., 556., 564., 572., 580., 588., 596., 604., 612.,
       620., 628., 636.])
```

`shift_x`:

```
array([[  4.,  12.,  20., ..., 812., 820., 828.],
       [  4.,  12.,  20., ..., 812., 820., 828.],
       [  4.,  12.,  20., ..., 812., 820., 828.],
       ...,
       [  4.,  12.,  20., ..., 812., 820., 828.],
       [  4.,  12.,  20., ..., 812., 820., 828.],
       [  4.,  12.,  20., ..., 812., 820., 828.]])
```

`shift_x.ravel()`:

```
array([  4.,  12.,  20., ..., 812., 820., 828.])
```

`shift_y`:

```
array([[  4.,   4.,   4., ...,   4.,   4.,   4.],
       [ 12.,  12.,  12., ...,  12.,  12.,  12.],
       [ 20.,  20.,  20., ...,  20.,  20.,  20.],
       ...,
       [620., 620., 620., ..., 620., 620., 620.],
       [628., 628., 628., ..., 628., 628., 628.],
       [636., 636., 636., ..., 636., 636., 636.]])
```

`shift_y.ravel()`:

```
array([  4.,   4.,   4., ..., 636., 636., 636.])
```

`shifts`: (x, y, x, y) * (104 * 80)

```
array([[  4.,   4.,   4.,   4.],
       [ 12.,   4.,  12.,   4.],
       [ 20.,   4.,  20.,   4.],
       ...,
       [812., 636., 812., 636.],
       [820., 636., 820., 636.],
       [828., 636., 828., 636.]])
```

#### shift anchors

```python
# add A anchors (1, A, 4) to
# cell K shifts (K, 1, 4) to get
# shift anchors (K, A, 4)
# reshape to (K * A, 4) shifted anchors
A = anchors.shape[0]
K = shifts.shape[0]
# A = 9, K = 8320

# anchors: (9, 4) -> (1, 9, 4)
# shifts: (8320, 4) -> (1, 8320, 4) -> (8320, 1, 4)
# all_anchors: (1, 9, 4) + (8320, 1, 4) -> (8320, 9, 4)
all_anchors = anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose(
    (1, 0, 2)
)

# all_anchors: (8320, 9, 4) -> (8320 * 9 = 74880, 4)
all_anchors = all_anchors.reshape((K * A, 4))
```