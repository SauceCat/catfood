# Training Data Sampling: CornerNet, CornerNet Squeeze

## allocating memory

```python
max_tag_len = 128

images = np.zeros(
    (batch_size, 3, input_size[0], input_size[1]), 
    dtype=np.float32
)
# images.shape: (49, 3, 511, 511)

tl_heatmaps = np.zeros(
    (batch_size, categories, output_size[0], output_size[1]), 
    dtype=np.float32
)
br_heatmaps = np.zeros(
    (batch_size, categories, output_size[0], output_size[1]), 
    dtype=np.float32
)
# tl_heatmaps.shape: (49, 80, 128, 128)

tl_regrs = np.zeros(
    (batch_size, max_tag_len, 2), 
    dtype=np.float32
)
br_regrs = np.zeros(
    (batch_size, max_tag_len, 2), 
    dtype=np.float32
)
# tl_regrs.shape: (49, 128, 2)

tl_tags = np.zeros(
    (batch_size, max_tag_len), 
    dtype=np.int64
)
br_tags = np.zeros(
    (batch_size, max_tag_len), 
    dtype=np.int64
)
# tl_tags.shape: (49, 128)

tag_masks = np.zeros(
    (batch_size, max_tag_len), 
    dtype=np.uint8
)
# tag_masks.shape: (49, 128)

# count how many detections per image
tag_lens = np.zeros((batch_size, ), dtype=np.int32)
# tag_lens.shape: (49,)
```

## for each image

### read image and detections

```python
# image_path = './data/coco/images/trainval2014/COCO_train2014_000000159777.jpg'
image_path = db.image_path(db_ind)
image = cv2.imread(image_path)

# detections.shape: (1, 5)
detections = db.detections(db_ind)
```

### cropping an image randomly

```python
# rand_scales: [0.6, 0.7, 0.8, 0.9, 1. , 1.1, 1.2, 1.3]
image, detections = random_crop(
    image, detections, rand_scales, input_size, border=border
)
```

### resize image and clip detections

```python
image, detections = _resize_image(image, detections, input_size)
detections = _clip_detections(image, detections)
```

### flipping an image randomly

```python
if np.random.uniform() > 0.5:
    image[:] = image[:, ::-1, :]
    width    = image.shape[1]
    detections[:, [0, 2]] = width - detections[:, [2, 0]] - 1
```

### color jittering and lighting adjustment

```python
image = image.astype(np.float32) / 255.
if rand_color:
    color_jittering_(data_rng, image)
    if lighting:
        lighting_(data_rng, image, 0.1, db.eig_val, db.eig_vec)
normalize_(image, db.mean, db.std)
```

## for each detection

### get detection

```python
# input_size: [511, 511], output_size: [128, 128]
width_ratio  = output_size[1] / input_size[1]
height_ratio = output_size[0] / input_size[0]

# start from 0 index
category = int(detection[-1]) - 1

# top-left and bottom-right
xtl, ytl = detection[0], detection[1]
xbr, ybr = detection[2], detection[3]

fxtl, fytl = (xtl * width_ratio), (ytl * height_ratio)
fxbr, fybr = (xbr * width_ratio), (ybr * height_ratio)

xtl, ytl = int(fxtl), int(fytl)
xbr, ybr = int(fxbr), int(fybr)
```

### gaussian bump

![](images/gaussian_bump.png)

```python
width  = detection[2] - detection[0]
height = detection[3] - detection[1]
width  = math.ceil(width * width_ratio)
height = math.ceil(height * height_ratio)

# gaussian_rad: -1
if gaussian_rad == -1:
    # (height, width): (116, 62)
    # gaussian_iou: 0.3
    radius = gaussian_radius(
        det_size=(height, width), 
        min_overlap=gaussian_iou
    )
    # radius = 17.636455930015487
    radius = max(0, int(radius))
else:
    radius = gaussian_rad

# tl_heatmaps.shape: (49, 80, 128, 128)
draw_gaussian(
    heatmap=tl_heatmaps[b_ind, category], 
    center=[xtl, ytl], 
    k=radius
)
draw_gaussian(
    heatmap=br_heatmaps[b_ind, category], 
    center=[xbr, ybr], 
    k=radius
)

def gaussian_radius(det_size, min_overlap):
    # det_size: (116, 62)
    # min_overlap: 0.3
    
    height, width = det_size
    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 - sq1) / (2 * a1)

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 - sq2) / (2 * a2)

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / (2 * a3)
    # r1, r2, r3: 25.373082619188473, 17.636455930015487, 34.08487980097274
    return min(r1, r2, r3)

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = center

    height, width = heatmap.shape[0:2]
    
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
```

### offsets and positions

```python
tag_ind = tag_lens[b_ind]

# tl_regrs[b_ind, tag_ind]: array([0.4473306, 0.611202 ], dtype=float32)
# [fxtl, xtl, fytl, ytl]: [92.44733060176125, 92, 5.611201993639922, 5]
tl_regrs[b_ind, tag_ind, :] = [fxtl - xtl, fytl - ytl]
br_regrs[b_ind, tag_ind, :] = [fxbr - xbr, fybr - ybr]

tl_tags[b_ind, tag_ind] = ytl * output_size[1] + xtl
br_tags[b_ind, tag_ind] = ybr * output_size[1] + xbr
tag_lens[b_ind] += 1
```

### mask

```python
for b_ind in range(batch_size):
    tag_len = tag_lens[b_ind]
    tag_masks[b_ind, :tag_len] = 1
```

## final outputs

```python
images      = torch.from_numpy(images)
tl_heatmaps = torch.from_numpy(tl_heatmaps)
br_heatmaps = torch.from_numpy(br_heatmaps)
tl_regrs    = torch.from_numpy(tl_regrs)
br_regrs    = torch.from_numpy(br_regrs)
tl_tags     = torch.from_numpy(tl_tags)
br_tags     = torch.from_numpy(br_tags)
tag_masks   = torch.from_numpy(tag_masks)

return {
    "xs": [images],
    "ys": [tl_heatmaps, br_heatmaps, tag_masks, tl_regrs, br_regrs, tl_tags, br_tags]
}, k_ind
```
