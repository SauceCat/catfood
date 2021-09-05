# Training Data Sampling: CornerNet Saccade

## extra inputs

Compared with CornerNet, CornerNet Squeeze.

```python
att_ratios = db.configs["att_ratios"]
# att_ratios: [16, 8, 4]
att_ranges = db.configs["att_ranges"]
# att_ranges: [[96, 256], [32, 96], [0, 32]]
att_sizes = db.configs["att_sizes"]
# att_sizes: [[16, 16], [32, 32], [64, 64]]
```

## allocating memory

```python
tl_valids = np.zeros(
    (batch_size, categories, output_size[0], output_size[1]), 
    dtype=np.float32
)
br_valids = np.zeros(
    (batch_size, categories, output_size[0], output_size[1]), 
    dtype=np.float32
)

attentions = [
    np.zeros(
        (batch_size, 1, att_size[0], att_size[1]), 
        dtype=np.float32
    ) 
    for att_size in att_sizes
]
# attentions.shape: [
#	(4, 1, 16, 16), 
#	(4, 1, 32, 32), 
#	(4, 1, 64, 64)
# ]
```

## for each image

### read images and clip the detections

```python
image = cv2.imread(image_path)
# image.shape: (333, 500, 3)

orig_detections = db.detections(db_ind)
# orig_detections.shape: (3, 5)
keep_inds = np.arange(orig_detections.shape[0])

# clip and only keep valid detections
detections = orig_detections.copy()
border = [0, image.shape[0], 0, image.shape[1]]
# border: [0, 333, 0, 500]
detections, clip_inds = clip_detections(border, detections)
keep_inds = keep_inds[clip_inds]
```

### random scale

```python
scale, ref_ind = ref_scale(detections, random_crop=rand_crop)
# scale: 0.5170575524740466, ref_ind: 0
scale = np.random.choice(rand_scales) if scale is None else scale
# scale: 0.5170575524740466

# scale detections
orig_detections[:, 0:4:2] *= scale
orig_detections[:, 1:4:2] *= scale

# scale image and detections
image, detections = scale_image_detections(image, detections, scale)
# image.shape: (172, 258, 3)
# detections the same as orig_detections
ref_detection = detections[ref_ind].copy()

def ref_scale(detections, random_crop=False):
    if detections.shape[0] == 0:
        return None, None

    if random_crop and np.random.uniform() > 0.7:
        return None, None

    ref_ind = np.random.randint(detections.shape[0])
    ref_det = detections[ref_ind].copy()
    ref_h   = ref_det[3] - ref_det[1]
    ref_w   = ref_det[2] - ref_det[0]
    ref_hw  = max(ref_h, ref_w)
    # ref_h, ref_w, ref_hw: 187.6, 126.45999, 187.6

    if ref_hw > 96:
        return np.random.randint(low=96, high=255) / ref_hw, ref_ind
    elif ref_hw > 32:
        return np.random.randint(low=32, high=97) / ref_hw, ref_ind
    return np.random.randint(low=16, high=33) / ref_hw, ref_ind
```

### random crop image and detections

```python
# input_size: [255, 255]
# rand_center: True
image, detections, border = crop_image_dets(
    image, detections, ref_ind, input_size, rand_center=rand_center)
# image.shape: (255, 255, 3)

# clip and only keep valid detections
detections, clip_inds = clip_detections(border, detections)
keep_inds = keep_inds[clip_inds]
```

### flipping an image randomly + color jittering and lighting adjustment

same as CornerNet.

### create attention mask

```python
create_attention_mask(
    [att[b_ind, 0] for att in attentions], 
    att_ratios, 
    att_ranges, 
    detections
)

def create_attention_mask(atts, ratios, sizes, detections):
    # atts.shape: [(16, 16), (32, 32), (64, 64)]
    # ratios: [16, 8, 4]
    # sizes: [[96, 256], [32, 96], [0, 32]]
    for det in detections:
        # det: [45.551296, 1.9675328, 186.58896, 247.96751, 24.]
        width  = det[2] - det[0]
        height = det[3] - det[1]
        # width, height: 141.03766, 245.99998

        max_hw = max(width, height)
        for att, ratio, size in zip(atts, ratios, sizes):
            if max_hw >= size[0] and max_hw <= size[1]:
                x = (det[0] + det[2]) / 2
                y = (det[1] + det[3]) / 2
                x = (x / ratio).astype(np.int32)
                y = (y / ratio).astype(np.int32)
                # x, y: 7, 7
                att[y, x] = 1
```

### get overlap between detections and the original ones

```python
# orig_detections
orig_detections = db.detections(db_ind)
orig_detections[:, 0:4:2] *= scale
orig_detections[:, 1:4:2] *= scale

# detections
detections = orig_detections.copy()
detections, clip_inds = clip_detections(border, detections)
image, detections = scale_image_detections(image, detections, scale)
image, detections, border = crop_image_dets(
    image, detections, ref_ind, input_size, rand_center=rand_center)
detections, clip_inds = clip_detections(border, detections)

# get overlaps
overlaps = bbox_overlaps(detections, orig_detections[keep_inds]) > 0.5
```

## for each detection

### filter out small detections

```python
# extra, filter out extreme small detections
det_height = int(ybr) - int(ytl)
det_width  = int(xbr) - int(xtl)
det_max    = max(det_height, det_width)
valid = det_max >= min_scale
```

### gaussian bump

```python
if overlap and valid:
    draw_gaussian(
        heatmap=tl_heats[b_ind, category], 
        center=[xtl, ytl], 
        k=radius
    )
    draw_gaussian(
        heatmap=br_heats[b_ind, category], 
        center=[xbr, ybr], 
        k=radius
    )
    tag_ind = tag_lens[b_ind]
    tl_regrs[b_ind, tag_ind, :] = [fxtl - xtl, fytl - ytl]
    br_regrs[b_ind, tag_ind, :] = [fxbr - xbr, fybr - ybr]
    tl_tags[b_ind, tag_ind] = ytl * output_size[1] + xtl
    br_tags[b_ind, tag_ind] = ybr * output_size[1] + xbr
    tag_lens[b_ind] += 1
else:
    # gaussian bump for invalid detections (skip in training loss calculation)
    draw_gaussian(
        heatmap=tl_valids[b_ind, category], 
        center=[xtl, ytl], 
        k=radius
    )
    draw_gaussian(
        heatmap=br_valids[b_ind, category], 
        center=[xbr, ybr], 
        k=radius
    )
    
tl_valids = (tl_valids == 0).astype(np.float32)
br_valids = (br_valids == 0).astype(np.float32)
```

## final outputs

```python
return {
    "xs": [images],
    "ys": [tl_heats, br_heats, tag_masks, tl_regrs, br_regrs, tl_tags, br_tags, tl_valids, br_valids, attentions]
}, k_ind
```
