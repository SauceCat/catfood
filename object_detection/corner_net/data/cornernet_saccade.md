# CornerNet Saccade

## Allocating memory

Compare with CornerNet, there is extra attentions.

```python
max_objects = 128

# images
# images.shape: (4, 3, 255, 255)
images = np.zeros((batch_size, 3, input_size[0], input_size[1]), dtype=np.float32)

# heatmaps
# tl_heats.shape: (4, 80, 64, 64)
tl_heats = np.zeros(
    (batch_size, categories, output_size[0], output_size[1]), dtype=np.float32
)
br_heats = np.zeros(
    (batch_size, categories, output_size[0], output_size[1]), dtype=np.float32
)

# extra
# tl_valids.shape: (4, 80, 64, 64)
tl_valids = np.zeros(
    (batch_size, categories, output_size[0], output_size[1]), dtype=np.float32
)
br_valids = np.zeros(
    (batch_size, categories, output_size[0], output_size[1]), dtype=np.float32
)

# offsets
# tl_regrs.shape: (4, 128, 2)
tl_regrs = np.zeros((batch_size, max_objects, 2), dtype=np.float32)
br_regrs = np.zeros((batch_size, max_objects, 2), dtype=np.float32)

# embeddings
# tl_tags.shape: (4, 128)
tl_tags = np.zeros((batch_size, max_objects), dtype=np.int64)
br_tags = np.zeros((batch_size, max_objects), dtype=np.int64)

# mask
# tag_masks.shape: (4, 128)
tag_masks = np.zeros((batch_size, max_objects), dtype=np.uint8)

# count how many detections per image
# tag_lens.shape: (4,)
tag_lens = np.zeros((batch_size,), dtype=np.int32)

# attentions
attentions = [
    np.zeros((batch_size, 1, att_size[0], att_size[1]), dtype=np.float32)
    for att_size in att_sizes
]
# attentions.shape: [(4, 1, 16, 16), (4, 1, 32, 32), (4, 1, 64, 64)]
```

## Loop for each image

Compare with CornerNet, need to generate attention maps.

```python
db_size = db.db_inds.size
for b_ind in range(batch_size):
    if not debug and k_ind == 0:
        db.shuffle_inds()

    db_ind = db.db_inds[k_ind]
    k_ind = (k_ind + 1) % db_size

    # read image
    image_path = db.image_path(db_ind)
    # image.shape: (333, 500, 3)
    image = cv2.imread(image_path)

    # read detections
    # orig_detections.shape: (3, 5)
    orig_detections = db.detections(db_ind)
    keep_inds = np.arange(orig_detections.shape[0])
    
    detections = orig_detections.copy()
    # border: [0, 333, 0, 500]
    border = [0, image.shape[0], 0, image.shape[1]]
    
    # clip and only keep valid detections
    detections, clip_inds = clip_detections(border, detections)
    keep_inds = keep_inds[clip_inds]
```

### scale based on a random reference detection

```python
    # scale based on a random reference detection
    scale, ref_ind = ref_scale(detections, random_crop=rand_crop)
    # scale: 0.5170575524740466, ref_ind: 0
    scale = np.random.choice(rand_scales) if scale is None else scale
    # scale: 0.5170575524740466

    # scale orig_detections
    orig_detections[:, 0:4:2] *= scale
    orig_detections[:, 1:4:2] *= scale

    # scale image and detections
    image, detections = scale_image_detections(image, detections, scale)
    # image.shape: (172, 258, 3)
    ref_detection = detections[ref_ind].copy()
```

```python
def ref_scale(detections, random_crop=False):
    if detections.shape[0] == 0:
        return None, None

    if random_crop and np.random.uniform() > 0.7:
        return None, None

    ref_ind = np.random.randint(detections.shape[0])
    ref_det = detections[ref_ind].copy()
    ref_h = ref_det[3] - ref_det[1]
    ref_w = ref_det[2] - ref_det[0]
    ref_hw = max(ref_h, ref_w)
    # ref_h, ref_w, ref_hw: 187.6, 126.45999, 187.6

    if ref_hw > 96:
        return np.random.randint(low=96, high=255) / ref_hw, ref_ind
    elif ref_hw > 32:
        return np.random.randint(low=32, high=97) / ref_hw, ref_ind
    return np.random.randint(low=16, high=33) / ref_hw, ref_ind
```

### crop image and detections

```python
    # input_size: [255, 255]
    # rand_center: True
    image, detections, border = crop_image_dets(
        image, detections, ref_ind, input_size, rand_center=rand_center
    )
    # image.shape: (255, 255, 3)
```

```python
def crop_image_dets(
    image, dets, ind, input_size, 
    output_size=None, random_crop=True, rand_center=True
):
    if ind is not None:
        det_x0, det_y0, det_x1, det_y1 = dets[ind, 0:4]
        # [94.797325, 0., 160.18442, 97.]
    else:
        det_x0, det_y0, det_x1, det_y1 = None, None, None, None

    input_height, input_width = input_size
    # input_height, input_width: 255, 255
    image_height, image_width = image.shape[0:2]
    # image_height, image_width: 172, 258

    # random center
    centered = rand_center and np.random.uniform() > 0.5
    if not random_crop or image_width <= input_width:
        xc = image_width // 2
    elif ind is None or not centered:
        xmin = max(det_x1 - input_width, 0) if ind is not None else 0
        xmax = (
            min(image_width - input_width, det_x0)
            if ind is not None
            else image_width - input_width
        )
        xrand = np.random.randint(int(xmin), int(xmax) + 1)
        xc = xrand + input_width // 2
    else:
        xmin = max((det_x0 + det_x1) // 2 - np.random.randint(0, 15), 0)
        xmax = min((det_x0 + det_x1) // 2 + np.random.randint(0, 15), image_width - 1)
        xc = np.random.randint(int(xmin), int(xmax) + 1)
        # xmin, xmax, xc: 123.0, 136.0, 130

    if not random_crop or image_height <= input_height:
        yc = image_height // 2
        # yc: 86
    elif ind is None or not centered:
        ymin = max(det_y1 - input_height, 0) if ind is not None else 0
        ymax = (
            min(image_height - input_height, det_y0)
            if ind is not None
            else image_height - input_height
        )
        yrand = np.random.randint(int(ymin), int(ymax) + 1)
        yc = yrand + input_height // 2
    else:
        ymin = max((det_y0 + det_y1) // 2 - np.random.randint(0, 15), 0)
        ymax = min((det_y0 + det_y1) // 2 + np.random.randint(0, 15), image_height - 1)
        yc = np.random.randint(int(ymin), int(ymax) + 1)

    # (yc, xc): 86, 130
    # input_size: [255, 255], output_size = None
    image, border, offset = crop_image(
        image, [yc, xc], input_size, output_size=output_size
    )
    # image.shape: (255, 255, 3)
    # border: [ 41., 213.,   0., 254.]
    # offset: array([-41,   3])
    dets[:, 0:4:2] -= offset[1]
    dets[:, 1:4:2] -= offset[0]
    return image, dets, border
```

### continue

```python
    # clip and only keep valid detections
    detections, clip_inds = clip_detections(border, detections)
    keep_inds = keep_inds[clip_inds]

    # input_size: [255, 255], output_size: [64, 64]
    width_ratio = output_size[1] / input_size[1]
    height_ratio = output_size[0] / input_size[0]

    # flipping an image randomly
    if not debug and np.random.uniform() > 0.5:
        image[:] = image[:, ::-1, :]
        width = image.shape[1]
        detections[:, [0, 2]] = width - detections[:, [2, 0]] - 1
```

### create attention mask

```python
    create_attention_mask(
        [att[b_ind, 0] for att in attentions], att_ratios, att_ranges, detections
    )
```

```python
def create_attention_mask(atts, ratios, sizes, detections):
    # atts.shape: [(16, 16), (32, 32), (64, 64)]
    # ratios: [16, 8, 4]
    # sizes: [[96, 256], [32, 96], [0, 32]]
    for det in detections:
        # det: [45.551296, 1.9675328, 186.58896, 247.96751, 24.]
        width = det[2] - det[0]
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

### overlap between detections and orig_detections

```python
    overlaps = bbox_overlaps(detections, orig_detections[keep_inds]) > 0.5
```

```python
def bbox_overlaps(a_dets, b_dets):
    a_widths = a_dets[:, 2] - a_dets[:, 0]
    a_heights = a_dets[:, 3] - a_dets[:, 1]
    a_areas = a_widths * a_heights

    b_widths = b_dets[:, 2] - b_dets[:, 0]
    b_heights = b_dets[:, 3] - b_dets[:, 1]
    b_areas = b_widths * b_heights

    return a_areas / b_areas
```

### continue

```python
    if not debug:
        image = image.astype(np.float32) / 255.0
        color_jittering_(data_rng, image)
        lighting_(data_rng, image, 0.1, db.eig_val, db.eig_vec)
        normalize_(image, db.mean, db.std)
    images[b_ind] = image.transpose((2, 0, 1))
```

## For each image, loop for each detection

```python
for ind, (detection, overlap) in enumerate(zip(detections, overlaps)):
    category = int(detection[-1]) - 1

    xtl, ytl = detection[0], detection[1]
    xbr, ybr = detection[2], detection[3]

    # filter out extreme small detections
    det_height = int(ybr) - int(ytl)
    det_width = int(xbr) - int(xtl)
    det_max = max(det_height, det_width)
    # min_scale: 16
    valid = det_max >= min_scale

    fxtl = xtl * width_ratio
    fytl = ytl * height_ratio
    fxbr = xbr * width_ratio
    fybr = ybr * height_ratio

    xtl = int(fxtl)
    ytl = int(fytl)
    xbr = int(fxbr)
    ybr = int(fybr)

    width = detection[2] - detection[0]
    height = detection[3] - detection[1]
    width = math.ceil(width * width_ratio)
    height = math.ceil(height * height_ratio)
	
    # Reducing Penalty to Negative Locations
    if gaussian_rad == -1:
        radius = gaussian_radius((height, width), gaussian_iou)
        radius = max(0, int(radius))
    else:
        radius = gaussian_rad

    if overlap and valid:
        draw_gaussian(tl_heats[b_ind, category], [xtl, ytl], radius)
        draw_gaussian(br_heats[b_ind, category], [xbr, ybr], radius)

        tag_ind = tag_lens[b_ind]
        
        # offsets
        tl_regrs[b_ind, tag_ind, :] = [fxtl - xtl, fytl - ytl]
        br_regrs[b_ind, tag_ind, :] = [fxbr - xbr, fybr - ybr]
        
        # class index
        tl_tags[b_ind, tag_ind] = ytl * output_size[1] + xtl
        br_tags[b_ind, tag_ind] = ybr * output_size[1] + xbr
        
        # count += 1
        tag_lens[b_ind] += 1
    else:
        # ignore invalid tags
        draw_gaussian(tl_valids[b_ind, category], [xtl, ytl], radius)
        draw_gaussian(br_valids[b_ind, category], [xbr, ybr], radius)
```

## Mask out the invalid part

```python
tl_valids = (tl_valids == 0).astype(np.float32)
br_valids = (br_valids == 0).astype(np.float32)

for b_ind in range(batch_size):
    tag_len = tag_lens[b_ind]
    tag_masks[b_ind, :tag_len] = 1
```

## Final outputs

Compare to CornerNet, there is extra attentions.

```python
return (
    {
        "xs": [images],
        "ys": [
            tl_heats,
            br_heats,
            tag_masks,
            tl_regrs,
            br_regrs,
            tl_tags,
            br_tags,
            tl_valids,
            br_valids,
            attentions,
        ],
    },
    k_ind,
)

# CornerNet
return (
    {
        "xs": [images],
        "ys": [
            tl_heatmaps,
            br_heatmaps,
            tag_masks,
            tl_regrs,
            br_regrs,
            tl_tags,
            br_tags,
        ],
    },
    k_ind,
)
```

