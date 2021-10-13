# CornerNet

## Allocating memory

```python
max_tag_len = 128

# images
# images.shape: (49, 3, 511, 511)
images = np.zeros((batch_size, 3, input_size[0], input_size[1]), dtype=np.float32)

# heatmaps
# tl_heatmaps.shape: (49, 80, 128, 128)
tl_heatmaps = np.zeros(
    (batch_size, categories, output_size[0], output_size[1]), dtype=np.float32
)
br_heatmaps = np.zeros(
    (batch_size, categories, output_size[0], output_size[1]), dtype=np.float32
)

# offsets
# tl_regrs.shape: (49, 128, 2)
tl_regrs = np.zeros((batch_size, max_tag_len, 2), dtype=np.float32)
br_regrs = np.zeros((batch_size, max_tag_len, 2), dtype=np.float32)

# embeddings
# tl_tags.shape: (49, 128)
tl_tags = np.zeros((batch_size, max_tag_len), dtype=np.int64)
br_tags = np.zeros((batch_size, max_tag_len), dtype=np.int64)

# mask
# tag_masks.shape: (49, 128)
tag_masks = np.zeros((batch_size, max_tag_len), dtype=np.uint8)

# count how many detections per image
# tag_lens.shape: (49,)
tag_lens = np.zeros((batch_size,), dtype=np.int32)
```

## Loop for each image

**Augmentations:** random crop, random flip, random color jittering, random lighting adjustment

```python
db_size = db.db_inds.size
for b_ind in range(batch_size):
    if not debug and k_ind == 0:
        db.shuffle_inds()

    db_ind = db.db_inds[k_ind]
    k_ind = (k_ind + 1) % db_size

    # read image
    # image_path = './data/coco/images/trainval2014/COCO_train2014_000000159777.jpg'
    image_path = db.image_path(db_ind)
    image = cv2.imread(image_path)

    # read detections
    # detections.shape: (1, 5)
    detections = db.detections(db_ind)

    # crop an image randomly
    if not debug and rand_crop:
        # rand_scales: [0.6, 0.7, 0.8, 0.9, 1. , 1.1, 1.2, 1.3]
        image, detections = random_crop(
            image, detections, rand_scales, input_size, border=border
        )
	
    # resize image, clip detections to the boundary
    # input_size: [511, 511]
    image, detections = _resize_image(image, detections, input_size)
    detections = _clip_detections(image, detections)

    # output_size: [128, 128]
    width_ratio = output_size[1] / input_size[1]
    height_ratio = output_size[0] / input_size[0]

    # flip an image randomly
    if not debug and np.random.uniform() > 0.5:
        image[:] = image[:, ::-1, :]
        width = image.shape[1]
        detections[:, [0, 2]] = width - detections[:, [2, 0]] - 1
	
    if not debug:
        image = image.astype(np.float32) / 255.0
        # random color jittering
        if rand_color:
            color_jittering_(data_rng, image)
            # random lighting adjustment
            if lighting:
                lighting_(data_rng, image, 0.1, db.eig_val, db.eig_vec)
        # normalization
        normalize_(image, db.mean, db.std)
    # HWC -> CHW
    images[b_ind] = image.transpose((2, 0, 1))
```

## For each image, loop for each detection

```python
for ind, detection in enumerate(detections):
    # start from 0 index
    category = int(detection[-1]) - 1

    # top-left and bottom-right
    xtl, ytl = detection[0], detection[1]
    xbr, ybr = detection[2], detection[3]
	
    # downsample
    # width_ratio: 0.25048923679060664
    # height_ratio: 0.25048923679060664
    fxtl = xtl * width_ratio
    fytl = ytl * height_ratio
    fxbr = xbr * width_ratio
    fybr = ybr * height_ratio
	
    # round up (lose precision here)
    xtl = int(fxtl)
    ytl = int(fytl)
    xbr = int(fxbr)
    ybr = int(fybr)

    # Reducing Penalty to Negative Locations
    if gaussian_bump:
        width = detection[2] - detection[0]
        height = detection[3] - detection[1]
        width = math.ceil(width * width_ratio)
        height = math.ceil(height * height_ratio)
		
        # get gaussian radius
        if gaussian_rad == -1:
            # (height, width): (116, 62)
            # gaussian_iou: 0.3
            radius = gaussian_radius((height, width), gaussian_iou)
            # radius = 17
            radius = max(0, int(radius))
        else:
            radius = gaussian_rad

        # draw_gaussian(heatmap, center, radius, k=1)
        draw_gaussian(tl_heatmaps[b_ind, category], [xtl, ytl], radius)
        draw_gaussian(br_heatmaps[b_ind, category], [xbr, ybr], radius)
    else:
        tl_heatmaps[b_ind, category, ytl, xtl] = 1
        br_heatmaps[b_ind, category, ybr, xbr] = 1
	
    # current detection index
    tag_ind = tag_lens[b_ind]
	
    # offsets
    # tl_regrs[b_ind, tag_ind]: array([0.4473306, 0.611202 ], dtype=float32)
    # [fxtl, xtl, fytl, ytl]: [92.44733060176125, 92, 5.611201993639922, 5]
    tl_regrs[b_ind, tag_ind, :] = [fxtl - xtl, fytl - ytl]
    br_regrs[b_ind, tag_ind, :] = [fxbr - xbr, fybr - ybr]
	
    # current location index
    tl_tags[b_ind, tag_ind] = ytl * output_size[1] + xtl
    br_tags[b_ind, tag_ind] = ybr * output_size[1] + xbr
    
    # number of detections + 1
    tag_lens[b_ind] += 1
```

## Mask out the invalid part

Images in the same batch have different number of objects.

```python
for b_ind in range(batch_size):
    tag_len = tag_lens[b_ind]
    tag_masks[b_ind, :tag_len] = 1
```

## Final outputs

```python
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

