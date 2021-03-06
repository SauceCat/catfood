# CenterNet

## Allocating memory

Compare with CornerNet, there is an extra center keypoint.

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
ct_heatmaps = np.zeros(
    (batch_size, categories, output_size[0], output_size[1]), dtype=np.float32
)

# offsets
# tl_regrs.shape: (49, 128, 2)
tl_regrs = np.zeros((batch_size, max_tag_len, 2), dtype=np.float32)
br_regrs = np.zeros((batch_size, max_tag_len, 2), dtype=np.float32)
ct_regrs = np.zeros((batch_size, max_tag_len, 2), dtype=np.float32)

# embeddings
# tl_tags.shape: (49, 128)
tl_tags = np.zeros((batch_size, max_tag_len), dtype=np.int64)
br_tags = np.zeros((batch_size, max_tag_len), dtype=np.int64)
ct_tags = np.zeros((batch_size, max_tag_len), dtype=np.int64)

# mask
# tag_masks.shape: (49, 128)
tag_masks = np.zeros((batch_size, max_tag_len), dtype=np.uint8)

# count how many detections per image
# tag_lens.shape: (49,)
tag_lens = np.zeros((batch_size,), dtype=np.int32)
```

## Loop for each image

Same as CornerNet.

## For each image, loop for each detection

Compare with CornerNet, there is an extra center keypoint.

```python
for ind, detection in enumerate(detections):
    # start from 0 index
    category = int(detection[-1]) - 1

    # top-left and bottom-right
    xtl, ytl = detection[0], detection[1]
    xbr, ybr = detection[2], detection[3]
    xct, yct = (
        (detection[2] + detection[0]) / 2.0,
        (detection[3] + detection[1]) / 2.0,
    )
	
    # downsample
    # width_ratio: 0.25048923679060664
    # height_ratio: 0.25048923679060664
    fxtl = xtl * width_ratio
    fytl = ytl * height_ratio
    fxbr = xbr * width_ratio
    fybr = ybr * height_ratio
    fxct = xct * width_ratio
    fyct = yct * height_ratio
	
    # round up
    xtl = int(fxtl)
    ytl = int(fytl)
    xbr = int(fxbr)
    ybr = int(fybr)
    xct = int(fxct)
    yct = int(fyct)

    # Reducing Penalty to Negative Locations
    if gaussian_bump:
        width = detection[2] - detection[0]
        height = detection[3] - detection[1]
        width = math.ceil(width * width_ratio)
        height = math.ceil(height * height_ratio)

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
        draw_gaussian(ct_heatmaps[b_ind, category], [xct, yct], radius, delte=5)
    else:
        tl_heatmaps[b_ind, category, ytl, xtl] = 1
        br_heatmaps[b_ind, category, ybr, xbr] = 1
        ct_heatmaps[b_ind, category, yct, xct] = 1
	
    # current detection index
    tag_ind = tag_lens[b_ind]
	
    # offsets
    # tl_regrs[b_ind, tag_ind]: array([0.4473306, 0.611202 ], dtype=float32)
    # [fxtl, xtl, fytl, ytl]: [92.44733060176125, 92, 5.611201993639922, 5]
    tl_regrs[b_ind, tag_ind, :] = [fxtl - xtl, fytl - ytl]
    br_regrs[b_ind, tag_ind, :] = [fxbr - xbr, fybr - ybr]
    ct_regrs[b_ind, tag_ind, :] = [fxct - xct, fyct - yct]
	
    # current location index
    tl_tags[b_ind, tag_ind] = ytl * output_size[1] + xtl
    br_tags[b_ind, tag_ind] = ybr * output_size[1] + xbr
    ct_tags[b_ind, tag_ind] = yct * output_size[1] + xct
    
    # number of detections + 1
    tag_lens[b_ind] += 1
```

## Mask out the invalid part

Same as CornerNet

## Final outputs

Compare with CornerNet, there is an extra center keypoint.

```python
return (
    {
        "xs": [images, tl_tags, br_tags, ct_tags],
        "ys": [
            tl_heatmaps,
            br_heatmaps,
            ct_heatmaps,
            tag_masks,
            tl_regrs,
            br_regrs,
            ct_regrs,
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
