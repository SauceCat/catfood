# CornerNet Saccade Inference

## input

```python
def cornernet_saccade_inference(db, nnet, image, decode_func=batch_decode): 
    init_sizes  = db.configs["init_sizes"]
    ref_dets    = db.configs["ref_dets"]
    # init_sizes: [192, 255]
    # ref_dets: True

    att_thresholds = db.configs["att_thresholds"]
    att_scales     = db.configs["att_scales"]
    att_max_crops  = db.configs["att_max_crops"]
    # att_thresholds: [0.3]
    # att_scales: [[1, 2, 4]]
    # att_max_crops: 30

    categories    = db.configs["categories"]
    nms_threshold = db.configs["nms_threshold"]
    max_per_image = db.configs["max_per_image"]
    # categories: 80
    # nms_threshold: 0.5
    # max_per_image: 100

    num_iterations = len(att_thresholds)
    # num_iterations: 1

    ...

    image = torch.from_numpy(image).cuda(non_blocking=True)
    # image.shape: torch.Size([3, 427, 640])
```

## get locations from attentions

```python
# image.shape: torch.Size([3, 427, 640])
# att_scales: [1, 2, 4]
# thresh: 0.3
# init_sizes: [192, 255]
# ref_dets: True
dets, locations, atts = get_locs(
    db=db, 
    nnet=nnet, 
    image=image, 
    im_mean=im_mean, 
    im_std=im_std, 
    att_scales=att_scales[0], 
    thresh=att_thresholds[0], 
    sizes=init_sizes, 
    ref_dets=ref_dets
)
# dets.shape: (17, 8)
# locations.shape: (22, 4)
# atts.shape: [
#     [(1, 16, 16), (1, 16, 16)], 
#     [(1, 32, 32), (1, 32, 32)], 
#     [(1, 64, 64), (1, 64, 64)]
# ]
```

### input

```python
height, width = image.shape[1:3]
# height, width: 427, 640

# sizes: [192, 255]
locations = []
for size in sizes:
    scale = size / max(height, width)
    location = [height // 2, width // 2, scale]
    locations.append(location)

# locations: [[213, 320, 0.3], [213, 320, 0.3984375]]
locations = np.array(locations, dtype=np.float32)
```

### prepare_images

```python
images, offsets = prepare_images(db, image, locations, flipped=False)
# images.shape: torch.Size([2, 3, 255, 255])
# offsets: [[-211.0, -104.0], [-107.0, 0.0]]
images -= im_mean
images /= im_std

def prepare_images(db, image, locs, flipped=True):
    # image.shape: torch.Size([3, 427, 640])
    # locs: [[213, 320, 0.3], [213, 320, 0.3984375]]

    input_size = db.configs["input_size"]
    # input_size: [255, 255]
    num_patches = locs.shape[0]
    # num_patches: 2

    images = torch.cuda.FloatTensor(
        num_patches, 3, input_size[0], input_size[1]
    ).fill_(0)
    # images.shape: torch.Size([2, 3, 255, 255])

    offsets = np.zeros((num_patches, 2), dtype=np.float32)
    for ind, (y, x, scale) in enumerate(locs[:, :3]):
        # scale: 0.3 = 192 / max(427, 640)
        # scale: 0.3984375 = 255 / max(427, 640)
        crop_height = int(input_size[0] / scale)
        crop_width = int(input_size[1] / scale)
        # crop_height, crop_width: int(255 / 0.3) = 849
        # crop_height, crop_width: int(255 / 0.3984375) = 640
        offsets[ind] = crop_image_gpu(
            image=image, 
            center=[int(y), int(x)], 
            size=[crop_height, crop_width], 
            out_image=images[ind]
        )
        # crop_image_gpu: center=[213, 320], size=[849, 849]
        # crop_image_gpu: center=[213, 320], size=[640, 640]
    # offsets: [[-211.0, -104.0], [-107.0, 0.0]]
    return images, offsets
```

### batch_decode

```python
dets, atts = batch_decode(db, nnet, images, no_att=False)
# dets.shape: (2, 12, 8)
# atts.shape: [
#     [(1, 16, 16), (1, 16, 16)], 
#     [(1, 32, 32), (1, 32, 32)], 
#     [(1, 64, 64), (1, 64, 64)]
# ]

def batch_decode(db, nnet, images, no_att=False):
    K = db.configs["top_k"]
    ae_threshold = db.configs["ae_threshold"]
    kernel = db.configs["nms_kernel"]
    num_dets = db.configs["num_dets"]

    att_nms_ks = db.configs["att_nms_ks"]
    att_ranges = db.configs["att_ranges"]
    num_images = images.shape[0]
    # num_images: 2
    detections = []
    attentions = [[] for _ in range(len(att_ranges))]
    # att_ranges: [[96, 256], [32, 96], [0, 32]]

    batch_size = 32
    for b_ind in range(math.ceil(num_images / batch_size)):
        b_start = b_ind * batch_size
        b_end = min(num_images, (b_ind + 1) * batch_size)

        b_images = images[b_start:b_end]
        # b_images.shape: torch.Size([2, 3, 255, 255])
        # K: 12
        # num_dets: 12
        b_outputs = nnet.test(
            [b_images],
            ae_threshold=ae_threshold,
            K=K,
            kernel=kernel,
            test=True,
            num_dets=num_dets,
            no_border=True,
            no_att=no_att,
        )
        # b_outputs = torch.cat([
        #	bboxes, scores, tl_scores, br_scores, clses
        # ], dim=2), atts
        
        if no_att:
            b_detections = b_outputs
        else:
            b_detections = b_outputs[0]
            # b_detections.shape: torch.Size([2, 12, 8])
            b_attentions = b_outputs[1]
            # b_attentions.shape: [
            #     torch.Size([2, 1, 16, 16]), 
            #     torch.Size([2, 1, 32, 32]), 
            #     torch.Size([2, 1, 64, 64])
            # ]
            # att_nms_ks: [3, 3, 3]
            b_attentions = att_nms(b_attentions, att_nms_ks)
            # att_nms: max_pool2d
            b_attentions = [
                b_attention.data.cpu().numpy() for b_attention in b_attentions
            ]
            # b_attentions.shape: [(2, 1, 16, 16), (2, 1, 32, 32), (2, 1, 64, 64)]

        b_detections = b_detections.data.cpu().numpy()
        # b_detections.shape: (2, 12, 8)
        detections.append(b_detections)
        if not no_att:
            for attention, b_attention in zip(attentions, b_attentions):
                attention.append(b_attention)

    # attentions: [[(2, 1, 16, 16)], [(2, 1, 32, 32)], [(2, 1, 64, 64)]]
    if not no_att:
        attentions = (
            [np.concatenate(atts, axis=0) for atts in attentions]
            if detections
            else None
        )
    # attentions: [(2, 1, 16, 16), (2, 1, 32, 32), (2, 1, 64, 64)]
    detections = (
        np.concatenate(detections, axis=0) 
        if detections else np.zeros((0, num_dets, 8))
    )
    # detections.shape: (2, 12, 8)
    return detections, attentions
```

### decode_atts

```python
scales = locations[:, 2]
# scales: array([0.3, 0.3984375], dtype=float32)
next_locations = decode_atts(
    db, atts, att_scales, scales, offsets, height, width, thresh
)
# next_locations: np.stack((next_ys, next_xs, next_scales, next_scores), axis=1)
# next_locations.shape: (40, 4)

def decode_atts(
    db, atts, att_scales, scales, offsets, height, width, thresh, ignore_same=False
):
    att_ranges = db.configs["att_ranges"]
    att_ratios = db.configs["att_ratios"]
    input_size = db.configs["input_size"]

    next_ys, next_xs, next_scales, next_scores = [], [], [], []
    # atts.shape: [(2, 1, 16, 16), (2, 1, 32, 32), (2, 1, 64, 64)]
    num_atts = atts[0].shape[0]
    for aind in range(num_atts):
        for att, att_range, att_ratio, att_scale in zip(
            atts, att_ranges, att_ratios, att_scales
        ):
            # att.shape, att_range, att_ratio, att_scale: (2, 1, 16, 16), [96, 256], 16, 1
            # att[aind, 0].shape: (16, 16)
            # xs, ys: [5], [10]
            # att[aind, 0].shape: (32, 32)
            # xs, ys: [4, 17, 20, 24, 7, 24], [10, 15, 15, 16, 17, 18]
            ys, xs = np.where(att[aind, 0] > thresh)
            scores = att[aind, 0, ys, xs]
			
            # map back to the true location
            ys = ys * att_ratio / scales[aind] + offsets[aind, 0]
            # ys = [10] * 16 / 0.3 - 211.0
            xs = xs * att_ratio / scales[aind] + offsets[aind, 1]
            # xs, ys: [162.66665607] [322.33331214]

            # only keep valid xs, ys
            keep = (ys >= 0) & (ys < height) & (xs >= 0) & (xs < width)
            ys, xs, scores = ys[keep], xs[keep], scores[keep]

            next_scale = att_scale * scales[aind]
            # next_scale = 1 * 0.3
            if (ignore_same and att_scale <= 1) or scales[aind] > 2 or next_scale > 4:
                continue

            next_scales += [next_scale] * len(xs)
            next_scores += scores.tolist()
            next_ys += ys.tolist()
            next_xs += xs.tolist()
            
    next_ys = np.array(next_ys)
    next_xs = np.array(next_xs)
    next_scales = np.array(next_scales)
    next_scores = np.array(next_scores)
    
    return np.stack((next_ys, next_xs, next_scales, next_scores), axis=1)
```

### rescale and remap detections

```python
# [64, 64] -> [255, 255]
rescale_dets_(db, dets)
# [255, 255] -> [[849, 849], [640, 640]]
remap_dets_(dets, scales, offsets)

# dets.shape: (24, 8)
dets = dets.reshape(-1, 8)
keep = dets[:, 4] > 0.3
dets = dets[keep]
# dets.shape: (17, 8)
```

### ref_dets

```python
ref_locations = get_ref_locs(dets)
# ref_locations.shape: (3, 4)
next_locations = np.concatenate((next_locations, ref_locations), axis=0)
# next_locations.shape: (43, 4)
next_locations = location_nms(next_locations, thresh=16)
# next_locations.shape: (22, 4)

def get_ref_locs(dets):
    # dets.shape: (17, 8)
    keep = dets[:, 4] > 0.5
    dets = dets[keep]
    # dets.shape: (3, 8)

    ref_xs = (dets[:, 0] + dets[:, 2]) / 2
    ref_ys = (dets[:, 1] + dets[:, 3]) / 2

    ref_maxhws = np.maximum(dets[:, 2] - dets[:, 0], dets[:, 3] - dets[:, 1])
    # ref_maxhws.shape: [277.2452 , 279.2398 ,  65.09927]
    ref_scales = np.zeros_like(ref_maxhws)
    ref_scores = dets[:, 4]

    large_inds = ref_maxhws > 96
    medium_inds = (ref_maxhws > 32) & (ref_maxhws <= 96)
    small_inds = ref_maxhws <= 32

    ref_scales[large_inds] = 192 / ref_maxhws[large_inds]
    ref_scales[medium_inds] = 64 / ref_maxhws[medium_inds]
    ref_scales[small_inds] = 24 / ref_maxhws[small_inds]
    # ref_scales: [0.6925278 , 0.68758106, 0.9831139 ]

    new_locations = np.stack((ref_ys, ref_xs, ref_scales, ref_scores), axis=1)
    new_locations[:, 3] = 1
    return new_locations

def location_nms(locations, thresh=15):
    next_locations = []
    sorted_inds = np.argsort(locations[:, -1])[::-1]

    locations = locations[sorted_inds]
    ys = locations[:, 0]
    xs = locations[:, 1]
    scales = locations[:, 2]

    dist_ys = np.absolute(ys.reshape(-1, 1) - ys.reshape(1, -1))
    # dist_ys.shape: (43, 43)
    dist_xs = np.absolute(xs.reshape(-1, 1) - xs.reshape(1, -1))
    dists = np.minimum(dist_ys, dist_xs)
    # dists.shape: (43, 43)
    ratios = scales.reshape(-1, 1) / scales.reshape(1, -1)
    # ratios.shape: (43, 43)
    while dists.shape[0] > 0:
        next_locations.append(locations[0])

        scale = scales[0]
        dist = dists[0]
        ratio = ratios[0]
        # ratio.shape: (43,)
        # dist.shape: (43,)
        # scale: 0.9831138849258423

        keep = (dist > (thresh / scale)) | (ratio > 1.2) | (ratio < 0.8)
        locations = locations[keep]

        scales = scales[keep]
        dists = dists[keep, :]
        dists = dists[:, keep]
        ratios = ratios[keep, :]
        ratios = ratios[:, keep]
    return np.stack(next_locations) if next_locations else np.zeros((0, 4))
```

## zoom in for locations

```python
detections = [dets]
num_patches = locations.shape[0]
# num_patches: 22

num_crops = 0
# num_iterations = len(att_thresholds)
for ind in range(1, num_iterations + 1):
    if num_patches == 0:
        break

    # att_max_crops: 30
    if num_crops + num_patches > att_max_crops:
        max_crops = min(att_max_crops - num_crops, num_patches)
        locations = locations[:max_crops]

    num_patches = locations.shape[0]
    num_crops += locations.shape[0]
    no_att = ind == num_iterations

    # prepare images on locations
    # dets, locations, atts = get_locs(...)
    images, offsets = prepare_images(db, image, locations, flipped=False)
    images -= im_mean
    images /= im_std

    # no_att = True, decode_func = batch_decode
    dets, atts = decode_func(db, nnet, images, no_att=no_att)
    # dets.shape: (22, 12, 8)
    dets = dets.reshape(num_patches, -1, 8)

    # [64, 64] -> [255, 255]
    rescale_dets_(db, dets)
    # [255, 255] -> [[849, 849], [640, 640]]
    # scales: locations[:, 2]
    remap_dets_(dets, locations[:, 2], offsets)

    dets = dets.reshape(-1, 8)
    # dets.shape: (264, 8)
    keeps = dets[:, 4] > -1
    dets = dets[keeps]
    # dets.shape: (151, 8)

    detections.append(dets)
    if num_crops == att_max_crops:
        break

    if ind < num_iterations:
        att_threshold = att_thresholds[ind]
        att_scale = att_scales[ind]

        next_locations = decode_atts(
            db,
            atts,
            att_scale,
            locations[:, 2],
            offsets,
            height,
            width,
            att_threshold,
            ignore_same=True,
        )

        if ref_dets:
            ref_locations = get_ref_locs(dets)
            next_locations = np.concatenate((next_locations, ref_locations), axis=0)
            next_locations = location_nms(next_locations, thresh=16)

        locations = next_locations
        num_patches = locations.shape[0]
```

## get top detections

Same as CornerNet.



