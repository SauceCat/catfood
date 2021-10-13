# CornerNet

## Input

```python
# image.shape: (427, 640, 3)
def cornernet_inference(db, nnet, image, decode_func=decode):
    # height, width = 427, 640
    height, width = image.shape[0:2]

    # input_size: [511, 511]
    # output_size: [128, 128]
    height_scale = (input_size[0] + 1) // output_size[0]
    width_scale = (input_size[1] + 1) // output_size[1]
    # height_scale, width_scale = 4, 4

    im_mean = torch.cuda.FloatTensor(db.mean).reshape(1, 3, 1, 1)
    im_std = torch.cuda.FloatTensor(db.std).reshape(1, 3, 1, 1)
    detections = []
```

## Multi-scale inference

Instead of resizing an image to a fixed size, we maintain the original resolution of the image and pad it with zeros before feeding it to CornerNet. Both the original and flipped images are used for testing. 

```python
# multi-scale: scales: [0.5, 0.75, 1, 1.25, 1.5]
for scale in scales:
    # height, width = 427, 640
    # scale: 0.5
    new_height = int(height * scale)
    new_width = int(width * scale)
    new_center = np.array([new_height // 2, new_width // 2])
    # new_height, new_width = 213, 320
    # new_center = [106, 160]

    inp_height = new_height | 127
    inp_width = new_width | 127
    # inp_height, inp_width: 255, 383

    images = np.zeros((1, 3, inp_height, inp_width), dtype=np.float32)
    ratios = np.zeros((1, 2), dtype=np.float32)
    borders = np.zeros((1, 4), dtype=np.float32)
    sizes = np.zeros((1, 2), dtype=np.float32)
	
    # height_scale, width_scale = 4, 4
    out_height = (inp_height + 1) // height_scale
    out_width = (inp_width + 1) // width_scale
    # out_height, out_width: 64, 96
    height_ratio = out_height / inp_height
    width_ratio = out_width / inp_width
    # height_ratio, width_ratio: 0.25098, 0.25065

    # image.shape: (427, 640, 3)
    resized_image = cv2.resize(image, (new_width, new_height))
    # resized_image.shape: (213, 320, 3)
    resized_image, border, offset = crop_image(
        resized_image, new_center, [inp_height, inp_width]
    )
    # resized_image.shape: (255, 383, 3)
    # border: [ 21., 234.,  31., 351.]
    # offset: [-21, -31]
    resized_image = resized_image / 255.

    images[0] = resized_image.transpose((2, 0, 1))
    borders[0] = border
    sizes[0] = [int(height * scale), int(width * scale)]
    ratios[0] = [height_ratio, width_ratio]

    if test_flipped:
        images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
        # images.shape: (2, 3, 255, 383)

    images = torch.from_numpy(images).cuda()
    images -= im_mean
    images /= im_std

    dets = decode_func(
        nnet, 
        images, 
        K, 
        ae_threshold=ae_threshold, 
        kernel=nms_kernel, 
        num_dets=num_dets
    )
    # dets.shape: (2, 1000, 8)
    if test_flipped:
        dets[1, :, [0, 2]] = out_width - dets[1, :, [2, 0]]
        dets = dets.reshape(1, -1, 8)
        # dets.shape: (1, 2000, 8)

    rescale_dets_(dets, ratios, borders, sizes)
    dets[:, :, 0:4] /= scale
    detections.append(dets)
```

## Get detections for each category

We combine the detections from the original and flipped images, and apply soft-nms (Bodla et al., 2017) to suppress redundant detections. Only the top 100 detections are reported.

```python
detections = np.concatenate(detections, axis=1)
# detections.shape: (1, 10000, 8)

classes = detections[..., -1]
classes = classes[0]
detections = detections[0]

# reject detections with negative scores
keep_inds = (detections[:, 4] > -1)
detections = detections[keep_inds]
# detections.shape: (2270, 8)
classes = classes[keep_inds]
# classes.shape: (2270,)

top_bboxes = {}
for j in range(categories):
    keep_inds = (classes == j)
    top_bboxes[j + 1] = detections[keep_inds][:, 0:7].astype(np.float32)
    # merge_box when multi-scale
    if merge_bbox:
        soft_nms_merge(
            top_bboxes[j + 1], 
            Nt=nms_threshold, 
            method=nms_algorithm, 
            weight_exp=weight_exp
        )
    else:
        # top_bboxes[j + 1].shape: (6, 7)
        soft_nms(
            top_bboxes[j + 1], 
            Nt=nms_threshold, 
            method=nms_algorithm
        )
    top_bboxes[j + 1] = top_bboxes[j + 1][:, 0:5]

scores = np.hstack([top_bboxes[j][:, -1] for j in range(1, categories + 1)])
# scores.shape: (2270,)

# max_per_image: 100
if len(scores) > max_per_image:
    kth = len(scores) - max_per_image
    thresh = np.partition(scores, kth)[kth]
    for j in range(1, categories + 1):
        keep_inds = (top_bboxes[j][:, -1] >= thresh)
        top_bboxes[j] = top_bboxes[j][keep_inds]

# final return
return top_bboxes
```

## Decode detections

```python
def decode(nnet, images, K, ae_threshold=0.5, kernel=3, num_dets=1000):
    detections = nnet.test(
        [images], 
        ae_threshold=ae_threshold, 
        test=True, 
        K=K, 
        kernel=kernel, 
        num_dets=num_dets
    )[0]
    return detections.data.cpu().numpy()
```

## `nnet.test`

Unlike many other state-of-the-art detectors, we only use the features from the last layer of the whole network to make predictions.

```python
class hg_net(nn.Module):
    def _test(self, *xs, **kwargs):
        image = xs[0]
        # image.shape: torch.Size([2, 3, 511, 767])
        
        cnvs = self.hg(image)
        # cnvs: [torch.Size([2, 256, 128, 192])] * stacks

        tl_mod = self.tl_modules[-1](cnvs[-1])
        br_mod = self.br_modules[-1](cnvs[-1])
        # tl_mod.shape: torch.Size([2, 256, 128, 192])

        tl_heat, br_heat = self.tl_heats[-1](tl_mod), self.br_heats[-1](br_mod)
        # tl_heat.shape: torch.Size([2, 80, 128, 192])

        tl_tag, br_tag = self.tl_tags[-1](tl_mod), self.br_tags[-1](br_mod)
        # tl_tag.shape: torch.Size([2, 1, 128, 192])
        
        tl_off, br_off = self.tl_offs[-1](tl_mod), self.br_offs[-1](br_mod)
        # tl_off.shape: torch.Size([2, 2, 128, 192])

        outs = [tl_heat, br_heat, tl_tag, br_tag, tl_off, br_off]
        return self._decode(*outs, **kwargs), tl_heat, br_heat, tl_tag, br_tag
```

## _decode

```python
def _decode(
    tl_heat, br_heat, 
    tl_tag, br_tag, 
    tl_regr, br_regr,
    K=100, kernel=1, ae_threshold=1, 
    num_dets=1000, no_border=False
):
    batch, cat, height, width = tl_heat.size()
    # batch, cat, height, width: 2, 80, 128, 192

    tl_heat = torch.sigmoid(tl_heat)
    br_heat = torch.sigmoid(br_heat)

    # perform nms on heatmaps
    tl_heat = _nms(tl_heat, kernel=kernel)
    br_heat = _nms(br_heat, kernel=kernel)

    tl_scores, tl_inds, tl_clses, tl_ys, tl_xs = _topk(tl_heat, K=K)
    br_scores, br_inds, br_clses, br_ys, br_xs = _topk(br_heat, K=K)

    # tl_ys.shape, tl_xs.shape, br_ys.shape, br_xs.shape: torch.Size([2, 100])
    tl_ys = tl_ys.view(batch, K, 1).expand(batch, K, K)
    tl_xs = tl_xs.view(batch, K, 1).expand(batch, K, K)
    br_ys = br_ys.view(batch, 1, K).expand(batch, K, K)
    br_xs = br_xs.view(batch, 1, K).expand(batch, K, K)
    # tl_ys.shape, tl_xs.shape, br_ys.shape, br_xs.shape: torch.Size([2, 100, 100])

    if no_border:
        tl_ys_binds = (tl_ys == 0)
        tl_xs_binds = (tl_xs == 0)
        br_ys_binds = (br_ys == height - 1)
        br_xs_binds = (br_xs == width  - 1)
	
    # add offsets
    if tl_regr is not None and br_regr is not None:
        # tl_regr.shape: torch.Size([2, 2, 128, 192])
        tl_regr = _tranpose_and_gather_feat(tl_regr, tl_inds)
        # tl_regr.shape: torch.Size([2, 100, 2])
        tl_regr = tl_regr.view(batch, K, 1, 2)
        # tl_regr.shape: torch.Size([2, 100, 1, 2])
        br_regr = _tranpose_and_gather_feat(br_regr, br_inds)
        br_regr = br_regr.view(batch, 1, K, 2)

        tl_xs = tl_xs + tl_regr[..., 0]
        tl_ys = tl_ys + tl_regr[..., 1]
        br_xs = br_xs + br_regr[..., 0]
        br_ys = br_ys + br_regr[..., 1]

    # all possible boxes based on top k corners (ignoring class)
    bboxes = torch.stack((tl_xs, tl_ys, br_xs, br_ys), dim=3)
    # bboxes.shape: torch.Size([2, 100, 100, 4])

    # tl_tag.shape: torch.Size([2, 1, 128, 192])
    tl_tag = _tranpose_and_gather_feat(tl_tag, tl_inds)
    tl_tag = tl_tag.view(batch, K, 1)
    # tl_tag.shape: torch.Size([2, 100, 1])

    # br_tag.shape: torch.Size([2, 1, 128, 192])
    br_tag = _tranpose_and_gather_feat(br_tag, br_inds)
    br_tag = br_tag.view(batch, 1, K)
    # br_tag.shape: torch.Size([2, 1, 100])

    dists = torch.abs(tl_tag - br_tag)
    # dists.shape: torch.Size([2, 100, 100])

    tl_scores = tl_scores.view(batch, K, 1).expand(batch, K, K)
    # tl_scores.shape: torch.Size([2, 100, 100])
    br_scores = br_scores.view(batch, 1, K).expand(batch, K, K)
    # br_scores.shape: torch.Size([2, 100, 100])
    scores = (tl_scores + br_scores) / 2
    # scores.shape: torch.Size([2, 100, 100])

    # reject boxes based on classes
    tl_clses = tl_clses.view(batch, K, 1).expand(batch, K, K)
    br_clses = br_clses.view(batch, 1, K).expand(batch, K, K)
    cls_inds = (tl_clses != br_clses)

    # reject boxes based on distances
    dist_inds = (dists > ae_threshold)

    # reject boxes based on widths and heights
    width_inds = (br_xs < tl_xs)
    height_inds = (br_ys < tl_ys)

    if no_border:
        scores[tl_ys_binds] = -1
        scores[tl_xs_binds] = -1
        scores[br_ys_binds] = -1
        scores[br_xs_binds] = -1
	
    # rejections
    scores[cls_inds] = -1
    scores[dist_inds] = -1
    scores[width_inds] = -1
    scores[height_inds] = -1

    scores = scores.view(batch, -1)
    # scores.shape: torch.Size([2, 1000])
    scores, inds = torch.topk(scores, num_dets)
    scores = scores.unsqueeze(2)
    # scores.shape: torch.Size([2, 1000, 1])

    # bboxes.shape: torch.Size([2, 100, 100, 4])
    bboxes = bboxes.view(batch, -1, 4)
    # bboxes.shape: torch.Size([2, 1000, 4])
    bboxes = _gather_feat(bboxes, inds)

    clses = tl_clses.contiguous().view(batch, -1, 1)
    clses = _gather_feat(clses, inds).float()
    # clses.shape: torch.Size([2, 1000, 1])

    # tl_scores.shape: torch.Size([2, 100, 100]) -> torch.Size([2, 10000, 1]) -> torch.Size([2, 1000, 1])
    tl_scores = tl_scores.contiguous().view(batch, -1, 1)
    tl_scores = _gather_feat(tl_scores, inds).float()
    br_scores = br_scores.contiguous().view(batch, -1, 1)
    br_scores = _gather_feat(br_scores, inds).float()

    detections = torch.cat([bboxes, scores, tl_scores, br_scores, clses], dim=2)
    # detections.shape: torch.Size([2, 1000, 8]), 4 + 1 + 1 + 1 + 1 = 8
    return detections
```

We calculate the L1 distances between the embeddings of the top-left and bottom-right corners. Pairs that have distances greater than 0.5 or contain corners from different categories are rejected. The average scores of the top-left and bottom-right corners are used as the detection scores.