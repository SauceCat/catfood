# Decode

## Forward

```python
def _test(self, *xs, **kwargs):
    image = xs[0]
    # image.shape: torch.Size([2, 3, 511, 767])
    
    # CornerNet, CornerNet Squeeze
    cnvs  = self.hg(image)
    # cnvs: [torch.Size([2, 256, 128, 192]), torch.Size([2, 256, 128, 192])]
    
    # CornerNet Saccade
    cnvs, ups = self.hg(image)
    # cnvs.shape: [torch.Size([2, 256, 64, 64])] * stacks
    # ups.shape: [
    #     torch.Size([2, 384, 16, 16]), 
    #     torch.Size([2, 384, 32, 32]), 
    #     torch.Size([2, 256, 64, 64])
    # ] * stacks
    ups = [up[self.up_start:] for up in ups]
    
    if not no_att:
        atts = [att_mod_(up) for att_mod_, up in zip(self.att_modules[-1], ups[-1])]
        atts = [torch.sigmoid(att) for att in atts]
        # atts.shape: [
        #	torch.Size([2, 1, 16, 16]), 
        #	torch.Size([2, 1, 32, 32]), 
        #	torch.Size([2, 1, 64, 64])
        # ]

    tl_mod = self.tl_modules[-1](cnvs[-1])
    # tl_mod.shape: torch.Size([2, 256, 128, 192])
    br_mod = self.br_modules[-1](cnvs[-1])
    # br_mod.shape: torch.Size([2, 256, 128, 192])

    tl_heat, br_heat = self.tl_heats[-1](tl_mod), self.br_heats[-1](br_mod)
    # tl_heat.shape: torch.Size([2, 80, 128, 192])
    # br_heat.shape: torch.Size([2, 80, 128, 192])
    
    tl_tag,  br_tag  = self.tl_tags[-1](tl_mod),  self.br_tags[-1](br_mod)
    # tl_tag.shape: torch.Size([2, 1, 128, 192])
    # br_tag.shape: torch.Size([2, 1, 128, 192])
    
    tl_off,  br_off  = self.tl_offs[-1](tl_mod),  self.br_offs[-1](br_mod)
    # tl_off.shape: torch.Size([2, 2, 128, 192])
    # br_off.shape: torch.Size([2, 2, 128, 192])
	
    # CornerNet, CornerNet Squeeze
    outs = [tl_heat, br_heat, tl_tag, br_tag, tl_off, br_off]
    return self._decode(*outs, **kwargs), tl_heat, br_heat, tl_tag, br_tag
	
    # CornerNet Saccade
    if not no_att:
        return self._decode(*outs, **kwargs), atts
        # self._decode(*outs, **kwargs).shape: torch.Size([2, 12, 8])
    else:
        return self._decode(*outs, **kwargs)
```

## inputs

```python
def _decode(
    tl_heat, br_heat, tl_tag, br_tag, tl_regr, br_regr,
    K=100, kernel=1, ae_threshold=1, num_dets=1000, no_border=False
):
    batch, cat, height, width = tl_heat.size()
    # batch, cat, height, width: 2, 80, 128, 192
```

## heatmaps

### nms on heatmaps

```python
tl_heat = torch.sigmoid(tl_heat)
br_heat = torch.sigmoid(br_heat)
```

Use `nn.functional.max_pool2d` with `kernel=3`.

```python
tl_heat = _nms(tl_heat, kernel=3)
br_heat = _nms(br_heat, kernel=3)

def _nms(heat, kernel=1):
    pad = (kernel - 1) // 2
    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep
```

Before `_nms`, `tl_heat[0, 0, :5, :5].cpu().numpy()`:

```
array([[0.0074614 , 0.00106283, 0.00092291, 0.00217108, 0.00233545],
       [0.00390295, 0.00057893, 0.00043272, 0.00101816, 0.00094775],
       [0.00313462, 0.00056406, 0.00034224, 0.00114702, 0.0010591 ],
       [0.0024638 , 0.00056017, 0.00027071, 0.00117814, 0.00104953],
       [0.00238915, 0.00047227, 0.00026424, 0.00118427, 0.00081381]],
      dtype=float32)
```

After `_nms`, `tl_heat[0, 0, :5, :5].cpu().numpy()`:

```
array([[0.0074614, 0.       , 0.       , 0.       , 0.       ],
       [0.       , 0.       , 0.       , 0.       , 0.       ],
       [0.       , 0.       , 0.       , 0.       , 0.       ],
       [0.       , 0.       , 0.       , 0.       , 0.       ],
       [0.       , 0.       , 0.       , 0.       , 0.       ]],
      dtype=float32)
```

### get topK results

```python
tl_scores, tl_inds, tl_clses, tl_ys, tl_xs = _topk(tl_heat, K=100)
br_scores, br_inds, br_clses, br_ys, br_xs = _topk(br_heat, K=100)

# tl_ys.shape, tl_xs.shape, br_ys.shape, br_xs.shape: torch.Size([2, 100])
tl_ys = tl_ys.view(batch, K, 1).expand(batch, K, K)
tl_xs = tl_xs.view(batch, K, 1).expand(batch, K, K)
br_ys = br_ys.view(batch, 1, K).expand(batch, K, K)
br_xs = br_xs.view(batch, 1, K).expand(batch, K, K)
# tl_ys.shape, tl_xs.shape, br_ys.shape, br_xs.shape: torch.Size([2, 100, 100])
```

## offsets

```python
# tl_regr.shape: torch.Size([2, 2, 128, 192])
tl_regr = _tranpose_and_gather_feat(tl_regr, tl_inds)
tl_regr = tl_regr.view(batch, K, 1, 2)
# tl_regr.shape: torch.Size([2, 100, 1, 2])

br_regr = _tranpose_and_gather_feat(br_regr, br_inds)
br_regr = br_regr.view(batch, 1, K, 2)

tl_xs = tl_xs + tl_regr[..., 0]
tl_ys = tl_ys + tl_regr[..., 1]
br_xs = br_xs + br_regr[..., 0]
br_ys = br_ys + br_regr[..., 1]
```

## embeddings

```python
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
width_inds  = (br_xs < tl_xs)
height_inds = (br_ys < tl_ys)

scores[cls_inds]    = -1
scores[dist_inds]   = -1
scores[width_inds]  = -1
scores[height_inds] = -1
```

## get top detections

```python
# scores.shape: torch.Size([2, 100, 100])
scores = scores.view(batch, -1)
# scores.shape: torch.Size([2, 10000])
scores, inds = torch.topk(scores, num_dets=1000)
# scores.shape, inds.shape: torch.Size([2, 1000])
scores = scores.unsqueeze(2)
# scores.shape: torch.Size([2, 1000, 1])

# bboxes.shape: torch.Size([2, 100, 100, 4])
bboxes = bboxes.view(batch, -1, 4)
# bboxes.shape: torch.Size([2, 10000, 4])
bboxes = _gather_feat(bboxes, inds)
# bboxes.shape: torch.Size([2, 1000, 4])

# tl_clses.shape: torch.Size([2, 100, 100])
clses  = tl_clses.contiguous().view(batch, -1, 1)
# clses.shape: torch.Size([2, 10000, 1])
clses  = _gather_feat(clses, inds).float()
# clses.shape: torch.Size([2, 1000, 1])

# tl_scores.shape: torch.Size([2, 100, 100])
tl_scores = tl_scores.contiguous().view(batch, -1, 1)
# tl_scores.shape: torch.Size([2, 10000, 1])
tl_scores = _gather_feat(tl_scores, inds).float()
# tl_scores.shape: torch.Size([2, 1000, 1])

br_scores = br_scores.contiguous().view(batch, -1, 1)
br_scores = _gather_feat(br_scores, inds).float()

detections = torch.cat([bboxes, scores, tl_scores, br_scores, clses], dim=2)
# detections.shape: torch.Size([2, 1000, 8]), 4 + 1 + 1 + 1 + 1 = 8
```
