# regressBoxes

### initialize

```python
self.regressBoxes = BBoxTransform()
```

The `mean` and `std` is calculated based on COCO dataset. 

```python
self.mean = torch.from_numpy(
    np.array([0, 0, 0, 0]).astype(np.float32)
).cuda()

self.std = torch.from_numpy(
    np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32)
).cuda()
```

---

### forward

```python
transformed_anchors = self.regressBoxes(anchors, regression)
```

#### transform anchors

from `(x1, y1, x2, y2)` -> `(x_ctr, y_ctr, w, h)`

```python
# boxes = anchors
widths = boxes[:, :, 2] - boxes[:, :, 0]
heights = boxes[:, :, 3] - boxes[:, :, 1]
ctr_x = boxes[:, :, 0] + 0.5 * widths
ctr_y = boxes[:, :, 1] + 0.5 * heights
```

similar snippet in `FocalLoss.forward`:

```python
anchor_widths = anchor[:, 2] - anchor[:, 0]
anchor_heights = anchor[:, 3] - anchor[:, 1]
anchor_ctr_x = anchor[:, 0] + 0.5 * anchor_widths
anchor_ctr_y = anchor[:, 1] + 0.5 * anchor_heights
```

#### compute diffs

```python
# deltas = regression
dx = deltas[:, :, 0] * self.std[0] + self.mean[0]
dy = deltas[:, :, 1] * self.std[1] + self.mean[1]
dw = deltas[:, :, 2] * self.std[2] + self.mean[2]
dh = deltas[:, :, 3] * self.std[3] + self.mean[3]
```

related snippet in `FocalLoss.forward`:

```python
gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights

targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
targets_dw = torch.log(gt_widths / anchor_widths_pi)
targets_dh = torch.log(gt_heights / anchor_heights_pi)

targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
targets = targets.t()

# the reason why we need to compute dx as deltas[:, :, 0] * self.std[0] + self.mean[0]
targets = targets / torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()
```

### compute predictions

in format: `(x1, y1, x2, y2)`

```python
# targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
pred_ctr_x = ctr_x + dx * widths
# targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
pred_ctr_y = ctr_y + dy * heights
# targets_dw = torch.log(gt_widths / anchor_widths_pi)
pred_w = torch.exp(dw) * widths
# targets_dh = torch.log(gt_heights / anchor_heights_pi)
pred_h = torch.exp(dh) * heights

# anchor_ctr_x = anchor[:, 0] + 0.5 * anchor_widths
pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
# anchor_ctr_y = anchor[:, 1] + 0.5 * anchor_heights
pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

# [1, 99765] * 4 -> [1, 99765, 4]
pred_boxes = torch.stack(
    [pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2
)
```

---

# ClipBoxes

```python
transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)
```

```python
# boxes = transformed_anchors
# img = img_batch
batch_size, num_channels, height, width = img.shape

# x1 >= 0, y1 >= 0
boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

# x2 <= width, y2 <= height
boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width)
boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height)
```
