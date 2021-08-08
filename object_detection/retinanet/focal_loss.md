# FocalLoss

### prepare

```python
img_batch, annotations = inputs
...
# fpn features
features = self.fpn([x2, x3, x4])

# bounding boxes
regression_features = [self.regressionModel(feature) for feature in features]
regression = torch.cat(regression_features, dim=1)
# regression_features: [2, 74880, 4], [2, 18720, 4], [2, 4680, 4], [2, 1170, 4], [2, 315, 4]
# regression: (batch_size, K * A, 4) = (2, 99765, 4) 

# classification
classification_features = [
    self.classificationModel(feature) for feature in features
]
classification = torch.cat(classification_features, dim=1)
# classification_features: [2, 74880, 20], [2, 18720, 20], [2, 4680, 20], [2, 1170, 20], [2, 315, 20]
# classification: (batch_size, K * A, num_classes) = (2, 99765, 20) 

anchors = self.anchors(img_batch)
self.focalLoss(classification, regression, anchors, annotations)
```

---

### initialize

```python
alpha = 0.25
gamma = 2.0
batch_size = classifications.shape[0]
classification_losses = []
regression_losses = []

# anchor: (1, 99765, 4) -> (99765, 4) 
anchor = anchors[0, :, :]
anchor_widths = anchor[:, 2] - anchor[:, 0]
anchor_heights = anchor[:, 3] - anchor[:, 1]
anchor_ctr_x = anchor[:, 0] + 0.5 * anchor_widths
anchor_ctr_y = anchor[:, 1] + 0.5 * anchor_heights
```

---

### obtain ground truths

```python
j = 0
classification = classifications[j, :, :]
# classification: [99765, 20]
regression = regressions[j, :, :]
# regression: [99765, 4]

bbox_annotation = annotations[j, :, :]
# only keep valid annotations
bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]
# bbox_annotation: [1, 5] (x1, y1, x2, y2, category)

classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)
```

---

### If there is no positive annotations

```python
# classification.shape: (99765, 20)
alpha_factor = torch.ones(classification.shape).cuda() * alpha
# focal loss for positive target: alpha * ((1 - p) ** gamma) * (-log(p))
# for negative target:
# alpha_neg = 1 - alpha
# p_neg = 1 - p
# focal loss for negative target: (1 - alpha) * (p ** gamma) * (-log(1 - p))
alpha_factor = 1.0 - alpha_factor
focal_weight = classification
focal_weight = alpha_factor * torch.pow(focal_weight, gamma)
bce = -(torch.log(1.0 - classification))
cls_loss = focal_weight * bce
```

---

### when there are positive annotations

#### calculate IoU

```python
IoU = calc_iou(
    anchors[0, :, :], bbox_annotation[:, :4]
)
# IoU.shape: [99765, 1]

IoU_max, IoU_argmax = torch.max(IoU, dim=1)
# IoU_max.shape, IoU_argmax.shape: [99765]
# for each anchor, which ground truth box has the max IoU
```

`IoU_max`:

```
[0.0000, 0.0000, 0.0000,  ..., 0.0048, 0.0363, 0.0677]
```

#### assign ground truths to anchors

Specifically, anchors are assigned to ground-truth object boxes using an intersection-over-union (IoU) threshold of 0.5; and to background if their IoU is in [0, 0.4). 

As each anchor is assigned to at most one object box, we set the corresponding entry in its length K label vector to 1 and all other entries to 0. If an anchor is unassigned, which may happen with overlap in [0.4, 0.5), it is ignored during training.

```python
# classification.shape: (99765, 20)
# targets.shape: (99765, 20)
targets = (torch.ones(classification.shape) * -1).cuda()

# negatives: [0, 0.4)
targets[torch.lt(IoU_max, 0.4), :] = 0
# positives: [0.5,)
positive_indices = torch.ge(IoU_max, 0.5)

num_positive_anchors = positive_indices.sum()
assigned_annotations = bbox_annotation[IoU_argmax, :]
# assigned_annotations: [99765, 5]

# assign one-hot classification target
targets[positive_indices, :] = 0
targets[
    positive_indices, assigned_annotations[positive_indices, 4].long()
] = 1
```

#### compute focal loss for classification

```python
alpha_factor = torch.ones(targets.shape).cuda() * alpha
# positive: alpha, negative: 1 - alpha
alpha_factor = torch.where(
    torch.eq(targets, 1.0), alpha_factor, 1.0 - alpha_factor
)
# positive: 1 - p, negative: p
focal_weight = torch.where(
    torch.eq(targets, 1.0), 1.0 - classification, classification
)
# positive: alpha * ((1 - p) ** gamma), negative: (1 - alpha) * (p ** gamma)
focal_weight = alpha_factor * torch.pow(focal_weight, gamma)
# positive: -log(p), negative: -log(1 - p)
bce = -(
    targets * torch.log(classification)
    + (1.0 - targets) * torch.log(1.0 - classification)
)
cls_loss = focal_weight * bce

# zero out loss for ignored anchors
cls_loss = torch.where(
    torch.ne(targets, -1.0),
    cls_loss,
    torch.zeros(cls_loss.shape).cuda(),
)
# normalize loss by num_positive_anchors
classification_losses.append(
    cls_loss.sum() / torch.clamp(num_positive_anchors.float(), min=1.0)
)
```

#### transform anchors and ground truth boxes (num_positive_anchors > 0)

```python
assigned_annotations = assigned_annotations[positive_indices, :]
anchor_widths_pi = anchor_widths[positive_indices]
anchor_heights_pi = anchor_heights[positive_indices]
anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights

# clip widths to 1
gt_widths = torch.clamp(gt_widths, min=1)
gt_heights = torch.clamp(gt_heights, min=1)
```

#### generate regression targets

For COCO dataset, the regression mean is `[0, 0, 0, 0]`, std is `[0.1, 0.1, 0.2, 0.2]`.

```python
targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
targets_dw = torch.log(gt_widths / anchor_widths_pi)
targets_dh = torch.log(gt_heights / anchor_heights_pi)

targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
# targets: [4, num_positive_anchors]
targets = targets.t()
# targets: [num_positive_anchors, 4]
targets = targets / torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()
```

#### compute regression loss (smooth L1 loss)

https://github.com/yhenon/pytorch-retinanet/issues/127

```python
regression_diff = torch.abs(targets - regression[positive_indices, :])

# smooth L1 loss
regression_loss = torch.where(
    torch.le(regression_diff, 1.0 / 9.0),
    0.5 * 9.0 * torch.pow(regression_diff, 2),
    regression_diff - 0.5 / 9.0,
)
# normalize by num_positive_anchors
regression_losses.append(regression_loss.mean())
```