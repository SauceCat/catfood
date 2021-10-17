# Target Creator

## AnchorTargetCreator

```python
class AnchorTargetCreator(object):
    def __init__(
        self, n_sample=256, pos_iou_thresh=0.7, neg_iou_thresh=0.3, pos_ratio=0.5
    ):
        # ...

    def __call__(self, bbox, anchor):
        argmax_ious, label = self._create_label(anchor, bbox)
        if (label > 0).any():
            # create loc target for anchors
            loc = bbox2loc(anchor, bbox[argmax_ious])
            return loc, label
        else:
            return np.zeros_like(anchor), label

    def _calc_ious(self, anchor, bbox):
        ious = bbox_iou(anchor, bbox)
        # ious.shape: (num_anchors, num_gt)

        if len(bbox) == 0:
            return (
                np.zeros(len(anchor), np.int32),
                np.zeros(len(anchor)),
                np.zeros(len(bbox)),
            )
        
        # for each anchor, get the closest gt box and the max iou
        argmax_ious = ious.argmax(axis=1)
        max_ious = np.max(ious, axis=1)
        
        # for each gt, get the closest anchor box
        gt_argmax_ious = ious.argmax(axis=0)
        # ensure anchor is always matched to gt 
        # when it's the closest anchor box to that gt
        for i in range(len(gt_argmax_ious)):
            argmax_ious[gt_argmax_ious[i]] = i

        return argmax_ious, max_ious, gt_argmax_ious
```

To train an RPN, a binary class label (an object or not) is assigned to each anchor.

- **Positive anchors:** 
  - (i) the anchor/anchors with the highest IoU overlap with a ground-truth box, or 
  - (ii) an anchor that has an IoU overlap higher than 0.7 with any ground-truth box.
- **Negative anchors:** if its IoU ratio is lower than 0.3 for all ground-truth boxes.
- **Other anchors:** anchors that are neither positive nor negative do not contribute to the training objective.

```python
    def _create_label(self, anchor, bbox):
        label = np.empty((len(anchor),), dtype=np.int32)
        # ignore label: -1
        label.fill(-1)

        argmax_ious, max_ious, gt_argmax_ious = self._calc_ious(anchor, bbox)
        # argmax_ious.shape: (num_anchors,)
        # max_ious.shape: (num_anchors,)
        # gt_argmax_ious.shape: (num_anchors,)

        # pos_iou_thresh=0.7, neg_iou_thresh=0.3
        label[max_ious < self.neg_iou_thresh] = 0
        label[max_ious >= self.pos_iou_thresh] = 1
        # each gt must have at least one anchor
        # the anchor/anchors with the highest IoU overlap with a ground-truth box
        if len(gt_argmax_ious) > 0:
            label[gt_argmax_ious] = 1

        # sampling positives: 128 samples
        n_pos = int(self.pos_ratio * self.n_sample)
        pos_index = np.where(label == 1)[0]
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(
                pos_index, size=(len(pos_index) - n_pos), replace=False
            )
            label[disable_index] = -1

        # sampling negatives
        n_neg = self.n_sample - np.sum(label == 1)
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(
                neg_index, size=(len(neg_index) - n_neg), replace=False
            )
            label[disable_index] = -1

        return argmax_ious, label
```

![](../images/rpn_box_regress.png)

The bounding box regression adopts the parameterizations of the `4 coordinates`. This can be thought of as bounding-box regression from an anchor box to a nearby ground-truth box.

```python
# loc = bbox2loc(anchor, bbox[argmax_ious])
# argmax_ious.shape: (num_anchors,)
def bbox2loc(src_bbox, dst_bbox):
    width = src_bbox[:, 2] - src_bbox[:, 0]
    height = src_bbox[:, 3] - src_bbox[:, 1]
    ctr_x = src_bbox[:, 0] + 0.5 * width
    ctr_y = src_bbox[:, 1] + 0.5 * height

    base_width = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_height = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_ctr_x = dst_bbox[:, 0] + 0.5 * base_width
    base_ctr_y = dst_bbox[:, 1] + 0.5 * base_height

    eps = np.finfo(height.dtype).eps
    width = np.maximum(width, eps)
    height = np.maximum(height, eps)

    dx = (base_ctr_x - ctr_x) / width
    dy = (base_ctr_y - ctr_y) / height
    dw = np.log(base_width / width)
    dh = np.log(base_height / height)

    loc = np.vstack((dx, dy, dw, dh)).transpose()
    return loc
```

## ProposalTargetCreator

```python
class ProposalTargetCreator(object):
    def __init__(
        self,
        n_sample=128,
        pos_ratio=0.5,
        pos_iou_thresh=0.5,
        neg_iou_thresh_high=0.5,
        neg_iou_thresh_low=0,
    ):
        self.pos_roi_per_image = np.round(self.n_sample * self.pos_ratio)
        # ...

    def __call__(self, roi, bbox, label, loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        roi = np.concatenate((roi.detach().cpu().numpy(), bbox), axis=0)
        # iou between proposals and gt boxes
        iou = bbox_iou(roi, bbox)
        # ious.shape: (num_rois, num_gt)

        if len(bbox) == 0:
            gt_assignment = np.zeros(len(roi), np.int32)
            max_iou = np.zeros(len(roi))
            gt_roi_label = np.zeros(len(roi))
        else:
            # for each roi, get the closest gt box and the largest iou value
            gt_assignment = iou.argmax(axis=1)
            max_iou = iou.max(axis=1)
            # + 1 because index 0 is background
            gt_roi_label = label[gt_assignment] + 1

        # pos_iou_thresh=0.5
        pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]
        pos_roi_per_this_image = int(min(self.pos_roi_per_image, pos_index.size))
        if pos_index.size > 0:
            pos_index = np.random.choice(
                pos_index, size=pos_roi_per_this_image, replace=False
            )
        
        # neg_iou_thresh_high=0.5, neg_iou_thresh_low=0
        neg_index = np.where(
            (max_iou < self.neg_iou_thresh_high) & (max_iou >= self.neg_iou_thresh_low)
        )[0]
        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image
        neg_roi_per_this_image = int(min(neg_roi_per_this_image, neg_index.size))
        if neg_index.size > 0:
            neg_index = np.random.choice(
                neg_index, size=neg_roi_per_this_image, replace=False
            )
            
        keep_index = np.append(pos_index, neg_index)
        sample_roi = roi[keep_index]
        if len(bbox) == 0:
            return sample_roi, np.zeros_like(sample_roi), gt_roi_label[keep_index]
        
        # create loc target for sample_roi
        gt_roi_loc = bbox2loc(sample_roi, bbox[gt_assignment[keep_index]])
        gt_roi_loc = gt_roi_loc / np.array(loc_normalize_std, np.float32)

        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[pos_roi_per_this_image:] = 0
        return sample_roi, gt_roi_loc, gt_roi_label
```
