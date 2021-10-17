# predict

```python
class FRCNN(object):
    def __init__(self, **kwargs):
        self.std = torch.Tensor(
            [0.1, 0.1, 0.2, 0.2]
        ).repeat(self.num_classes + 1)[None]
        self.bbox_util = DecodeBox(self.std, self.num_classes)
        
        # get model, self.net
        self.generate()
```

## detect_image

```python
    def detect_image(self, image):
        # image_data = ...
        
        with torch.no_grad():
            images = torch.from_numpy(image_data).cuda()
            roi_cls_locs, roi_scores, rois, _ = self.net(images)
            # decode box
            results = self.bbox_util.forward(
                roi_cls_locs,
                roi_scores,
                rois,
                image_shape,
                input_shape,
                nms_iou=self.nms_iou,
                confidence=self.confidence,
            )
```

## DecodeBox

```python
class DecodeBox:
    def forward(
        self, roi_cls_locs, roi_scores, rois,
        image_shape, input_shape,
        nms_iou=0.3, confidence=0.5,
    ):
        results = []
        bs = len(roi_cls_locs)
        rois = rois.view((bs, -1, 4))
        # rois.shape: (batch_size, num_rois, 4)

        for i in range(bs):
            # self.std: [0.1, 0.1, 0.2, 0.2]
            roi_cls_loc = roi_cls_locs[i] * self.std
            roi_cls_loc = roi_cls_loc.view([-1, self.num_classes, 4])

            # num_rois, 4 -> num_rois, 1, 4 -> num_rois, num_classes, 4
            roi = rois[i].view((-1, 1, 4)).expand_as(roi_cls_loc)
            
            # loc2bbox(src_bbox, loc): adjust src_bbox with loc
            cls_bbox = loc2bbox(
                roi.contiguous().view((-1, 4)), 
                roi_cls_loc.contiguous().view((-1, 4))
            )
            cls_bbox = cls_bbox.view([-1, (self.num_classes), 4])
            cls_bbox[..., [0, 2]] = (cls_bbox[..., [0, 2]]) / input_shape[1]
            cls_bbox[..., [1, 3]] = (cls_bbox[..., [1, 3]]) / input_shape[0]

            roi_score = roi_scores[i]
            # get class score
            prob = F.softmax(roi_score, dim=-1)

            results.append([])
            for c in range(1, self.num_classes):
                c_confs = prob[:, c]
                c_confs_m = c_confs > confidence

                if len(c_confs[c_confs_m]) > 0:
                    boxes_to_process = cls_bbox[c_confs_m, c]
                    confs_to_process = c_confs[c_confs_m]

                    keep = nms(boxes_to_process, confs_to_process, nms_iou)
                    good_boxes = boxes_to_process[keep]
                    confs = confs_to_process[keep][:, None]
                    labels = (
                        (c - 1) * torch.ones((len(keep), 1)).cuda()
                        if confs.is_cuda
                        else torch.ones((len(keep), 1))
                    )
                    c_pred = torch.cat((good_boxes, confs, labels), dim=1).cpu().numpy()
                    results[-1].extend(c_pred)
```



