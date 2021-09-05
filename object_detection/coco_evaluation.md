# COCO Evaluation

### convert to coco format

```python
def convert_to_coco(self, all_bboxes):
    # (x1, y1, x2, y2) -> (x1, y1, width, height)
    detections = []
    for image_id in all_bboxes:
        # image_id: 'COCO_val2014_000000397133.jpg'
        coco_id = self._eval_ids[image_id]
        # coco_id: 397133
        for cls_ind in all_bboxes[image_id]:
            # cls_ind: 1
            category_id = self._cls2coco[cls_ind]
            # category_id: 1

            # all_bboxes[image_id][cls_ind].shape: (4, 5)
            for bbox in all_bboxes[image_id][cls_ind]:
                # bbox: [386.74838  ,  68.51108  , 498.9558   , 347.0734   ,   0.8976282]
                bbox[2] -= bbox[0]
                bbox[3] -= bbox[1]
                # bbox: [386.74838  ,  68.51108  , 112.20743  , 278.56232  ,   0.8976282]

                score = bbox[4]
                bbox  = list(map(self._to_float, bbox[0:4]))
                # bbox: [386.75, 68.51, 112.21, 278.56]

                detection = {
                    "image_id": coco_id,
                    "category_id": category_id,
                    "bbox": bbox,
                    "score": float("{:.2f}".format(score))
                }

                detections.append(detection)
    return detections
```

## evaluate through `COCOeval`

```python
result_json = os.path.join(result_dir, "results.json")
detections  = db.convert_to_coco(top_bboxes)
with open(result_json, "w") as f:
    json.dump(detections, f)

cls_ids   = list(range(1, categories + 1))
image_ids = [db.image_ids(ind) for ind in db_inds]
db.evaluate(result_json, cls_ids, image_ids)

def evaluate(self, result_json, cls_ids, image_ids):
    from pycocotools.cocoeval import COCOeval

    if self._split == "testdev":
        return None

    coco = self._coco

    eval_ids = [self._eval_ids[image_id] for image_id in image_ids]
    cat_ids  = [self._cls2coco[cls_id] for cls_id in cls_ids]

    coco_dets = coco.loadRes(result_json)
    coco_eval = COCOeval(coco, coco_dets, "bbox")
    coco_eval.params.imgIds = eval_ids
    coco_eval.params.catIds = cat_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats[0], coco_eval.stats[12:]
```

