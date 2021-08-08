# Evaluation

### obtain prediction boxes

```python
transformed_anchors = self.regressBoxes(anchors, regression)
transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)
```

### prepare container

```python
finalScores = torch.Tensor([]).cuda()
finalAnchorBoxesIndexes = torch.Tensor([]).long().cuda()
finalAnchorBoxesCoordinates = torch.Tensor([]).cuda()
```

### loop and finalize results for each category

```python
# loop through each category
for i in range(classification.shape[2]):
    # scores: [1, 99765] -> [99765 (total number of anchor boxes)]
    scores = torch.squeeze(classification[:, :, i])
    
    # only keep box above a certain threshold
    scores_over_thresh = scores > 0.05
    if scores_over_thresh.sum() == 0:
        # no boxes to NMS, just continue
        continue
    scores = scores[scores_over_thresh]
    
    # anchorBoxes: [1, 99765, 4] -> [99765, 4]
    anchorBoxes = torch.squeeze(transformed_anchors)
    # anchorBoxes: [3 (scores_over_thresh.sum()), 4]
    anchorBoxes = anchorBoxes[scores_over_thresh]
    # from torchvision.ops import nms, iou_threshold=0.5
    anchors_nms_idx = nms(anchorBoxes, scores, 0.5)
	
    finalScores = torch.cat((finalScores, scores[anchors_nms_idx]))
    finalAnchorBoxesIndexesValue = torch.tensor(
        [i] * anchors_nms_idx.shape[0]
    ).cuda()
    finalAnchorBoxesIndexes = torch.cat(
        (finalAnchorBoxesIndexes, finalAnchorBoxesIndexesValue)
    )
    finalAnchorBoxesCoordinates = torch.cat(
        (finalAnchorBoxesCoordinates, anchorBoxes[anchors_nms_idx])
    )
```

### return

```python
return [finalScores, finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates]
# finalScores.shape: [882]
# finalAnchorBoxesIndexes.shape: [882]
# finalAnchorBoxesCoordinates.shape: [882, 4]
```

`finalScores[:10]`:

```
tensor([0.0651, 0.1280, 0.1249, 0.1203, 0.0999, 0.0953, 0.0924, 0.0912, 0.0851,
        0.0796], device='cuda:0')
```

`finalAnchorBoxesIndexes[:10]`:

```
tensor([0, 1, 1, 1, 1, 1, 1, 1, 1, 1], device='cuda:0')
```

`finalAnchorBoxesCoordinates[:10]`:

```
tensor([[ 58.6583, 143.2424, 777.3287, 545.4927],
        [  0.0000, 338.8678, 471.8625, 598.1266],
        [153.6316, 251.8049, 554.2400, 577.6003],
        [280.1880, 266.3284, 693.0161, 575.8690],
        [363.5241, 305.7518, 565.9952, 513.8391],
        [441.0872, 309.3282, 769.8794, 598.6996],
        [ 12.1046, 369.1140, 281.6197, 590.1002],
        [228.3918, 351.3098, 474.9415, 587.5016],
        [  0.0000, 362.1358, 147.4707, 587.0435],
        [  0.0000, 123.4422, 790.8958, 565.5623]], device='cuda:0')
```
