# intersection-over-union (IoU)

```python
def calc_iou(a, b):
    # a.shape: [99765, 4]
    # b.shape: [1, 4]
    # (x1, y1, x2, y2)
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    # area.shape: [1]
	
    # min(x2) - max(x1)
    iw = torch.min(
        torch.unsqueeze(a[:, 2], dim=1), b[:, 2]
    ) - torch.max(
        torch.unsqueeze(a[:, 0], 1), b[:, 0]
    )
    # min(y2) - max(y1)
    ih = torch.min(
        torch.unsqueeze(a[:, 3], dim=1), b[:, 3]
    ) - torch.max(
        torch.unsqueeze(a[:, 1], 1), b[:, 1]
    )
    # iw.shape, ih.shape: [99765, 1]
    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)
    # intersection.shape: [99765, 1]
    intersection = iw * ih
	
    # (x2 - x1) * (y2 - y1) + w * h - intersection
    ua = (
        torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1)
        + area
        - iw * ih
    )
    ua = torch.clamp(ua, min=1e-8)
	
    IoU = intersection / ua
    # IoU.shape: [99765, 1]
    return IoU
```

