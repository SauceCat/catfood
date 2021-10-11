# Dataset

## Table of contents

- [Dataset Configuration](#dataset_config)
  - [COCO](#dataset_coco)
  - [VOC](#dataset_voc)
- [Data Format](#data_format)

## Dataset Configuration <a name="dataset_config"></a>

### COCO <a name="dataset_coco"></a>

```yaml
# COCO 2017 dataset http://cocodataset.org
# Train command: python train.py --data coco.yaml
# Default dataset location is next to YOLOv3:
#   /parent_folder
#     /coco
#     /yolov3


# download command/URL (optional)
download: bash data/scripts/get_coco.sh

# train and val data as 1) directory: path/images/, 2) file: path/images.txt, or 3) list: [path1/images/, path2/images/]
train: ../coco/train2017.txt  # 118287 images
val: ../coco/val2017.txt  # 5000 images
test: ../coco/test-dev2017.txt  # 20288 of 40670 images, submit to https://competitions.codalab.org/competitions/20794

# number of classes
nc: 80

# class names
names: [ 
	'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 
	'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
	'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush' 
]
```

### VOC <a name="dataset_voc"></a>

```yaml
# PASCAL VOC dataset http://host.robots.ox.ac.uk/pascal/VOC/
# Train command: python train.py --data voc.yaml
# Default dataset location is next to YOLOv3:
#   /parent_folder
#     /VOC
#     /yolov3


# download command/URL (optional)
download: bash data/scripts/get_voc.sh

# train and val data as 1) directory: path/images/, 2) file: path/images.txt, or 3) list: [path1/images/, path2/images/]
train: ../VOC/images/train/  # 16551 images
val: ../VOC/images/val/  # 4952 images

# number of classes
nc: 20

# class names
names: [ 
	'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor' 
]
```

## Data Format <a name="data_format"></a>

```python
# label: class, x_center, y_center, width, height
# x[im_file] = [label, shape, segments]
{
    '../coco128/images/train2017/000000000009.jpg': [
        array([
            [45, 0.47949, 0.68877, 0.95561, 0.5955],
            [45, 0.73652, 0.24719, 0.49887, 0.47642],
            [50, 0.63706, 0.73294, 0.49413, 0.51058],
            [45, 0.33944, 0.4189,  0.67888, 0.7815],
            [49, 0.64684, 0.13255, 0.11805, 0.096937],
            [49, 0.77315, 0.1298,  0.09073, 0.097229],
            [49, 0.6683,  0.22691, 0.13128, 0.1469],
            [49, 0.64286, 0.07921, 0.14806, 0.14806]
        ], dtype=float32), 
        (640, 480), 
        []
    ]
}
```

