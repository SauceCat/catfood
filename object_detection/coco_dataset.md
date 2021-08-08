# COCO Dataset

Dataset website: https://cocodataset.org/

Dataset API: https://github.com/cocodataset/cocoapi



## COCO Dataset Format

### Folder structure

```
voc_trainval_2007_coco
├── annotations
│   ├── pascal_train2007.json
│   └── pascal_val2007.json
└── images
    ├── pascal_train2007
    └── pascal_val2007
```

### Object Detection

**File:** `pascal_train2007.json`

**Keys:** `['images', 'type', 'annotations', 'categories']`

- `'images'`: a list of images

  ```
  {
      "file_name": "000012.jpg",
      "height": 333,
      "width": 500,
      "id": 12
  }
  ```

- `'type'`: `'instances'`

- `'annotations'`: a list of annotations

  ```
  {
      "segmentation": [
          [
              155,
              96,
              155,
              270,
              351,
              270,
              351,
              96
          ]
      ],
      "area": 34104,
      "iscrowd": 0,
      "image_id": 12,
      "bbox": [
          155,
          96,
          196,
          174
      ],
      "category_id": 7,
      "id": 1,
      "ignore": 0
  }
  ```

- `'categories'`: a list of categories

  ```
  {
      "supercategory": "none",
      "id": 1,
      "name": "aeroplane"
  }
  ```

---

## Pytorch customized Dataset

```python
class CocoDataset(Dataset):
    """Coco dataset."""

    def __init__(self, root_dir, set_name='train2017', transform=None):
        """
        Args:
            root_dir (string): COCO directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.set_name = set_name
        self.transform = transform
        # voc_trainval_2007_coco/annotations/pascal_train2007.json
        self.data_path = os.path.join(self.root_dir, 'annotations', self.set_name + '.json')
        self.coco      = COCO(self.data_path)
        self.image_ids = self.coco.getImgIds()

        self.load_classes()

    def load_classes(self):
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes             = {}
        self.coco_labels         = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path       = os.path.join(self.root_dir, 'images', self.set_name, image_info['file_name'])
        img = skimage.io.imread(path)

        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)

        return img.astype(np.float32)/255.0

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations     = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation        = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4]  = self.coco_label_to_label(a['category_id'])
            annotations       = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]


    def label_to_coco_label(self, label):
        return self.coco_labels[label]

    def image_aspect_ratio(self, image_index):
        image = self.coco.loadImgs(self.image_ids[image_index])[0]
        return float(image['width']) / float(image['height'])

    def num_classes(self):
        return len(self.classes)
```





