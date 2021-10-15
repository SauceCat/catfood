# dataset and dataloader

## dataset

```python
# CocoDataset
dataset_train = CocoDataset(
    parser.coco_path, 
    set_name='train2017',
    transform=transforms.Compose([
        Normalizer(), 
        Augmenter(), 
        Resizer()
    ])
)

# CSVDataset
dataset_train = CSVDataset(
    train_file=parser.csv_train, 
    class_list=parser.csv_classes,
    transform=transforms.Compose([
        Normalizer(), 
        Augmenter(), 
        Resizer()
    ])
)

# difference for validation dataset
transform=transforms.Compose([
    Normalizer(), 
    Resizer()
])
```

### normalize images

```python
class Normalizer(object):
    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        return {
            'img':((image.astype(np.float32) - self.mean) / self.std), 
            'annot': annots
        }
```

### simple augmentation

```python
class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]
            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()
            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp
            sample = {'img': image, 'annot': annots}

        return sample
```

### resize images and detections

```python
class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, min_side=608, max_side=1024):
        image, annots = sample['img'], sample['annot']
        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)
        # rescale the image so the smallest side is min_side
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        image = skimage.transform.resize(
            image, (int(round(rows * scale)), int(round(cols * scale)))
        )
        rows, cols, cns = image.shape
        
        # make sure width and height can be divided by 32
        pad_w = 32 - rows % 32
        pad_h = 32 - cols % 32
        
        # only pad to right side and bottom
        # so there is no offsets for detections
        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)
        annots[:, :4] *= scale

        return {
            'img': torch.from_numpy(new_image), 
            'annot': torch.from_numpy(annots), 
            'scale': scale
        }
```

## dataloader

```python
sampler = AspectRatioBasedSampler(
    dataset_train, 
    batch_size=2, 
    drop_last=False
)
dataloader_train = DataLoader(
    dataset_train, 
    num_workers=3, 
    collate_fn=collater, 
    batch_sampler=sampler
)

sampler_val = AspectRatioBasedSampler(
    dataset_val, 
    batch_size=1, 
    drop_last=False
)
dataloader_val = DataLoader(
    dataset_val, 
    num_workers=3, 
    collate_fn=collater, 
    batch_sampler=sampler_val
)
```

### collate_fn

```python
def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]
        
    widths = [int(s.shape[0]) for s in imgs]
    heights = [int(s.shape[1]) for s in imgs]
    batch_size = len(imgs)
    
    # form a batch of images
    max_width = np.array(widths).max()
    max_height = np.array(heights).max()
    padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)

    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img
    
    # form a batch of detections
    max_num_annots = max(annot.shape[0] for annot in annots)
    if max_num_annots > 0:
        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1
        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1
    
    # BHWC -> BCHW
    padded_imgs = padded_imgs.permute(0, 3, 1, 2)
    
    return {'img': padded_imgs, 'annot': annot_padded, 'scale': scales}
```

### batch_sampler

```python
class AspectRatioBasedSampler(Sampler):
    def __init__(self, data_source, batch_size, drop_last):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.groups = self.group_images()

    def __iter__(self):
        random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def group_images(self):
        # determine the order of the images
        order = list(range(len(self.data_source)))
        order.sort(key=lambda x: self.data_source.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        return [
            [order[x % len(order)] 
             for x in range(i, i + self.batch_size)] 
            for i in range(0, len(order), self.batch_size)
        ]
```