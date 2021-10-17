# FRCNNDataset

```python
train_dataset = FRCNNDataset(
    train_lines, input_shape, train=True
)
```

## initialize

```python
class FRCNNDataset(Dataset):
    def __init__(self, annotation_lines, input_shape=[600, 600], train=True):
        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)
        self.input_shape = input_shape
        self.train = train
```

## `__getitem__`

```python
    def __getitem__(self, index):
        index = index % self.length
        
        # do data augmentation when training
        image, y = self.get_random_data(
            annotation_line=self.annotation_lines[index], 
            input_shape=self.input_shape[0:2], 
            random=self.train
        )
        
        image = np.transpose(
            # image /= 255.0
            preprocess_input(
                np.array(image, dtype=np.float32)
            ), (2, 0, 1)
        )
        
        box_data = np.zeros((len(y), 5))
        if len(y) > 0:
            box_data[: len(y)] = y
        box = box_data[:, :4]
        label = box_data[:, -1]
        
        return image, box, label
```

## get_random_data

```python
    def get_random_data(
        self, annotation_line, input_shape,
        jitter=0.3, hue=0.1, sat=1.5, val=1.5, random=True,
    ):
        line = annotation_line.split()
        
        # RGB image
        image = Image.open(line[0])
        image = cvtColor(image)
        iw, ih = image.size
        h, w = input_shape
        
        box = np.array([
            np.array(list(map(int, box.split(",")))) 
            for box in line[1:]
        ])
```

### `if not random`

```python
        if not random:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new("RGB", (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)
            # transform image_data to input_shape

            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                # discard invalid box
                box = box[np.logical_and(box_w > 1, box_h > 1)]  

            return image_data, box
```

### `random`

```python
        new_ar = (
            w / h * self.rand(
                1 - jitter, 1 + jitter
            ) / self.rand(
                1 - jitter, 1 + jitter
            )
        )
        scale = self.rand(0.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
            
        image = image.resize((nw, nh), Image.BICUBIC)
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new("RGB", (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image
        
        # random flip
        flip = self.rand() < 0.5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        # color augmentation 
        hue = self.rand(-hue, hue)
        sat = self.rand(1, sat) if self.rand() < 0.5 else 1 / self.rand(1, sat)
        val = self.rand(1, val) if self.rand() < 0.5 else 1 / self.rand(1, val)
        x = cv2.cvtColor(np.array(image, np.float32) / 255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue * 360
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:, :, 0] > 360, 0] = 360
        x[:, :, 1:][x[:, :, 1:] > 1] = 1
        x[x < 0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255

        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip:
                box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]

        return image_data, box
```