# Augmentation

```python
class SSDAugmentation(object):
    def __init__(self, size=300, mean=(104, 117, 123)):
        self.mean = mean
        self.size = size
        self.augment = Compose([
            # image.astype(np.float32)
            ConvertFromInts(),
            
            # boxes[:, 0] *= width, boxes[:, 2] *= width
            # boxes[:, 1] *= height, boxes[:, 3] *= height
            ToAbsoluteCoords(),
            
            # adjust contrast, saturation, hue, brightness, lighting
            PhotometricDistort(),
            
            # randomly expand image dimension
            Expand(self.mean),
            
            # random crop image
            RandomSampleCrop(),
            
            # random horizontal flip
            RandomMirror(),
            
            # boxes[:, 0] /= width, boxes[:, 2] /= width
            # boxes[:, 1] /= height, boxes[:, 3] /= height
            ToPercentCoords(),
            
            # image = cv2.resize(image, (self.size, self.size))
            Resize(self.size),
            
            # image -= self.mean
            SubtractMeans(self.mean)
        ])

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)
```

## PhotometricDistort

```python
class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, boxes, labels):
        im = image.copy()
        im, boxes, labels = self.rand_brightness(im, boxes, labels)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, boxes, labels = distort(im, boxes, labels)
        return self.rand_light_noise(im, boxes, labels)
```

### RandomContrast

Multiply image values with a random number (`[0.5, 1.5]`).

```python
# self.lower=0.5, self.upper=1.5
# assert self.upper >= self.lower, "contrast upper must be >= lower."
# assert self.lower >= 0, "contrast lower must be non-negative."
if random.randint(2):
    alpha = random.uniform(self.lower, self.upper)
    image *= alpha
```

### ConvertColor

```python
# self.current='BGR', self.transform='HSV'
if self.current == 'BGR' and self.transform == 'HSV':
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
elif self.current == 'HSV' and self.transform == 'BGR':
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
else:
    raise NotImplementedError
```

![](../images/hsv.png)

### RandomSaturation

Multiply the image saturation values with a random number (`[0.5, 1.5]`).

```python
# self.lower=0.5, self.upper=1.5
# assert self.upper >= self.lower, "contrast upper must be >= lower."
# assert self.lower >= 0, "contrast lower must be non-negative."
# image in HSV
if random.randint(2):
    image[:, :, 1] *= random.uniform(self.lower, self.upper)
```

### RandomHue

Add a random number to the image hue values (`[0.0, 360.0]`).

```python
# self.delta=18.0
# assert delta >= 0.0 and delta <= 360.0
# image in HSV
if random.randint(2):
    image[:, :, 0] += random.uniform(-self.delta, self.delta)
    image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
    image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
```

### RandomBrightness

Add a random number to the image values (`[0.0, 255.0]`).

```python
# self.delta=32
# assert delta >= 0.0 and delta <= 255.0
if random.randint(2):
    delta = random.uniform(-self.delta, self.delta)
    image += delta
```

### RandomLightingNoise

Randomly shuffle image channels.

```python
self.perms = ((0, 1, 2), (0, 2, 1),
              (1, 0, 2), (1, 2, 0),
              (2, 0, 1), (2, 1, 0))

if random.randint(2):
    swap = self.perms[random.randint(len(self.perms))]
    image = image[:, :, swap]
```

## Expand

```python
class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, boxes, labels):
        if random.randint(2):
            return image, boxes, labels

        height, width, depth = image.shape
        ratio = random.uniform(1, 4)
        # left: [0, 3 * width], top: [0, 3 * height]
        left = random.uniform(0, width*ratio - width)
        top = random.uniform(0, height*ratio - height)

        expand_image = np.zeros(
            (int(height*ratio), int(width*ratio), depth),
            dtype=image.dtype
        )
        expand_image[:, :, :] = self.mean
        expand_image[
            int(top):int(top + height), 
            int(left):int(left + width)
        ] = image
        image = expand_image

        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))

        return image, boxes, labels
```

## RandomSampleCrop

```python
class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """
    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2], :]

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(
                    current_boxes[:, :2], rect[:2]
                )
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(
                    current_boxes[:, 2:], rect[2:]
                )
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                return current_image, current_boxes, current_labels
```

```python
def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ??? B / A ??? B = A ??? B / (area(A) + area(B) - A ??? B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]
```

## RandomMirror

```python
_, width, _ = image.shape
if random.randint(2):
    image = image[:, ::-1]
    boxes = boxes.copy()
    boxes[:, 0::2] = width - boxes[:, 2::-2]
```