# Custom Transforms

**Reference:** https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

We can write custom transforms as callable classes instead of simple functions so that parameters of the transform need not be passed everytime it's called. We just need to implement `__call__` method and if required, `__init__` method. 

```python
class Normalizer:
    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):

        image, annots = sample["img"], sample["annot"]

        return {
            "img": ((image.astype(np.float32) - self.mean) / self.std),
            "annot": annots,
        }
    
    
class Augmenter:
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):

        if np.random.rand() < flip_x:
            image, annots = sample["img"], sample["annot"]
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {"img": image, "annot": annots}

        return sample
    
    
class Resizer:
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, min_side=608, max_side=1024):
        image, annots = sample["img"], sample["annot"]

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
            image, (int(round(rows * scale)), int(round((cols * scale))))
        )
        rows, cols, cns = image.shape

        pad_w = 32 - rows % 32
        pad_h = 32 - cols % 32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)

        annots[:, :4] *= scale

        return {
            "img": torch.from_numpy(new_image),
            "annot": torch.from_numpy(annots),
            "scale": scale,
        }
```

