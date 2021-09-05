# CornerNet Inference (multi-scale)

## read image

```python
db_ind = db_inds[ind]

image_id = db.image_ids(db_ind)
# image_path: './data/coco/images/minival2014/COCO_val2014_000000397133.jpg'
image_path = db.image_path(db_ind)
# image.shape: (427, 640, 3)
image = cv2.imread(image_path)

top_bboxes[image_id] = cornernet_inference(db, nnet, image)
# top_bboxes[image_id]:
# (1, (4, 5)), (16, (1, 5)), (40, (1, 5)), (42, (6, 5)), (43, (1, 5)), 
# (44, (8, 5)), (45, (32, 5)), (46, (30, 5)), (47, (1, 5)), (59, (1, 5)), 
# (61, (3, 5)), (70, (7, 5)), (72, (4, 5)), (76, (1, 5))
```

## input

```python
def cornernet_inference(db, nnet, image, decode_func=decode):
    ...
    # height, width = 427, 640
    height, width = image.shape[0:2]

    # input_size: [511, 511]
    # output_size: [128, 128]
    height_scale  = (input_size[0] + 1) // output_size[0]
    width_scale   = (input_size[1] + 1) // output_size[1]
    # height_scale, width_scale = 4, 4

    im_mean = torch.cuda.FloatTensor(db.mean).reshape(1, 3, 1, 1)
    im_std  = torch.cuda.FloatTensor(db.std).reshape(1, 3, 1, 1)
    
    detections = []
    # multi-scale inference
    # scales: [0.5, 0.75, 1, 1.25, 1.5]
    for scale in scales:
        ...
```

## inference for each scale

### prepare inference input

```python
new_height = int(height * scale)
new_width  = int(width * scale)
new_center = np.array([new_height // 2, new_width // 2])
# scale: 0.5
# new_height, new_width = 213, 320
# new_center = [106, 160]

inp_height = new_height | 127
inp_width  = new_width  | 127
# inp_height, inp_width: 255, 383

images  = np.zeros((1, 3, inp_height, inp_width), dtype=np.float32)
ratios  = np.zeros((1, 2), dtype=np.float32)
borders = np.zeros((1, 4), dtype=np.float32)
sizes   = np.zeros((1, 2), dtype=np.float32)

out_height, out_width = (inp_height + 1) // height_scale, (inp_width + 1) // width_scale
height_ratio = out_height / inp_height
width_ratio  = out_width  / inp_width
# out_height, out_width: 64, 96
# height_ratio, width_ratio: 0.25098, 0.25065

# image.shape: (427, 640, 3)
resized_image = cv2.resize(image, (new_width, new_height))
resized_image, border, offset = crop_image(
    image=resized_image, 
    center=new_center, 
    size=[inp_height, inp_width]
)
# resized_image.shape: (255, 383, 3)
# border: [ 21., 234.,  31., 351.]
# offset: [-21, -31]
resized_image = resized_image / 255.

images[0]  = resized_image.transpose((2, 0, 1))
borders[0] = border
sizes[0]   = [int(height * scale), int(width * scale)]
ratios[0]  = [height_ratio, width_ratio]

if test_flipped:
    images  = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
    # images.shape: (2, 3, 255, 383)

images  = torch.from_numpy(images).cuda()
images -= im_mean
images /= im_std
```

### decode detections

```python
dets = decode_func(
    nnet, images, K, 
    ae_threshold=ae_threshold, 
    kernel=nms_kernel, 
    num_dets=num_dets
)
# dets.shape: (2, 1000, 8)

if test_flipped:
    dets[1, :, [0, 2]] = out_width - dets[1, :, [2, 0]]
    dets = dets.reshape(1, -1, 8)
    # dets.shape: (1, 2000, 8)

# scale back to the original resolution 
rescale_dets_(dets, ratios, borders, sizes)
dets[:, :, 0:4] /= scale
```

## get top detections

```python
detections = np.concatenate(detections, axis=1)
# detections.shape: (1, 10000, 8)

classes    = detections[..., -1]
classes    = classes[0]
detections = detections[0]

# reject detections with negative scores
keep_inds  = (detections[:, 4] > -1)
detections = detections[keep_inds]
# detections.shape: (2270, 8)
classes    = classes[keep_inds]
# classes.shape: (2270,)

top_bboxes = {}
for j in range(categories):
    keep_inds = (classes == j)
    top_bboxes[j + 1] = detections[keep_inds][:, 0:7].astype(np.float32)
    # merge_box when multi-scale
    if merge_bbox:
        soft_nms_merge(
            top_bboxes[j + 1], 
            Nt=nms_threshold, 
            method=nms_algorithm, 
            weight_exp=weight_exp
        )
    else:
        # top_bboxes[j + 1].shape: (6, 7)
        soft_nms(
            top_bboxes[j + 1], 
            Nt=nms_threshold, 
            method=nms_algorithm
        )
    top_bboxes[j + 1] = top_bboxes[j + 1][:, 0:5]

scores = np.hstack([
    top_bboxes[j][:, -1] for j in range(1, categories + 1)
])
# scores.shape: (2270,)

# max_per_image: 100
if len(scores) > max_per_image:
    kth    = len(scores) - max_per_image
    thresh = np.partition(scores, kth)[kth]
    for j in range(1, categories + 1):
        keep_inds     = (top_bboxes[j][:, -1] >= thresh)
        top_bboxes[j] = top_bboxes[j][keep_inds]
```
