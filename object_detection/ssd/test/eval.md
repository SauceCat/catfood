# eval

## load network

```python
# load net
# +1 for background
num_classes = len(labelmap) + 1

# initialize SSD
net = build_ssd('test', 300, num_classes)            
net.load_state_dict(torch.load(args.trained_model))
net.eval()
print('Finished loading model!')

if args.cuda:
    net = net.cuda()
    cudnn.benchmark = True
```

## load dataset

```python
dataset = VOCDetection(
    args.voc_root, [('2007', 'test')],
    BaseTransform(size=300, mean=(104, 117, 123)),
    # [[xmin, ymin, xmax, ymax, label_ind], ... ]
    VOCAnnotationTransform()
)

class BaseTransform:
    def __init__(self, size, mean):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        return base_transform(image, self.size, self.mean), boxes, labels
    
def base_transform(image, size, mean):
    x = cv2.resize(image, (size, size)).astype(np.float32)
    x -= mean
    x = x.astype(np.float32)
    return x
```

## evaluation

```python
test_net(
    save_folder=args.save_folder, 
    net=net, 
    cuda=args.cuda, dataset,
    transform=BaseTransform(net.size, dataset_mean), 
    top_k=args.top_k, 
    im_size=300,
    thresh=args.confidence_threshold
)

def test_net(
    save_folder, net, cuda, dataset, transform, top_k,
    im_size=300, thresh=0.05
):
    num_images = len(dataset)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [
        [
            [] for _ in range(num_images)
        ] for _ in range(len(labelmap) + 1)
    ]

    for i in range(num_images):
        im, gt, h, w = dataset.pull_item(i)

        x = Variable(im.unsqueeze(0)).cuda()
        detections = net(x).data
        # detections.shape: (batch_size, num_classes, top_k, 5)

        # skip j = 0, because it's the background class
        for j in range(1, detections.size(1)):
            dets = detections[0, j, :]
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)
            if dets.size(0) == 0:
                continue
            boxes = dets[:, 1:]
            # get absolute box values
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            scores = dets[:, 0].cpu().numpy()
            cls_dets = np.hstack(
                (
                    boxes.cpu().numpy(),
                    scores[:, np.newaxis]
                )
            ).astype(np.float32, copy=False)
            all_boxes[j][i] = cls_dets

    evaluate_detections(all_boxes, output_dir, dataset)
    
def evaluate_detections(box_list, output_dir, dataset):
    write_voc_results_file(box_list, dataset)
    do_python_eval(output_dir)
    
def do_python_eval(output_dir='output', use_07=True):
    # evaluate per label
    for i, cls in enumerate(labelmap):
        rec, prec, ap = voc_eval(
            detpath=filename, 
            annopath=annopath, 
            imagesetfile=imgsetpath.format(set_type), 
            classname=cls, 
            cachedir=cachedir,
            ovthresh=0.5, 
            use_07_metric=use_07
        )
```

## voc_eval

```python
# evaluate per class
def voc_eval(classname=cls, ovthresh=0.5, ...):    
    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        npos = npos + sum(~difficult)
        class_recs[imagename] = {
            'bbox': np.array([x['bbox'] for x in R]),
            'difficult': np.array([x['difficult'] for x in R]).astype(np.bool),
            'det': [False] * len(R)
        }

    # lines: detection results
    if any(lines) == 1:
        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        
        # loop for all detections
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)
            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin, 0.)
                ih = np.maximum(iymax - iymin, 0.)
                inters = iw * ih
                uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                       (BBGT[:, 2] - BBGT[:, 0]) *
                       (BBGT[:, 3] - BBGT[:, 1]) - inters)
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        # one detection box can be only used once
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
    else:
        rec = -1.
        prec = -1.
        ap = -1.

    return rec, prec, ap
```

## voc_ap

```python
def voc_ap(rec, prec, use_07_metric=True):
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
```

