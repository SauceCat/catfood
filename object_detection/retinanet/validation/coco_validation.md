# COCO Validation

## dataset

```python
dataset_val = CocoDataset(
    parser.coco_path, 
    set_name='val2017',
    transform=transforms.Compose([
        Normalizer(), Resizer()
    ])
)
```

## model

```python
retinanet = model.resnet50(
    num_classes=dataset_val.num_classes(), 
    pretrained=True
).cuda()
retinanet.load_state_dict(torch.load(parser.model_path))
retinanet = torch.nn.DataParallel(retinanet).cuda()

retinanet.training = False
retinanet.eval()
retinanet.module.freeze_bn()
```

## evaluation

```python
coco_eval.evaluate_coco(dataset_val, retinanet)
```

## evaluate_coco

utilize `pycocotools.cocoeval`

```python
def evaluate_coco(dataset, model, threshold=0.05):
    model.eval()
    
    with torch.no_grad():
        # start collecting results
        results = []
        image_ids = []

        for index in range(len(dataset)):
            data = dataset[index]
            scale = data['scale']

            # run network
            # HWC -> CHW, in training, this is done in collater
            # in model, 
            # return [finalScores, finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates]
            scores, labels, boxes = model(
                data['img'].permute(2, 0, 1).cuda().float().unsqueeze(dim=0)
            )
            scores = scores.cpu()
            labels = labels.cpu()
            boxes  = boxes.cpu()

            # correct boxes for image scale
            boxes /= scale

            if boxes.shape[0] > 0:
                # change to (x, y, w, h) (MS COCO standard)
                boxes[:, 2] -= boxes[:, 0]
                boxes[:, 3] -= boxes[:, 1]

                # compute predicted labels and scores
                for box_id in range(boxes.shape[0]):
                    score = float(scores[box_id])
                    label = int(labels[box_id])
                    box = boxes[box_id, :]

                    # scores are sorted, so we can break
                    if score < threshold:
                        break

                    # append detection for each positively labeled class
                    image_result = {
                        'image_id'    : dataset.image_ids[index],
                        'category_id' : dataset.label_to_coco_label(label),
                        'score'       : float(score),
                        'bbox'        : box.tolist(),
                    }

                    # append detection to results
                    results.append(image_result)

            # append image to list of processed images
            image_ids.append(dataset.image_ids[index])

            # print progress
            print('{}/{}'.format(index, len(dataset)), end='\r')

        if not len(results):
            return

        # write output
        json.dump(
            results, 
            open('{}_bbox_results.json'.format(dataset.set_name), 'w'), 
            indent=4
        )

        # load results in COCO evaluation tool
        # from pycocotools.coco import COCO
        # dataset.coco = COCO(
        #     os.path.join(
        #         dataset.root_dir, 'annotations', 
        #         'instances_' + dataset.set_name + '.json'
        #     )
        # )
        coco_true = dataset.coco
        coco_pred = coco_true.loadRes(
            '{}_bbox_results.json'.format(dataset.set_name)
        )

        # run COCO evaluation
        # from pycocotools.cocoeval import COCOeval
        coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        model.train()

        return
```

