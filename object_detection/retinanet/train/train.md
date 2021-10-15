# train

## datasets

```python
dataset_train = CSVDataset(
    train_file=parser.csv_train, 
    class_list=parser.csv_classes,
    transform=transforms.Compose([
        Normalizer(), Augmenter(), Resizer()
    ])
)

dataset_val = CSVDataset(
    train_file=parser.csv_val, 
    class_list=parser.csv_classes,
    transform=transforms.Compose([
        Normalizer(), Resizer()
    ])
)
```

## dataloader

Details: [dataloader](dataloader.md)

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

## model

```python
if parser.depth == 18:
    retinanet = model.resnet18(
        num_classes=dataset_train.num_classes(), pretrained=True
    )
elif parser.depth == 34:
    retinanet = model.resnet34(
        num_classes=dataset_train.num_classes(), pretrained=True
    )
elif parser.depth == 50:
    retinanet = model.resnet50(
        num_classes=dataset_train.num_classes(), pretrained=True
    )
elif parser.depth == 101:
    retinanet = model.resnet101(
        num_classes=dataset_train.num_classes(), pretrained=True
    )
elif parser.depth == 152:
    retinanet = model.resnet152(
        num_classes=dataset_train.num_classes(), pretrained=True
    )
else:
    raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

retinanet = retinanet.cuda()
retinanet = torch.nn.DataParallel(retinanet).cuda()
retinanet.training = True
```

## optimizer, lr_scheduler

```python
optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=3, verbose=True
)
```

## training loop

```python
loss_hist = collections.deque(maxlen=500)
print('Num training images: {}'.format(len(dataset_train)))

for epoch_num in range(parser.epochs):
    retinanet.train()
    retinanet.module.freeze_bn()

    epoch_loss = []
    for iter_num, data in enumerate(dataloader_train):
        try:
            optimizer.zero_grad()
            
            # compute loss
            classification_loss, regression_loss = retinanet([
                data['img'].cuda().float(), 
                data['annot']
            ])
            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()
            loss = classification_loss + regression_loss

            if bool(loss == 0):
                continue
            loss.backward()
            
            # optimize
            torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)
            optimizer.step()

            loss_hist.append(float(loss))
            epoch_loss.append(float(loss))

            print(
                'Epoch: {} | Iteration: {} | \
                Classification loss: {:1.5f} | \
                Regression loss: {:1.5f} | \
                Running loss: {:1.5f}'.format(
                    epoch_num, 
                    iter_num, 
                    float(classification_loss), 
                    float(regression_loss), 
                    np.mean(loss_hist)
                )
            )
            del classification_loss
            del regression_loss
        except Exception as e:
            print(e)
            continue
    
    # run validation per epoch
    mAP = csv_eval.evaluate(dataset_val, retinanet)
    scheduler.step(np.mean(epoch_loss))
    torch.save(
        retinanet.module, 
        '{}_retinanet_{}.pt'.format(parser.dataset, epoch_num)
    )
```

## save model

```python
retinanet.eval()
torch.save(retinanet, 'model_final.pt')
```

