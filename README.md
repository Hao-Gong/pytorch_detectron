### pytorch > 1.0
### setup：
```
python setup.py install
```

### support efficientDetD0-D7, fpn, cascade rcnn, faster_rcnn,yoloV3,yoloV4,yoloV4-tiny

| name | built |test |
| ------ | ------ | ------ |
| efficientdetD0| 是 |是 |
| efficientdetD1| 是 |是 |
| efficientdetD2| 是 |是 |
| efficientdetD3| 是 |是 |
| efficientdetD4| 是 |是 |
| efficientdetD5| 是 |是 |
| efficientdetD6| 是 |是 |
| efficientdetD7| 是 |是 |
| fpn resnet18|  是 |是 |
| fpn resnet34|  是 |是 |
| fpn resnet50|  是 |是 |
| fpn resnet101|  是 |是 |
| fpn resnet152|  是 |是 |
| cascade_fpn resnet18|  是 |是 |
| cascade_fpn resnet34|  是 |是 |
| cascade_fpn resnet50|  是 |是 |
| cascade_fpn resnet101|  是 |是 |
| cascade_fpn resnet152|  是 |是 |
| faster_rcnn resnet18|  是 |是 |
| faster_rcnn resnet34|  是 |是 |
| faster_rcnn resnet50|  是 |是 |
| faster_rcnn resnet101|  是 |是 |
| faster_rcnn resnet152|  是 |是 |
| yolov3|  是 |是 |
| yolov4|  是 |是 |
| yolov4-tiny|  是 |是 |

## dataset  augmentation：
| train_dataset_transforms | Function |
| ------ | ------ | 
| BrightJitter| -|
| HueSaturationJitter|  - |
| RandomCroper|  - |
| Mosaic|  - |
| Normalizer|  - |
| HorizontalFliper|  - |
| VerticalFlipe|  - |
| RandomBlur|  - |
| NoiseAdder|  - |
|RandomAffiner|  - |
| TransformAnnotXYWH| bbox XminYminXmaxYmax -> XYWH inside 0-1 |

# because of lake of training resource, some test only done on VOC2007 trainval/Test
model    | #GPUs | batch size | lr        | lr_decay | max_epoch     |  time/epoch | mem/GPU | mAP 
---------|--------|-----|--------|-----|-----|-------|--------|-----|-----
[Res-101-Faster-Rcnn-800] | 1 | 4 | 1e-3 | -   | -   |  - hr | - MB  | 71.1% 
[Res-101-FPN-800] | 1 | 8 | 1e-3 | -   | -   |  - hr | - MB  | 74.3% 
[Res-50-FPN-800] | 1 | 8 | 1e-3 | -   | -   |  - hr | - MB  | 71.4% 
[Res-101-Cascader-Rcnn-800] | 1 | 4 | 1e-3 | -   | -   |  - hr | - MB  | 72.7%
[EfficientDet-D0] | 1 | 4 | 1e-3 | -   | -   |  - hr | - MB  | 48.9% 
[EfficientDet-D1] | 1 | 4 | 1e-3 | -   | -   |  - hr | - MB  | 52.1%
[YoloV4-tiny-608] | 1 | 4 | 1e-4 | -   | -   |  - hr | 23.7 MB  | 53.1% 
[YoloV4-608] | 1 | 4 | 1e-4 | -   | -   |  - hr | 257 MB  | 70.3%
[YoloV3-608] | 1 | 4 | 1e-4 | -   | -   |  - hr | 247 MB  | 65.7% 

## VOC2007+2012train/VOC2007Test
model    | #GPUs | batch size | lr        | lr_decay | max_epoch     |  time/epoch | mem/GPU | mAP 
---------|--------|-----|--------|-----|-----|-------|--------|-----|-----
[Res-101-FPN-800] | 1 | 1 | 1e-3 | -   | -   |  - hr | - MB  | 82.9% 
[Res-101-Faster-Rcnn-800] | 1 | 1 | 1e-3 | -   | -   |  - hr | - MB  | 78.8%

### COCO Dataloader is also support, you can modified it yourself