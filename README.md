### 注意：考虑到gitlhub的上传方便性，pretrained model未上传
### pytorch > 1.0
### 运行：
```
python setup.py install
```

## 本工具箱提供多种通用 目标检测 工具，为将来平台提供目标检测工具：

### 目前支持efficientDetD0-D7, fpn, cascade rcnn, faster_rcnn,yoloV3,yoloV4,yoloV4-tiny

| 网络名称 | 是否构建完成 |是否测试完成 |
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

## dataset 数据增强方式，柔性化数据增强方式，通过删减配置文件中train_dataset_transforms的增强方式名字开启或关闭功能：
| train_dataset_transforms | Function |
| ------ | ------ | 
| BrightJitter| -|
| HueSaturationJitter|  - |
| RandomCroper|  - |
| Mosaic|  马赛克 |
| Normalizer|  - |
| HorizontalFliper|  - |
| VerticalFlipe|  - |
| RandomBlur|  - |
| NoiseAdder|  - |
|RandomAffiner|  - |
| TransformAnnotXYWH| 标注框从 XminYminXmaxYmax -> XYWH并且缩放0-1之间 |

使用方式：
For efficientDetD0-D7, fpn, cascade rcnn, faster_rcnn:
```
"train_dataset_transforms":["BrightJitter","HueSaturationJitter","RandomCroper","RandomSelfMosaic","Normalizer","HorizontalFliper", "VerticalFliper", "Resizer"]  
```
For yoloV3,yoloV4,yoloV4-tiny:
```
"train_dataset_transforms":["BrightJitter","HueSaturationJitter","RandomCroper","RandomSelfMosaic","Normalizer","HorizontalFliper", "VerticalFliper", "Resizer","TransformAnnotXYWH"]  
```

## 因为整个工程都把朝参放到了config中，你可以使用autodlml-server来训练～