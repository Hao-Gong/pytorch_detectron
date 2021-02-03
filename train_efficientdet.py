
# coding: utf-8
# #### EfficientDet: Scalable and Efficient Object Detection 论文连接： https://arxiv.org/abs/1911.09070
from pytorch_detectron.detector import detector

#定义好数据集的类别名字，需要与标注中的保持一致
voc_set_class_list=['aeroplane', 'bicycle', 'bird', 'boat',
                     'bottle', 'bus', 'car', 'cat', 'chair',
                     'cow', 'diningtable', 'dog', 'horse',
                     'motorbike', 'person', 'pottedplant',
                     'sheep', 'sofa', 'train', 'tvmonitor']

#定义好VOC格式数据集的路径
voc_set_path="/home/gong/datasets/VOCtrainval_06-Nov-2007/VOCdevkit"

#配置efficientdet网络训练参数
cfg_efficientdet={
    # 网络名字，可以自己根据项目取名
    "name":"efficientdet",
    # 网络的主干网络类型，此处可以使用d0,d1,d2,d3,d4,d5,d6,d7
    # 数字越大，网络拟合能力越强，计算复杂度越高，一般任务推荐 d0-d3
    # Deep Residual Learning for Image Recognition 论文地址 https://arxiv.org/abs/1905.11946
    "backbone": "d0",
    # 是否使用GPU，请使用True
    "use_cuda":True,
    # 选择哪一块显卡训练，一般在多卡服务器上
    "CUDA_VISIBLE_DEVICES":"0",
    # 是否只训练检测头部分，默认 False
    "train_head_only":False,
    # 训练使用的优化器，有sgd adam adamw，这里比较适合sgd和adamw
    "optim":'adamw',
    # 初始化learning rate
    "lr":1e-3,
    # learning rate 调节策略 StepLR LambdaLR ReduceLROnPlateau，这里适合ReduceLROnPlateau
    "lr_scheduler":"ReduceLROnPlateau",
    # 一共训练的次数
    "epochs":20,
    # 每隔多少次在测试集上获取一次评分，并且保存评分和模型
    "evaluate_step":10,
    # 保存训练好模型的路径
    "saved_path": "trained_models",
    # efficientdet参数，不用修改
    "anchor_ratios" :[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)],
    "anchor_scales" : [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)],
}

cfg_efficientdet_datasets={
    # dataset的格式名字VOC格式还是coco格式，这里推荐使用VOC格式
    "name":"VOC",
    # VOC格式数据集位置
    "train_voc_data_path":voc_set_path,
    # VOC格式数据集，里面用于train的txt路径，有多个合并可以类似[('2007', 'trainval')，('2012', 'trainval')]
    "train_voc_set":[('2007', 'trainval')],
    # VOC格式数据集，里面用于test的txt路径，就是用于测评网络
    "test_voc_set":[('2007', 'test')],
    # 数据在线随机增强方法，参考base_function_tutorials
    "train_dataset_transforms":["BrightJitter","HueSaturationJitter","RandomCroper","RandomBlur","NoiseAdder","GrubRandomPadding","Mosiac","Normalizer","HorizontalFliper", "Resizer","ToTensor"], 
    # 训练的batch_size，根据显存大小选择
    "batch_size":1,
    # 数据pipeline的多线程读取数据的数量
    "num_workers":1,
    # 数据是否打乱顺序
    "shuffle":True,
    # 图片输入的归一化，此处使用imageNet的默认方式
    "mean":[0.485, 0.456, 0.406],
    "std":[0.229, 0.224, 0.225],
    # 选择输入是rgb或bgr，imageNet的默认方式是rgb
    "color_mode":"rgb",
    # 数据集类别，这里coco和voc都沿用了这个名字
    "voc_classes_list" :voc_set_class_list,
    # 网络输入图像的大小，必须在下面中选
    "train_image_resize":512,# [512,640,768,896,1024,1280,1536],
    # resize图像后，是不是保持正方形输入
    "padding_to_rect":True,
}

global_config={
    "datasets":cfg_efficientdet_datasets,
    # 网络名字，必须在"efficientdet","fpn","cascade_fpn","faster_rcnn","yolov3","yolov4","yolov4_tiny"中
    "model_name":"efficientdet",
    # 可以使用的网络类型
    "model_list":["efficientdet","fpn","cascade_fpn","faster_rcnn","yolov3","yolov4","yolov4_tiny"],
    "model_cfg":cfg_efficientdet
}

# 通过导入cfg初始化一个检测网络
Detector=detector(global_config)
Detector.trainval()

