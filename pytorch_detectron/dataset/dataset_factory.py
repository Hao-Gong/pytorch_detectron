import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# self defined packages
from ..dataset.dataset_voc import*
from ..dataset.augmentation import*
from ..dataset.dataset_coco import*

class trainDatasetParser(object):
    def __init__(self, cfg):
        if cfg["name"]=="VOC" or cfg["name"]=="voc":
            self.dataset=VOCDetection(cfg=cfg,mode="train")
        elif cfg["name"]=="COCO" or cfg["name"]=="coco":
            self.datasetRootPath=cfg["train_coco_data_path"]
            self.cocoImageSets=cfg["train_coco_set"]
            self.dataset=CocoDataset(root_dir=self.datasetRootPath, set_name=self.cocoImageSets,transform=self.transform)

        self.loader=DataLoader(self.dataset,
                                            batch_size=cfg["batch_size"],
                                            num_workers=cfg["num_workers"],
                                            shuffle=cfg["shuffle"],
                                            collate_fn=collater,
                                            pin_memory=True)

class testDatasetParser(object):
    def __init__(self, cfg):
        # self.vocImageSets=cfg["test_voc_set"]
        # cfg["padding_mode"]="leftTop"
        self.dataset= VOCDetection(cfg=cfg,mode="test")
        self.loader=DataLoader(self.dataset,
                                            batch_size=1,
                                            num_workers=cfg["num_workers"],
                                            shuffle=False,
                                            collate_fn=collater,
                                            pin_memory=True)


def singleImageTransform(img,cfg):
    img = img.astype(np.float32)
    normalizer=Normalizer(mean=cfg["mean"],std=cfg["std"],single_img=True)
    resizer=Resizer(common_size=cfg["train_image_resize"],padding_to_rect=cfg["padding_to_rect"],single_img=True)
    totensor=ToTensor(True)
    singleDatasetGenerator=SingleDatasetGenerator()
    transform=transforms.Compose([resizer,normalizer,totensor, singleDatasetGenerator])
    return transform({"img":img})

### additional experiment dataloader area
class trainUnetDatasetParser(object):
    def __init__(self, cfg):
        transformList=parseCfgTransformer(cfg)
        transformList.append(segTargetGenerator())
        self.transform=transforms.Compose(transformList)
        if cfg["name"]=="VOC" or cfg["name"]=="voc":
            self.datasetRootPath=cfg["train_voc_data_path"]
            self.vocImageSets=cfg["train_voc_set"]
            self.dataset=VOCDetection(root=self.datasetRootPath, image_sets=self.vocImageSets,transform=self.transform,target_transform=None,voc_classes=cfg["voc_classes_list"])
        elif cfg["name"]=="COCO" or cfg["name"]=="coco":
            self.datasetRootPath=cfg["train_coco_data_path"]
            self.cocoImageSets=cfg["train_coco_set"]
            # def __init__(self, root_dir, set_name='train2017', transform=None):
            self.dataset=CocoDataset(root_dir=self.datasetRootPath, set_name=self.cocoImageSets,transform=self.transform)

        self.loader=DataLoader(self.dataset,
                                            batch_size=cfg["batch_size"],
                                            num_workers=cfg["num_workers"],
                                            shuffle=cfg["shuffle"],
                                            collate_fn=segmentation_collater,
                                            pin_memory=True)