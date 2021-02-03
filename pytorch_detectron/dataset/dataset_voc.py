import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# self defined packages
from ..dataset.augmentation import*

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

class VOCDetection(data.Dataset):
    """VOC Detection Dataset Object
    input is image, target is annotation
    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """
    class VOCAnnotationTransform(object):
        """Transforms a VOC annotation into a Tensor of bbox coords and label index
        Initilized with a dictionary lookup of classnames to indexes
        Arguments:
            class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
                (default: alphabetic indexing of VOC's 20 classes)
            keep_difficult (bool, optional): keep difficult instances or not
                (default: False)
            height (int): height
            width (int): width
        """

        def __init__(self,voc_classes, class_to_ind=None, keep_difficult=False):

            self.class_to_ind = class_to_ind or dict(
                zip(voc_classes, range(len(voc_classes))))
            self.keep_difficult = keep_difficult

        def __call__(self, target, width, height):
            """
            Arguments:
                target (annotation) : the target annotation to be made usable
                    will be an ET.Element
            Returns:
                a list containing lists of bounding boxes  [bbox coords, class name]
            """
            res = []
            
#             print(self.class_to_ind)
            for obj in target.iter('object'):
                difficult = int(obj.find('difficult').text) == 1
                if not self.keep_difficult and difficult:
                    continue
                name = obj.find('name').text.lower().strip()
                bbox = obj.find('bndbox')

                pts = ['xmin', 'ymin', 'xmax', 'ymax']
                bndbox = []
                for i, pt in enumerate(pts):
                    cur_pt = float(bbox.find(pt).text) - 1
                    bndbox.append(cur_pt)
                label_idx = self.class_to_ind[name]
                bndbox.append(label_idx)
                res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
                # img_id = target.find('filename').text[:-4]

            return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]

    def __init__(self,cfg,mode="train"):
        self.mode=mode
        self.cfg=cfg
        self.root = cfg["train_voc_data_path"]
        self.voc_classes=cfg["voc_classes_list"]
        if self.mode=="train":
            image_sets=cfg["train_voc_set"]
            aug_transform_list,totensor_tranform_list=parseCfgTransformer(cfg)
            self.aug_transform = transforms.Compose(aug_transform_list)
            self.totensor_transform= transforms.Compose(totensor_tranform_list)
        else:
            image_sets=cfg["test_voc_set"]
            aug_transform_list,totensor_tranform_list=parseCfgTransformer(cfg)
            self.aug_transform = transforms.Compose([Resizer(common_size=cfg["train_image_resize"],padding_to_rect=cfg["padding_to_rect"])])
            self.totensor_transform= transforms.Compose(totensor_tranform_list)

        self.target_transform = self.VOCAnnotationTransform(voc_classes=self.voc_classes)

        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        for (year, name) in image_sets:
            rootpath = osp.join(self.root, 'VOC' + year)
            for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))

    # def augSingleImg(self,index,padding_mode="leftTop"):
    #     img_id = self.ids[index]
    #     target = ET.parse(self._annopath % img_id).getroot()
    #     imgOrigin = cv2.imread(self._imgpath % img_id)
    #     img=imgOrigin.copy()

    #     img = img.astype(np.float32)
    #     height, width, channels = img.shape
    #     target = self.target_transform(target, width, height)
    #     target = np.array(target)
    #     addition_info={"padding_mode":padding_mode}
    #     sample = {'img': img, 'annot': target,'origin_image':imgOrigin,"addition_info":addition_info}
    #     sample = self.aug_transform(sample)
    #     return sample

    def sampleSingleImg(self,index):
        img_id = self.ids[index]
        target = ET.parse(self._annopath % img_id).getroot()
        imgOrigin = cv2.imread(self._imgpath % img_id)
        img=imgOrigin.copy()

        img = img.astype(np.float32)
        height, width, channels = img.shape
        target = self.target_transform(target, width, height)
        target = np.array(target)
        sample = {'img': img, 'annot': target,'origin_image':imgOrigin}
        return sample

    def augSingleImg(self,sample,padding_mode="leftTop"):
        addition_info={"padding_mode":padding_mode}
        sample["addition_info"]=addition_info
        sample = self.aug_transform(sample)
        return sample

    def __getitem__(self, index):
        
        if self.mode=="train":
            if "Mosiac" in self.cfg["train_dataset_transforms"]:
                sample1=self.sampleSingleImg(index)
                sample2=self.sampleSingleImg(int(np.random.uniform(0,len(self.ids))))
                sample3=self.sampleSingleImg(int(np.random.uniform(0,len(self.ids))))
                sample4=self.sampleSingleImg(int(np.random.uniform(0,len(self.ids))))

                if "GrubRandomPadding" in self.cfg["train_dataset_transforms"]:
                    # sample1=GrubRandomPadding(sample1,self.sampleSingleImg(int(np.random.uniform(0,len(self.ids)))))
                    # sample2=GrubRandomPadding(sample2,self.sampleSingleImg(int(np.random.uniform(0,len(self.ids)))))
                    # sample3=GrubRandomPadding(sample3,self.sampleSingleImg(int(np.random.uniform(0,len(self.ids)))))
                    # sample4=GrubRandomPadding(sample4,self.sampleSingleImg(int(np.random.uniform(0,len(self.ids)))))
                    sample1=GrubRandomPadding(sample1)
                    sample2=GrubRandomPadding(sample2)
                    sample3=GrubRandomPadding(sample3)
                    sample4=GrubRandomPadding(sample4)

                sample1=self.augSingleImg(sample1,padding_mode="leftTop")
                sample2=self.augSingleImg(sample2,padding_mode="leftBottom")
                sample3=self.augSingleImg(sample3,padding_mode="rightTop")
                sample4=self.augSingleImg(sample4,padding_mode="rightBottom")

                sample=Mosaic4Pics(sample1,sample2,sample3,sample4)
                return self.totensor_transform(sample)
            else:
                sample1=self.sampleSingleImg(index)
                if "GrubRandomPadding" in self.cfg["train_dataset_transforms"]:
                    sample1=GrubRandomPadding(sample1,self.sampleSingleImg(int(np.random.uniform(0,len(self.ids)))))
                sample1=self.augSingleImg(sample1,padding_mode="randomPadding")
                return self.totensor_transform(sample1)
        else:
            
            sample1=self.sampleSingleImg(index)
            sample1=self.augSingleImg(sample1,padding_mode="leftTop")
            return self.totensor_transform(sample1)


    def __len__(self):
        return len(self.ids)

    def num_classes(self):
        return len(self.voc_classes)

    def label_to_name(self, label):
        return self.voc_classes[label]

    def load_annotations(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        gt = np.array(gt)
        return gt
