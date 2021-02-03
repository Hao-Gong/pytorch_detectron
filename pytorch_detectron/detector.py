import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import logging

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import pdb

from .base_functions.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from .base_functions.utils.net_utils import weights_normal_init, save_net, load_net, \
      adjust_learning_rate, save_checkpoint, clip_gradient

# import resnet_fpn
from .detection_models.fpn.fpn import resnet_fpn
#  import resnet_faster_rcnn
from .detection_models.faster_rcnn.faster_rcnn import resnet_faster_rcnn
#  import fpn_cascade
from .detection_models.fpn_cascade.fpn_cascade import resnet_fpn_cascade
# yolov3 v4 v4-tiny
from .detection_models.yolo.yolo_loss import YOLOLoss,YOLOTinyLoss
from .detection_models.yolo.yoloV3 import YoloV3Body
from .detection_models.yolo.yoloV4_tiny import YoloV4TinyBody
from .detection_models.yolo.yoloV4 import YoloV4Body
from .detection_models.yolo.yolo_utils import non_max_suppression, bbox_iou, DecodeBox,get_yolo_detections,get_yolo_tiny_detections
# efficientdet D0-D7
from .detection_models.efficientDet.model import EfficientDetBackbone,ModelWithLoss
from .detection_models.efficientDet.utils import init_efficientdet_weights
from .base_functions.utils.visualizations import *
from .detection_models.efficientDet.utils import *
from .base_functions.utils.net_utils import save_net, load_net, vis_detections,vis_detections_better

from .dataset.evaluator import *
from .dataset.dataset_factory import *
from .base_functions.rpn.bbox_transform import bbox_transform_inv,clip_boxes
from .lib_c.model.roi_layers import nms
from .base_functions.train_utils.optim import *

class detector(object):
    def __init__(self, input):
        self.net_loaded=False
        if isinstance(input,dict):
            self.load_cfg(input)
            self.generate_model_from_cfg()
        elif isinstance(input,str):
            self.load_net(input)
        else:
            print("No Network Initiated!")

    def __call__(self, input):
        self.net_loaded=False
        if isinstance(input,dict):
            self.load_cfg(input)
            self.generate_model_from_cfg()
        elif isinstance(input,str):
            self.load_net(input)
        else:
            print("No Network Initiated!")
            
    def load_cfg(self,global_config):
        self.net_loaded=True
        self.global_config=global_config
        # init config 
        np.random.seed(cfg.RNG_SEED)
        for key in global_config:
          if key not in cfg:
            cfg.setdefault(key,global_config[key])

        for key in global_config["model_cfg"]:
          if key not in cfg:
            print("adding model_cfg ",key,"------->",global_config["model_cfg"][key])
            cfg["model_cfg"].setdefault(key,global_config["model_cfg"][key])

        for key in global_config["datasets"]:
          if key not in cfg:
            print("adding datasets ",key,"------->",global_config["datasets"][key])
            cfg["datasets"].setdefault(key,global_config["datasets"][key])

        for key in global_config["model_cfg"]:
          if key in cfg and isinstance(cfg[key],(dict)):
            for sub_key in global_config["model_cfg"][key]:
              if sub_key in cfg[key]:
                cfg[key][sub_key]=global_config["model_cfg"][key][sub_key]
                print(key,sub_key,cfg[key][sub_key],"------->",global_config["model_cfg"][key][sub_key])

          elif key in cfg:
            print(key,cfg[key],"------->",global_config["model_cfg"][key])
            cfg[key]=global_config["model_cfg"][key]
        cfg.USE_GPU_NMS = global_config["model_cfg"]["use_cuda"]

        self.cfg=cfg
        print(self.cfg)

    def generate_model_from_cfg(self):
        # choose model
        if(self.cfg["model_name"]=="fpn"):
            if self.cfg["datasets"]["voc_classes_list"][0]!='__background__':
                self.cfg["datasets"]["voc_classes_list"]= ['__background__']+cfg["datasets"]["voc_classes_list"]
            self.model=resnet_fpn(self.cfg)
        elif (self.cfg["model_name"]=="faster_rcnn"):
            if self.cfg["datasets"]["voc_classes_list"][0]!='__background__':
                self.cfg["datasets"]["voc_classes_list"]= ['__background__']+cfg["datasets"]["voc_classes_list"]
            self.model=resnet_faster_rcnn(self.cfg)
        elif (self.cfg["model_name"]=="fpn_cascade"):
            if self.cfg["datasets"]["voc_classes_list"][0]!='__background__':
                self.cfg["datasets"]["voc_classes_list"]= ['__background__']+cfg["datasets"]["voc_classes_list"]
            self.model=resnet_fpn_cascade(self.cfg)
        elif (self.cfg["model_name"]=="yolov3" or self.cfg["model_name"]=="yoloV3"):
            if 'TransformAnnotXYWH' not in self.cfg["datasets"]['train_dataset_transforms']:
                self.cfg["datasets"]['train_dataset_transforms'].append('TransformAnnotXYWH')
            self.model=YoloV3Body(self.cfg)
        elif (self.cfg["model_name"]=="yolov4" or self.cfg["model_name"]=="yoloV4"):
            if 'TransformAnnotXYWH' not in self.cfg["datasets"]['train_dataset_transforms']:
                self.cfg["datasets"]['train_dataset_transforms'].append('TransformAnnotXYWH')
            self.model=YoloV4Body(self.cfg)
        elif (self.cfg["model_name"]=="yolov4_tiny" or self.cfg["model_name"]=="yoloV4_tiny"):
            if 'TransformAnnotXYWH' not in self.cfg["datasets"]['train_dataset_transforms']:
                self.cfg["datasets"]['train_dataset_transforms'].append('TransformAnnotXYWH')
            self.model=YoloV4TinyBody(self.cfg)
        elif (self.cfg["model_name"]=="efficientdet"):
                compound_coef=int(self.cfg["model_cfg"]["backbone"].split("d")[1])
                self.model = EfficientDetBackbone(num_classes=len(self.cfg["datasets"]["voc_classes_list"]), compound_coef=compound_coef,
                                ratios=self.cfg["model_cfg"]["anchor_ratios"], scales=self.cfg["model_cfg"]["anchor_scales"],load_weights=self.cfg["model_cfg"]["pretrained"])
        else:
            print("ERROR! Please use models in efficientdet,fpn,cascade_fpn,faster_rcnn,yolov3,yolov4,yolov4_tiny!!!")
            return 0

        if self.cfg["model_cfg"]["use_cuda"]:
            self.model.cuda()

        print(self.model)
        
        # print(self.cfg["model_cfg"])


    # training the model
    def trainval(self):
        self.train_dataset=trainDatasetParser(self.cfg["datasets"])
        self.test_dataset=testDatasetParser(self.cfg["datasets"])
        self.optSched=optimizerSchedulerController(self.model,self.cfg)
        self.evaluator=evaluator('VOC',self.cfg["datasets"]["train_voc_data_path"],tuple(self.cfg["datasets"]["voc_classes_list"]),self.cfg["datasets"]["test_voc_set"])
        if(self.cfg["model_name"]=="fpn"):
            self.train_fpn()
        elif(self.cfg["model_name"]=="faster_rcnn"):
            self.train_faster_rcnn()
        elif (self.cfg["model_name"]=="fpn_cascade"):
            self.train_fpn_cascade()
        elif (self.cfg["model_name"]=="yolov3" or self.cfg["model_name"]=="yoloV3"):
            self.train_yolov3()
        elif (self.cfg["model_name"]=="yolov4" or self.cfg["model_name"]=="yoloV4"):
            self.train_yolov4()
        elif (self.cfg["model_name"]=="yolov4_tiny" or self.cfg["model_name"]=="yoloV4_tiny"):
            self.train_yolov4_tiny()
        elif (self.cfg["model_name"]=="efficientdet"):
            self.train_efficientdet()
        else:
            print("ERROR! Please use models in efficientdet,fpn,cascade_fpn,faster_rcnn,yolov3,yolov4,yolov4_tiny!!!")
            return 0

    # evaluate the model performance 
    def evaluate(self,all_boxes):
        print('Evaluating detections,file stored in: output')
        return self.evaluator.evaluate_detections(all_boxes)
    
    # inference the model
    def predict_one_image(self,image_path,draw=True):
        if self.net_loaded==False:
            print("no net instance,use load_net() or load_cfg() first")
            return 0

        imgOrigin=cv2.imread(image_path)
        data=singleImageTransform(imgOrigin,cfg["datasets"])
        self.model.eval()

        if(self.cfg["model_name"]=="fpn"):
            pred_boxes=self.test_fpn(data)
        elif(self.cfg["model_name"]=="faster_rcnn"):
            pred_boxes=self.test_faster_rcnn(data)
        elif (self.cfg["model_name"]=="fpn_cascade"):
            pred_boxes=self.test_fpn_cascade(data)
        elif (self.cfg["model_name"]=="yolov3" or self.cfg["model_name"]=="yoloV3"):
            pred_boxes=self.test_yolov3(data)
        elif (self.cfg["model_name"]=="yolov4" or self.cfg["model_name"]=="yoloV4"):
            pred_boxes=self.test_yolov4(data)
        elif (self.cfg["model_name"]=="yolov4_tiny" or self.cfg["model_name"]=="yoloV4_tiny"):
            pred_boxes=self.test_yolov4_tiny(data)
        elif (self.cfg["model_name"]=="efficientdet"):
            pred_boxes=self.test_efficientdet(data)
        else:
            print("ERROR! Please use models in efficientdet,fpn,cascade_fpn,faster_rcnn,yolov3,yolov4,yolov4_tiny!!!")
            return 0

        if draw:
            self.drawImage=self.draw(imgOrigin,pred_boxes)
            return pred_boxes,self.drawImage
        else:
            return pred_boxes,imgOrigin

    # export annotation
    def export_xml_annotation(self,image_path,output_xml_folder):
        pred_boxes,imgOrigin=self.predict_one_image(image_path,draw=False)
        height,width,c=imgOrigin.shape
        imageName=image_path.split("/")[-1].split(".")[0]
        xml_O=xmlObj(imageName, width,height)
        for det in pred_boxes:
            class_name,xmin,ymin,xmax,ymax,score=det
            x=xmin
            y=ymin
            w=xmax-xmin
            h=ymax-ymin
            xml_O.add_anno(class_name,x,y,w,h)
        xml_O.save(output_xml_folder)
        # return xml_O.root

    # val the model
    def val(self):
        # self.train_dataset=trainDatasetParser(self.cfg["datasets"])
        self.test_dataset=testDatasetParser(self.cfg["datasets"])
        # self.optSched=optimizerSchedulerController(self.model,self.cfg)
        self.evaluator=evaluator('VOC',self.cfg["datasets"]["train_voc_data_path"],tuple(self.cfg["datasets"]["voc_classes_list"]),self.cfg["datasets"]["test_voc_set"])
        if self.net_loaded:
            if(self.cfg["model_name"]=="fpn"):
                self.val_fpn()
            elif(self.cfg["model_name"]=="faster_rcnn"):
                self.val_faster_rcnn()
            elif (self.cfg["model_name"]=="fpn_cascade"):
                self.val_fpn_cascade()
            elif (self.cfg["model_name"]=="yolov3" or self.cfg["model_name"]=="yoloV3"):
                self.val_yolov3()
            elif (self.cfg["model_name"]=="yolov4" or self.cfg["model_name"]=="yoloV4"):
                self.val_yolov4()
            elif (self.cfg["model_name"]=="yolov4_tiny" or self.cfg["model_name"]=="yoloV4_tiny"):
                self.val_yolov4_tiny()
            elif (self.cfg["model_name"]=="efficientdet"):
                self.val_efficientdet()
            else:
                print("ERROR! Please use models in efficientdet,fpn,cascade_fpn,faster_rcnn,yolov3,yolov4,yolov4_tiny!!!")
                return 0
        else:
            print("no net instance,use load_net() or load_cfg() first")

    # load trained net weights
    def load_net(self,path):
        self.net_loaded=True
        load_dict=torch.load(path)
        cfg=load_dict["cfg"]
        self.load_cfg(cfg)
        self.generate_model_from_cfg()
        self.model.load_state_dict(load_dict["model"])
        print("loading finidhed!!")

    #========================================================Different trainers 
    def train_faster_rcnn(self):
        self.train_fpn()

    def train_fpn(self):
        train_loader=self.train_dataset.loader
        val_loader=self.test_dataset.loader
        num_iter_per_epoch = len(train_loader)

        for epoch in range(self.cfg["model_cfg"]["epochs"]):
            epoch_loss=[]
            self.model.train()

            for step, data in enumerate(train_loader):
                start = time.time()
                if self.cfg["model_cfg"]["use_cuda"]:
                    im_data = data["images"].cuda().float()
                    annot = data["annot"].cuda()
                    im_info=data["image_info"].cuda()
                    num_boxes=data["num_boxes"].cuda()
                else:
                    im_data = data["images"].float()
                    annot = data["annot"]
                    im_info=data["image_info"]
                    num_boxes=data["num_boxes"]

                self.model.zero_grad()
                _, _, _, rpn_loss_cls, rpn_loss_box, RCNN_loss_cls, RCNN_loss_bbox, rois_label = self.model(im_data, im_info, annot, num_boxes)

                loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
                    + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()

                epoch_loss.append(loss.item())
                # backward
                self.optSched.optimizer.zero_grad()
                loss.backward()
                self.optSched.optimizer.step()

                loss_rpn_cls = rpn_loss_cls.item()
                loss_rpn_box = rpn_loss_box.item()
                loss_rcnn_cls = RCNN_loss_cls.item()
                loss_rcnn_box = RCNN_loss_bbox.item()
                fg_cnt = torch.sum(rois_label.data.ne(0))
                bg_cnt = rois_label.data.numel() - fg_cnt

                end = time.time()
                print(epoch,num_iter_per_epoch,":",step)
                print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end-start))
                print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
                              % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))

                if self.cfg["model_cfg"]["lr_scheduler"]=='LambdaLR':
                    self.optSched.schedulerStep()
            if self.cfg["model_cfg"]["lr_scheduler"]!='LambdaLR':
                self.optSched.schedulerStep()

            # evaluator
            if epoch%self.cfg["model_cfg"]["evaluate_step"]==0 and epoch>0:
                self.val_fpn()

    def train_fpn_cascade(self):
        train_loader=self.train_dataset.loader
        val_loader=self.test_dataset.loader
        num_iter_per_epoch = len(train_loader)

        for epoch in range(self.cfg["model_cfg"]["epochs"]):
            epoch_loss=[]
            self.model.train()

            for step, data in enumerate(train_loader):
                start = time.time()
                if self.cfg["model_cfg"]["use_cuda"]:
                    im_data = data["images"].cuda().float()
                    annot = data["annot"].cuda()
                    im_info=data["image_info"].cuda()
                    num_boxes=data["num_boxes"].cuda()
                else:
                    im_data = data["images"].float()
                    annot = data["annot"]
                    im_info=data["image_info"]
                    num_boxes=data["num_boxes"]

                self.model.zero_grad()
                _, _, _, rpn_loss_cls, rpn_loss_box, \
                        RCNN_loss_cls, RCNN_loss_bbox, RCNN_loss_cls_2nd, RCNN_loss_bbox_2nd, RCNN_loss_cls_3rd, RCNN_loss_bbox_3rd, \
                        roi_labels = self.model(im_data, im_info, annot, num_boxes)

                loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
                              + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean() \
                              + RCNN_loss_cls_2nd.mean() + RCNN_loss_bbox_2nd.mean() \
                              + RCNN_loss_cls_3rd.mean() + RCNN_loss_bbox_3rd.mean()

                epoch_loss.append(loss.item())
                # backward
                self.optSched.optimizer.zero_grad()
                loss.backward()
                self.optSched.optimizer.step()

                end = time.time()
                loss_rpn_cls = rpn_loss_cls.item()
                loss_rpn_box = rpn_loss_box.item()
                loss_rcnn_cls = RCNN_loss_cls.item()
                loss_rcnn_box = RCNN_loss_bbox.item()

                loss_rcnn_cls_2nd = RCNN_loss_cls_2nd.item()
                loss_rcnn_box_2nd = RCNN_loss_bbox_2nd.item()
                loss_rcnn_cls_3rd = RCNN_loss_cls_3rd.item()
                loss_rcnn_box_3rd = RCNN_loss_bbox_3rd.item()

                fg_cnt = torch.sum(roi_labels.data.ne(0))
                bg_cnt = roi_labels.data.numel() - fg_cnt

                print("[epoch %2d][iter %4d/%4d] loss: %.4f" \
                        % (epoch, step, num_iter_per_epoch, loss.item()), )
                print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end - start), )

                print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f, rcnn_cls_2nd: %.4f, "
                        "rcnn_box_2nd %.4f, rcnn_cls_3rd: %.4f, rcnn_box_3rd %.4f" % (loss_rpn_cls, loss_rpn_box,
                    loss_rcnn_cls, loss_rcnn_box, loss_rcnn_cls_2nd, loss_rcnn_box_2nd, loss_rcnn_cls_3rd, loss_rcnn_box_3rd))
                # 
                if self.cfg["model_cfg"]["lr_scheduler"]=='LambdaLR':
                    self.optSched.schedulerStep()
            if self.cfg["model_cfg"]["lr_scheduler"]!='LambdaLR':
                self.optSched.schedulerStep()

            # evaluator
            if epoch%self.cfg["model_cfg"]["evaluate_step"]==0 and epoch>0:
                self.val_fpn_cascade()

    def train_yolov4(self):
          self.train_yolov3()

    def train_yolov3(self):
        train_loader=self.train_dataset.loader
        val_loader=self.test_dataset.loader
        num_iter_per_epoch = len(train_loader)

        yolo_losses = []
        anchors=np.array(cfg["model_cfg"]["anchors"])
        for i in range(3):
            yolo_losses.append(YOLOLoss(anchors=np.reshape(anchors,[-1,2]),\
                                                    num_classes=len(cfg["datasets"]["voc_classes_list"]), \
                                                    img_size=(cfg["datasets"]["train_image_resize"], cfg["datasets"]["train_image_resize"]), label_smooth=False))


        for epoch in range(self.cfg["model_cfg"]["epochs"]):
            epoch_loss=[]
            self.model.train()
            for step, data in enumerate(train_loader):
                start = time.time()
                if self.cfg["model_cfg"]["use_cuda"]:
                    imgs = data["images"].cuda().float()
                else:
                    imgs = data["images"].float()

                annot = data["annot"].numpy()
                annot_targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in annot]

                outputs = self.model(imgs)
                # print(outputs[0])
                losses = []
                loss_conf=0
                loss_cls=0
                loss_loc=0
                for i in range(3):
                    loss_item = yolo_losses[i](outputs[i], annot_targets)
                    losses.append(loss_item[0])
                    print("yolo",i,"conf loss:{:.4f}".format(loss_item[1]),"cls loss:{:.4f}".format(loss_item[2]),"loc loss:{:.4f}".format(loss_item[3]))
                    loss_conf+=loss_item[1]
                    loss_cls+=loss_item[2]
                    loss_loc+=loss_item[3]
                loss = sum(losses)

                self.optSched.optimizer.zero_grad()
                loss.backward()
                self.optSched.optimizer.step()

                print("epoch:",epoch,"iter:",num_iter_per_epoch,"/",step,"loss:{:.4f}".format(loss.item()),"conf loss:{:.4f}".format(loss_conf),"cls loss:{:.4f}".format(loss_cls),"loc loss:{:.4f}".format(loss_loc))

                if self.cfg["model_cfg"]["lr_scheduler"]=='LambdaLR':
                    self.optSched.schedulerStep()
            if self.cfg["model_cfg"]["lr_scheduler"]!='LambdaLR':
                self.optSched.schedulerStep()

            # evaluator
            if epoch%self.cfg["model_cfg"]["evaluate_step"]==0 and epoch>0:
                self.val_yolov3()

    def train_yolov4_tiny(self):
        train_loader=self.train_dataset.loader
        val_loader=self.test_dataset.loader
        num_iter_per_epoch = len(train_loader)

        yolo_losses = []
        anchors=np.array(cfg["model_cfg"]["anchors"])
        for i in range(2):
            yolo_losses.append(YOLOTinyLoss(anchors=np.reshape(anchors,[-1,2]),\
                                                    num_classes=len(cfg["datasets"]["voc_classes_list"]), \
                                                    img_size=(cfg["datasets"]["train_image_resize"], cfg["datasets"]["train_image_resize"]), label_smooth=False))

        for epoch in range(self.cfg["model_cfg"]["epochs"]):
            epoch_loss=[]
            self.model.train()
            for step, data in enumerate(train_loader):
                start = time.time()
                if self.cfg["model_cfg"]["use_cuda"]:
                    imgs = data["images"].cuda().float()
                else:
                    imgs = data["images"].float()

                annot = data["annot"].numpy()
                annot_targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in annot]

                outputs = self.model(imgs)
                # print(outputs[0])
                losses = []
                loss_conf=0
                loss_cls=0
                loss_loc=0
                for i in range(2):
                    loss_item = yolo_losses[i](outputs[i], annot_targets)
                    losses.append(loss_item[0])
                    print("yolo",i,"conf loss:{:.4f}".format(loss_item[1]),"cls loss:{:.4f}".format(loss_item[2]),"loc loss:{:.4f}".format(loss_item[3]))
                    loss_conf+=loss_item[1]
                    loss_cls+=loss_item[2]
                    loss_loc+=loss_item[3]
                loss = sum(losses)

                self.optSched.optimizer.zero_grad()
                loss.backward()
                self.optSched.optimizer.step()

                print("epoch:",epoch,"iter:",num_iter_per_epoch,"/",step,"loss:{:.4f}".format(loss.item()),"conf loss:{:.4f}".format(loss_conf),"cls loss:{:.4f}".format(loss_cls),"loc loss:{:.4f}".format(loss_loc))

                if self.cfg["model_cfg"]["lr_scheduler"]=='LambdaLR':
                    self.optSched.schedulerStep()
            if self.cfg["model_cfg"]["lr_scheduler"]!='LambdaLR':
                self.optSched.schedulerStep()

            # evaluator
            if epoch%self.cfg["model_cfg"]["evaluate_step"]==0 and epoch>0:
                self.val_yolov4_tiny()

    def train_efficientdet(self):
        train_loader=self.train_dataset.loader
        num_iter_per_epoch = len(train_loader)
        modelTrain = ModelWithLoss(self.model, debug=False)
        self.optSched=optimizerSchedulerController(modelTrain,self.cfg)

        if self.cfg["model_cfg"]["use_cuda"]:
            modelTrain = modelTrain.cuda()

        for epoch in range(self.cfg["model_cfg"]["epochs"]):
            epoch_loss=[]
            modelTrain.train()
            for idx, data in enumerate(train_loader):
                if self.cfg["model_cfg"]["use_cuda"]:
                    imgs = data["images"].cuda().float()
                    annot = data["annot"].cuda()
                else:
                    imgs = data["images"].float()
                    annot = data["annot"]
                self.optSched.optimizer.zero_grad()

                current_time=time.time()
                cls_loss, reg_loss = modelTrain(imgs=imgs, annotations= annot,obj_list=self.cfg["datasets"]["voc_classes_list"])
                cls_loss = cls_loss.mean()
                reg_loss = reg_loss.mean()

                loss = cls_loss + reg_loss
                if loss == 0 or not torch.isfinite(loss):
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(modelTrain.parameters(), 0.1)
                self.optSched.optimizer.step()
                epoch_loss.append(loss.item())

                print("epoch:",epoch,"iter:",num_iter_per_epoch,"/",idx,"loss:",loss.item())

                if self.cfg["model_cfg"]["lr_scheduler"]=='LambdaLR':
                    self.optSched.schedulerStep()
            if self.cfg["model_cfg"]["lr_scheduler"]!='LambdaLR':
                self.optSched.schedulerStep()

            # evaluator
            if epoch%self.cfg["model_cfg"]["evaluate_step"]==0 and epoch>0:
                self.val_efficientdet()

    #========================================================Different vals
    def val_fpn(self):
        val_loader=self.test_dataset.loader
        num_images = len(val_loader)
        self.model.eval()
        thresh=self.cfg["model_cfg"]["thresh"]

        start = time.time()
        max_per_image = 100

        all_boxes = [[[] for _ in range(num_images)]
                  for _ in range( len(self.cfg["datasets"]["voc_classes_list"]))]    

        for i, data in enumerate(val_loader):
            det_tic = time.time()
            if self.cfg["model_cfg"]["use_cuda"]:
                im_data = data["images"].cuda().float()
                annot = data["annot"].cuda()
                im_info=data["image_info"].cuda()
                num_boxes=data["num_boxes"].cuda()
            else:
                im_data = data["images"].float()
                annot = data["annot"]
                im_info=data["image_info"]
                num_boxes=data["num_boxes"]

            with torch.no_grad():
                rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_box, RCNN_loss_cls, RCNN_loss_bbox, rois_label = self.model(im_data, im_info, annot, num_boxes)

            scores = cls_prob.data
            boxes = rois.data[:, :, 1:5]

            if cfg["TEST"]["BBOX_REG"]:
                # Apply bounding-box regression deltas
                box_deltas = bbox_pred.data
                if self.cfg["model_cfg"]["use_cuda"]:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(self.cfg["TRAIN"]["BBOX_NORMALIZE_STDS"]).cuda() \
                                + torch.FloatTensor(self.cfg["TRAIN"]["BBOX_NORMALIZE_MEANS"]).cuda()
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(self.cfg["TRAIN"]["BBOX_NORMALIZE_STDS"])\
                                + torch.FloatTensor(self.cfg["TRAIN"]["BBOX_NORMALIZE_MEANS"])

                box_deltas = box_deltas.view(1, -1, 4 * len(self.cfg["datasets"]["voc_classes_list"]))
                pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
            else:
                # Simply repeat the boxes, once for each class
                pred_boxes = np.tile(boxes, (1, scores.shape[1]))

            pred_boxes /= im_info[0][2].item()
            scores = scores.squeeze()
            pred_boxes = pred_boxes.squeeze()
            det_toc = time.time()
            detect_time = det_toc - det_tic
            misc_tic = time.time()

            empty_array = np.transpose(np.array([[],[],[],[],[]]), (1,0))
            for j in range(1, len(self.cfg["datasets"]["voc_classes_list"])):
                inds = torch.nonzero(scores[:,j]>thresh).view(-1)
                # if there is det
                if inds.numel() > 0:
                    cls_scores = scores[:,j][inds]
                    _, order = torch.sort(cls_scores, 0, True)
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
                    cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                    cls_dets = cls_dets[order]
                    keep = nms(cls_boxes[order, :], cls_scores[order], self.cfg["TEST"]["NMS"])
                    cls_dets = cls_dets[keep.view(-1).long()]
                    all_boxes[j][i] = cls_dets.cpu().numpy()
                else:
                    all_boxes[j][i] = empty_array

            # Limit to max_per_image detections *over all classes*
            if max_per_image > 0:
                image_scores = np.hstack([all_boxes[j][i][:, -1]
                                        for j in range(1, len(self.cfg["datasets"]["voc_classes_list"]))])
                if len(image_scores) > max_per_image:
                    image_thresh = np.sort(image_scores)[-max_per_image]
                    for j in range(1, len(self.cfg["datasets"]["voc_classes_list"])):
                        keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                        all_boxes[j][i] = all_boxes[j][i][keep, :]

            misc_toc = time.time()
            nms_time = misc_toc - misc_tic
            sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
                .format(i + 1, num_images, detect_time, nms_time))
            sys.stdout.flush()

        end = time.time()
        print("test time: %0.4fs" % (end - start))
        
        score=self.evaluate(all_boxes)
        torch.save({"model":self.model.state_dict(),"cfg":self.cfg},self.cfg["saved_path"]["name"]+"/"+self.cfg["model_cfg"]["name"]+"_"+self.cfg["model_cfg"]["backbone"]+"_"+self.cfg["datasets"]["name"]+"_epoch"+str(epoch)+"_score_{:.4f}".format(score)+".pth")


    def val_faster_rcnn(self):
        self.val_fpn()

    def val_fpn_cascade(self):
        val_loader=self.test_dataset.loader
        num_images = len(val_loader)
        self.model.eval()
        thresh=self.cfg["model_cfg"]["thresh"]

        start = time.time()
        max_per_image = 100

        all_boxes = [[[] for _ in range(num_images)]
                  for _ in range( len(self.cfg["datasets"]["voc_classes_list"]))]    

        for i, data in enumerate(val_loader):
            det_tic = time.time()
            if self.cfg["model_cfg"]["use_cuda"]:
                im_data = data["images"].cuda().float()
                annot = data["annot"].cuda()
                im_info=data["image_info"].cuda()
                num_boxes=data["num_boxes"].cuda()
            else:
                im_data = data["images"].float()
                annot = data["annot"]
                im_info=data["image_info"]
                num_boxes=data["num_boxes"]

            with torch.no_grad():
                rois, cls_prob, bbox_pred, \
                rpn_loss_cls, rpn_loss_box, \
                    RCNN_loss_cls, RCNN_loss_bbox, RCNN_loss_cls_2nd, RCNN_loss_bbox_2nd, RCNN_loss_cls_3rd, RCNN_loss_bbox_3rd, \
                    roi_labels = self.model(im_data, im_info, annot, num_boxes)

            scores = cls_prob.data
            boxes = rois.data[:, :, 1:5]

            if cfg["TEST"]["BBOX_REG"]:
                # Apply bounding-box regression deltas
                box_deltas = bbox_pred.data
                if self.cfg["model_cfg"]["use_cuda"]:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(self.cfg["TRAIN"]["BBOX_NORMALIZE_STDS"]).cuda() \
                                + torch.FloatTensor(self.cfg["TRAIN"]["BBOX_NORMALIZE_MEANS"]).cuda()
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(self.cfg["TRAIN"]["BBOX_NORMALIZE_STDS"])\
                                + torch.FloatTensor(self.cfg["TRAIN"]["BBOX_NORMALIZE_MEANS"])

                box_deltas = box_deltas.view(1, -1, 4 * len(self.cfg["datasets"]["voc_classes_list"]))
                pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
            else:
                # Simply repeat the boxes, once for each class
                pred_boxes = np.tile(boxes, (1, scores.shape[1]))

            pred_boxes /= im_info[0][2].item()
            scores = scores.squeeze()
            pred_boxes = pred_boxes.squeeze()
            det_toc = time.time()
            detect_time = det_toc - det_tic
            misc_tic = time.time()

            empty_array = np.transpose(np.array([[],[],[],[],[]]), (1,0))
            for j in range(1, len(self.cfg["datasets"]["voc_classes_list"])):
                inds = torch.nonzero(scores[:,j]>thresh).view(-1)
                # if there is det
                if inds.numel() > 0:
                    cls_scores = scores[:,j][inds]
                    _, order = torch.sort(cls_scores, 0, True)
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
                    cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                    cls_dets = cls_dets[order]
                    keep = nms(cls_boxes[order, :], cls_scores[order], self.cfg["TEST"]["NMS"])
                    cls_dets = cls_dets[keep.view(-1).long()]
                    all_boxes[j][i] = cls_dets.cpu().numpy()
                else:
                    all_boxes[j][i] = empty_array

            # Limit to max_per_image detections *over all classes*
            if max_per_image > 0:
                image_scores = np.hstack([all_boxes[j][i][:, -1]
                                        for j in range(1, len(self.cfg["datasets"]["voc_classes_list"]))])
                if len(image_scores) > max_per_image:
                    image_thresh = np.sort(image_scores)[-max_per_image]
                    for j in range(1, len(self.cfg["datasets"]["voc_classes_list"])):
                        keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                        all_boxes[j][i] = all_boxes[j][i][keep, :]

            misc_toc = time.time()
            nms_time = misc_toc - misc_tic
            sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
                .format(i + 1, num_images, detect_time, nms_time))
            sys.stdout.flush()

        end = time.time()
        print("test time: %0.4fs" % (end - start))
        score=self.evaluate(all_boxes)
        torch.save({"model":self.model.state_dict(),"cfg":self.cfg},self.cfg["saved_path"]["name"]+"/"+self.cfg["model_cfg"]["name"]+"_"+self.cfg["model_cfg"]["backbone"]+"_"+self.cfg["datasets"]["name"]+"_epoch"+str(epoch)+"_score_{:.4f}".format(score)+".pth")


    def val_yolov3(self):
        val_loader=self.test_dataset.loader
        num_images = len(val_loader)
        self.model.eval()
        start = time.time()
        max_per_image = 100

        all_boxes = [[[] for _ in range(num_images)]
                    for _ in range( len(self.cfg["datasets"]["voc_classes_list"]))]    
        empty_array = np.transpose(np.array([[],[],[],[],[]]), (1,0))

        for i, data in enumerate(val_loader):
            misc_tic = time.time()
            if self.cfg["model_cfg"]["use_cuda"]:
                im_data = data["images"].cuda().float()
                annot = data["annot"].cuda()
                im_info=data["image_info"].cuda()
                num_boxes=data["num_boxes"].cuda()
            else:
                im_data = data["images"].float()
                annot = data["annot"]
                im_info=data["image_info"]
                num_boxes=data["num_boxes"]

            outputs = self.model(im_data)
            preds=get_yolo_detections(outputs,self.cfg,im_info)

            scores = preds['scores']
            class_ids = preds['class_ids']
            rois = preds['rois']

            for j in range(0, len(self.cfg["datasets"]["voc_classes_list"])):
                if j not in class_ids:
                    all_boxes[j][i] = empty_array
                    continue
                dets=[]
                for k in range(len(class_ids)):
                    if(class_ids[k]==j):
                        det=np.concatenate((rois[k],[scores[k]]),axis=0)
                        dets.append(det)
                dets=np.array(dets)
                all_boxes[j][i]=dets

            misc_toc = time.time()
            detect_time = misc_toc - misc_tic
            sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s   \r' \
                .format(i + 1, num_images, detect_time))
            sys.stdout.flush()
        end = time.time()
        print("test time: %0.4fs" % (end - start))
        score=self.evaluate(all_boxes)
        torch.save({"model":self.model.state_dict(),"cfg":self.cfg},self.cfg["saved_path"]["name"]+"/"+self.cfg["model_cfg"]["name"]+"_"+self.cfg["model_cfg"]["backbone"]+"_"+self.cfg["datasets"]["name"]+"_epoch"+str(epoch)+"_score_{:.4f}".format(score)+".pth")

    def val_yolov4(self):
        self.val_yolov3()

    def val_yolov4_tiny(self):
        val_loader=self.test_dataset.loader
        num_images = len(val_loader)
        self.model.eval()
        start = time.time()
        max_per_image = 100

        all_boxes = [[[] for _ in range(num_images)]
                    for _ in range( len(self.cfg["datasets"]["voc_classes_list"]))]    
        empty_array = np.transpose(np.array([[],[],[],[],[]]), (1,0))

        for i, data in enumerate(val_loader):
            misc_tic = time.time()
            if self.cfg["model_cfg"]["use_cuda"]:
                im_data = data["images"].cuda().float()
                annot = data["annot"].cuda()
                im_info=data["image_info"].cuda()
                num_boxes=data["num_boxes"].cuda()
            else:
                im_data = data["images"].float()
                annot = data["annot"]
                im_info=data["image_info"]
                num_boxes=data["num_boxes"]
            with torch.no_grad():
                outputs = self.model(im_data)
            preds=get_yolo_tiny_detections(outputs,self.cfg,im_info)

            scores = preds['scores']
            class_ids = preds['class_ids']
            rois = preds['rois']

            for j in range(0, len(self.cfg["datasets"]["voc_classes_list"])):
                if j not in class_ids:
                    all_boxes[j][i] = empty_array
                    continue
                dets=[]
                for k in range(len(class_ids)):
                    if(class_ids[k]==j):
                        det=np.concatenate((rois[k],[scores[k]]),axis=0)
                        dets.append(det)
                dets=np.array(dets)
                all_boxes[j][i]=dets

            misc_toc = time.time()
            detect_time = misc_toc - misc_tic
            sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s   \r' \
                .format(i + 1, num_images, detect_time))
            sys.stdout.flush()
        end = time.time()
        print("test time: %0.4fs" % (end - start))
        score=self.evaluate(all_boxes)
        torch.save({"model":self.model.state_dict(),"cfg":self.cfg},self.cfg["saved_path"]["name"]+"/"+self.cfg["model_cfg"]["name"]+"_"+self.cfg["model_cfg"]["backbone"]+"_"+self.cfg["datasets"]["name"]+"_epoch"+str(epoch)+"_score_{:.4f}".format(score)+".pth")


    def val_efficientdet(self):
        val_loader=self.test_dataset.loader
        num_images = len(val_loader)
        self.model.eval()
        threshold=self.cfg["model_cfg"]["thresh"]
        nms_threshold=self.cfg["model_cfg"]["nms_thresh"]
        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()

        for i, data in enumerate(val_loader):
            misc_tic = time.time()
            if self.cfg["model_cfg"]["use_cuda"]:
                im_data = data["images"].cuda().float()
                annot = data["annot"].cuda()
                im_info=data["image_info"].cuda()
                num_boxes=data["num_boxes"].cuda()
            else:
                im_data = data["images"].float()
                annot = data["annot"]
                im_info=data["image_info"]
                num_boxes=data["num_boxes"]

            with torch.no_grad():
                features, regression, classification, anchors = self.model(im_data)
            preds = postprocess(im_data,
                                anchors, regression, classification,
                                regressBoxes, clipBoxes,
                                threshold, nms_threshold)[0]
                        
            # no objects fill with empty_array
            if not preds:
                for j in range(0, len(cfg["datasets"]["voc_classes_list"])):
                    all_boxes[j][i] = empty_array
                continue

            scores = preds['scores']
            class_ids = preds['class_ids']
            rois = preds['rois']
            rois/= im_info[0][2].item()

            for j in range(0, len(cfg["datasets"]["voc_classes_list"])):
                if j not in class_ids:
                    all_boxes[j][i] = empty_array
                    continue
                dets=[]
                for k in range(len(class_ids)):
                    if(class_ids[k]==j):
                        det=np.concatenate((rois[k],[scores[k]]),axis=0)
                        dets.append(det)
                dets=np.array(dets)
                all_boxes[j][i]=dets

            misc_toc = time.time()
            detect_time = misc_toc - misc_tic
            sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s   \r' \
                .format(i + 1, num_images, detect_time))
            sys.stdout.flush()

        score=self.evaluate(all_boxes)
        torch.save({"model":self.model.state_dict(),"cfg":self.cfg},self.cfg["saved_path"]["name"]+"/"+self.cfg["model_cfg"]["name"]+"_"+self.cfg["model_cfg"]["backbone"]+"_"+self.cfg["datasets"]["name"]+"_epoch"+str(epoch)+"_score_{:.4f}".format(score)+".pth")

    #========================================================Different test once
    def test_fpn(self,data):
        thresh=self.cfg["model_cfg"]["thresh"]
        mns_thresh=self.cfg["model_cfg"]["nms_thresh"]
        if self.cfg["model_cfg"]["use_cuda"]:
            im_data = data["images"].cuda().float()
            annot = data["annot"].cuda()
            im_info=data["image_info"].cuda()
            num_boxes=data["num_boxes"].cuda()
        else:
            im_data = data["images"].float()
            annot = data["annot"]
            im_info=data["image_info"]
            num_boxes=data["num_boxes"]
        with torch.no_grad():
            rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_box, RCNN_loss_cls, RCNN_loss_bbox, rois_label = self.model(im_data, im_info, annot, num_boxes)

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        box_deltas = bbox_pred.data
        if self.cfg["model_cfg"]["use_cuda"]:
            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg["TRAIN"]["BBOX_NORMALIZE_STDS"]).cuda() \
                        + torch.FloatTensor(cfg["TRAIN"]["BBOX_NORMALIZE_MEANS"]).cuda()
        else:
            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg["TRAIN"]["BBOX_NORMALIZE_STDS"])\
                        + torch.FloatTensor(cfg["TRAIN"]["BBOX_NORMALIZE_MEANS"])
        box_deltas = box_deltas.view(1, -1, 4 * len(cfg["datasets"]["voc_classes_list"]))
        pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
        pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)


        pred_boxes /= im_info[0][2].item()
        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()

        self.pred_boxes=[]
        for j in range(1, len(cfg["datasets"]["voc_classes_list"])):
            inds = torch.nonzero(scores[:,j]>thresh).view(-1)
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:,j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                cls_dets = cls_dets[order]
                keep = nms(cls_boxes[order, :], cls_scores[order],mns_thresh)
                cls_dets = cls_dets[keep.view(-1).long()]
                dets_np=cls_dets.cpu().numpy()
                # print(dets_np.shape)
                for det_id in range(dets_np.shape[0]):
                  xmin,ymin,xmax,ymax,score=dets_np[det_id]
                  self.pred_boxes.append([cfg["datasets"]["voc_classes_list"][j],xmin,ymin,xmax,ymax,score])

        return self.pred_boxes

    def test_faster_rcnn(self,data):
        self.test_fpn(data)

    def test_fpn_cascade(self,data):
        if self.cfg["model_cfg"]["use_cuda"]:
            im_data = data["images"].cuda().float()
            annot = data["annot"].cuda()
            im_info=data["image_info"].cuda()
            num_boxes=data["num_boxes"].cuda()
        else:
            im_data = data["images"].float()
            annot = data["annot"]
            im_info=data["image_info"]
            num_boxes=data["num_boxes"]

        with torch.no_grad():
            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
                RCNN_loss_cls, RCNN_loss_bbox, RCNN_loss_cls_2nd, RCNN_loss_bbox_2nd, RCNN_loss_cls_3rd, RCNN_loss_bbox_3rd, \
                roi_labels = self.model(im_data, im_info, annot, num_boxes)

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        if cfg["TEST"]["BBOX_REG"]:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if self.cfg["model_cfg"]["use_cuda"]:
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg["TRAIN"]["BBOX_NORMALIZE_STDS"]).cuda() \
                            + torch.FloatTensor(cfg["TRAIN"]["BBOX_NORMALIZE_MEANS"]).cuda()
            else:
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg["TRAIN"]["BBOX_NORMALIZE_STDS"])\
                            + torch.FloatTensor(cfg["TRAIN"]["BBOX_NORMALIZE_MEANS"])
            box_deltas = box_deltas.view(1, -1, 4 * len(cfg["datasets"]["voc_classes_list"]))
            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        pred_boxes /= im_info[0][2].item()
        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()

        self.pred_boxes=[]
        for j in range(1, len(cfg["datasets"]["voc_classes_list"])):
            inds = torch.nonzero(scores[:,j]>thresh).view(-1)
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:,j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                cls_dets = cls_dets[order]
                keep = nms(cls_boxes[order, :], cls_scores[order],mns_thresh)
                cls_dets = cls_dets[keep.view(-1).long()]
                dets_np=cls_dets.cpu().numpy()
                # print(dets_np.shape)
                for det_id in range(dets_np.shape[0]):
                  xmin,ymin,xmax,ymax,score=dets_np[det_id]
                  self.pred_boxes.append([cfg["datasets"]["voc_classes_list"][j],xmin,ymin,xmax,ymax,score])
        return self.pred_boxes


    def test_yolov3(self,data):
        if self.cfg["model_cfg"]["use_cuda"]:
            im_data = data["images"].cuda().float()
            annot = data["annot"].cuda()
            im_info=data["image_info"].cuda()
            num_boxes=data["num_boxes"].cuda()
        else:
            im_data = data["images"].float()
            annot = data["annot"]
            im_info=data["image_info"]
            num_boxes=data["num_boxes"]

        with torch.no_grad():
            outputs = self.model(im_data)
        preds=get_yolo_detections(outputs,self.cfg,im_info)
        scores = preds['scores']
        class_ids = preds['class_ids']
        rois = preds['rois']
        self.pred_boxes=[]
        for j in range(0, len(cfg["datasets"]["voc_classes_list"])):
            if j not in class_ids:
                continue
            for k in range(len(class_ids)):
                if(class_ids[k]==j):
                    dets=np.concatenate((rois[k],[scores[k]]),axis=0)
                    xmin,ymin,xmax,ymax,score=dets
                    self.pred_boxes.append([cfg["datasets"]["voc_classes_list"][j],int(xmin),int(ymin),int(xmax),int(ymax),float(score)])
                    # for det in dets:
                    #     xmin,ymin,xmax,ymax,score=det
                    #     self.pred_boxes.append([cfg["datasets"]["voc_classes_list"][j],xmin,ymin,xmax,ymax,score])
        return self.pred_boxes

    def test_yolov4(self,data):
        self.test_yolov3(data)

    def test_yolov4_tiny(self,data):
        if self.cfg["model_cfg"]["use_cuda"]:
            im_data = data["images"].cuda().float()
            annot = data["annot"].cuda()
            im_info=data["image_info"].cuda()
            num_boxes=data["num_boxes"].cuda()
        else:
            im_data = data["images"].float()
            annot = data["annot"]
            im_info=data["image_info"]
            num_boxes=data["num_boxes"]

        with torch.no_grad():
            outputs = self.model(im_data)

        preds=get_yolo_tiny_detections(outputs,self.cfg,im_info)
        scores = preds['scores']
        class_ids = preds['class_ids']
        rois = preds['rois']
        # print(rois)
        self.pred_boxes=[]
        for j in range(0, len(cfg["datasets"]["voc_classes_list"])):
            if j not in class_ids:
                continue
            for k in range(len(class_ids)):
                if(class_ids[k]==j):
                    dets=np.concatenate((rois[k],[scores[k]]),axis=0)
                    xmin,ymin,xmax,ymax,score=dets
                    self.pred_boxes.append([cfg["datasets"]["voc_classes_list"][j],int(xmin),int(ymin),int(xmax),int(ymax),float(score)])
                    # print(dets)
                    # for det in dets:
                    #     xmin,ymin,xmax,ymax,score=det
                    #     self.pred_boxes.append([cfg["datasets"]["voc_classes_list"][j],xmin,ymin,xmax,ymax,score])

        return self.pred_boxes

    def test_efficientdet(self,data):
        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()
        threshold=self.cfg["model_cfg"]["thresh"]
        nms_threshold=self.cfg["model_cfg"]["nms_thresh"]

        if self.cfg["model_cfg"]["use_cuda"]:
            im_data = data["images"].cuda().float()
            annot = data["annot"].cuda()
            im_info=data["image_info"].cuda()
            num_boxes=data["num_boxes"].cuda()
        else:
            im_data = data["images"].float()
            annot = data["annot"]
            im_info=data["image_info"]
            num_boxes=data["num_boxes"]

        with torch.no_grad():
            features, regression, classification, anchors = self.model(im_data)
        preds = postprocess(im_data,
                            anchors, regression, classification,
                            regressBoxes, clipBoxes,
                            threshold, nms_threshold)[0]

        scores = preds['scores']
        class_ids = preds['class_ids']
        rois = preds['rois']
        rois/= im_info[0][2].item()

        self.pred_boxes=[]
        for j in range(0, len(cfg["datasets"]["voc_classes_list"])):
            if j not in class_ids:
                continue
            for k in range(len(class_ids)):
                if(class_ids[k]==j):
                    dets=np.concatenate((rois[k],[scores[k]]),axis=0)
                    xmin,ymin,xmax,ymax,score=dets
                    self.pred_boxes.append([cfg["datasets"]["voc_classes_list"][j],int(xmin),int(ymin),int(xmax),int(ymax),float(score)])
                    # for det in dets:
                    #     xmin,ymin,xmax,ymax,score=det
                    #     self.pred_boxes.append([cfg["datasets"]["voc_classes_list"][j],xmin,ymin,xmax,ymax,score])
        return self.pred_boxes

    def draw(self,imOrigin,pred_boxes):
        color_list=[]
        for r in range(5):
            for g in range(5):
                for b in range(5):
                    color_list.append((120+20*r,120+20*g,120+20*b))
        for det in pred_boxes:
            class_name,xmin,ymin,xmax,ymax,score=det
            cv2.rectangle(imOrigin, (xmin,ymin), (xmax,ymax), color_list[cfg["datasets"]["voc_classes_list"].index(class_name)], 2)
            cv2.putText(imOrigin, '%s %.3f' % (class_name,score), (xmin,ymin), cv2.FONT_HERSHEY_PLAIN,
                          2.0,  (0, 255,0), thickness=2)

        return imOrigin