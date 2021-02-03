# Author: Zylo117

import os

import cv2
import numpy as np
import torch
from glob import glob
from torch import nn
from torchvision.ops import nms
from torchvision.ops.boxes import batched_nms
from typing import Union
import uuid
import xml.etree.ElementTree as ET
import pickle
from os import listdir, getcwd
from os.path import join
import xml.dom.minidom

class xmlObj(object):
    def __init__(self,name,width,height):
        self.name=name

        self.doc =xml.dom.minidom.Document()
        self.root = self.doc.createElement('annotation')
        self.doc.appendChild(self.root)
        xml_folder = self.doc.createElement('folder')
        xml_folder.appendChild(self.doc.createTextNode(str('Annotations')))
        xml_filename = self.doc.createElement('filename')
        xml_filename.appendChild(self.doc.createTextNode(name))
        xml_path=self.doc.createElement('path')
        xml_path.appendChild(self.doc.createTextNode("images/"+name))
        self.root.appendChild(xml_folder)
        self.root.appendChild(xml_filename)
        self.root.appendChild(xml_path)

        xml_source=self.doc.createElement('source')
        xml_database=self.doc.createElement('database')
        xml_database.appendChild(self.doc.createTextNode("unknow"))
        xml_source.appendChild(xml_database)
        self.root.appendChild(xml_source)

        xml_size=self.doc.createElement('size')
        xml_width=self.doc.createElement('width')
        xml_width.appendChild(self.doc.createTextNode(str(width)))
        xml_height=self.doc.createElement('height')
        xml_height.appendChild(self.doc.createTextNode(str(height)))
        xml_depth=self.doc.createElement('depth')
        xml_depth.appendChild(self.doc.createTextNode(str(3)))
        xml_size.appendChild(xml_width)
        xml_size.appendChild(xml_height)
        xml_size.appendChild(xml_depth)

        xml_segmented=self.doc.createElement('segmente')
        xml_segmented.appendChild(self.doc.createTextNode(str(0)))
        self.root.appendChild(xml_size)
        self.root.appendChild(xml_segmented)
    
    def add_anno(self,cls_name,x,y,w,h):
        xml_obj=self.doc.createElement('object')
        xml_name=self.doc.createElement('name')
        xml_name.appendChild(self.doc.createTextNode(cls_name))
        xml_pos=self.doc.createElement('pos')
        xml_pos.appendChild(self.doc.createTextNode("Unspecified"))
        xml_truncated=self.doc.createElement('truncated')
        xml_truncated.appendChild(self.doc.createTextNode("0"))
        xml_difficult=self.doc.createElement('difficult')
        xml_difficult.appendChild(self.doc.createTextNode("0"))

        xml_bnbbox=self.doc.createElement('bndbox')
        xml_xmin = self.doc.createElement('xmin')
        xml_xmin.appendChild(self.doc.createTextNode(str(x)))
        xml_ymin = self.doc.createElement('ymin')
        xml_ymin.appendChild(self.doc.createTextNode(str(y)))
        xml_xmax = self.doc.createElement('xmax')
        xml_xmax.appendChild(self.doc.createTextNode(str(x+w)))
        xml_ymax = self.doc.createElement('ymax')
        xml_ymax.appendChild(self.doc.createTextNode(str(y+h)))
        xml_bnbbox.appendChild(xml_xmin)
        xml_bnbbox.appendChild(xml_ymin)
        xml_bnbbox.appendChild(xml_xmax)
        xml_bnbbox.appendChild(xml_ymax)

        xml_obj.appendChild(xml_name)
        xml_obj.appendChild(xml_pos)
        xml_obj.appendChild(xml_truncated)
        xml_obj.appendChild(xml_difficult)
        xml_obj.appendChild(xml_bnbbox)
        self.root.appendChild(xml_obj)

    def save(self,folder):
        fp = open(folder+"/"+self.name.split('.')[0]+'.xml', 'w')
        self.doc.writexml(fp, indent='\t', addindent='\t', newl='\n', encoding="utf-8")



def invert_affine(metas: Union[float, list, tuple], preds):
    for i in range(len(preds)):
        if len(preds[i]['rois']) == 0:
            continue
        else:
            if metas is float:
                preds[i]['rois'][:, [0, 2]] = preds[i]['rois'][:, [0, 2]] / metas
                preds[i]['rois'][:, [1, 3]] = preds[i]['rois'][:, [1, 3]] / metas
            else:
                new_w, new_h, old_w, old_h, padding_w, padding_h = metas[i]
                preds[i]['rois'][:, [0, 2]] = preds[i]['rois'][:, [0, 2]] / (new_w / old_w)
                preds[i]['rois'][:, [1, 3]] = preds[i]['rois'][:, [1, 3]] / (new_h / old_h)
    return preds


def aspectaware_resize_padding(image, width, height, interpolation=None, means=None):
    old_h, old_w, c = image.shape
    if old_w > old_h:
        new_w = width
        new_h = int(width / old_w * old_h)
    else:
        new_w = int(height / old_h * old_w)
        new_h = height

    canvas = np.zeros((height, height, c), np.float32)
    if means is not None:
        canvas[...] = means

    if new_w != old_w or new_h != old_h:
        if interpolation is None:
            image = cv2.resize(image, (new_w, new_h))
        else:
            image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    padding_h = height - new_h
    padding_w = width - new_w

    if c > 1:
        canvas[:new_h, :new_w] = image
    else:
        if len(image.shape) == 2:
            canvas[:new_h, :new_w, 0] = image
        else:
            canvas[:new_h, :new_w] = image

    return canvas, new_w, new_h, old_w, old_h, padding_w, padding_h,


def preprocess(*image_path, max_size=512, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)):
    ori_imgs = [cv2.imread(img_path) for img_path in image_path]
    normalized_imgs = [(img / 255 - mean) / std for img in ori_imgs]
    imgs_meta = [aspectaware_resize_padding(img[..., ::-1], max_size, max_size,
                                            means=None) for img in normalized_imgs]
    framed_imgs = [img_meta[0] for img_meta in imgs_meta]
    framed_metas = [img_meta[1:] for img_meta in imgs_meta]

    return ori_imgs, framed_imgs, framed_metas


def preprocess_video(*frame_from_video, max_size=512, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)):
    ori_imgs = frame_from_video
    normalized_imgs = [(img / 255 - mean) / std for img in ori_imgs]
    imgs_meta = [aspectaware_resize_padding(img[..., ::-1], max_size, max_size,
                                            means=None) for img in normalized_imgs]
    framed_imgs = [img_meta[0] for img_meta in imgs_meta]
    framed_metas = [img_meta[1:] for img_meta in imgs_meta]

    return ori_imgs, framed_imgs, framed_metas


def postprocess(x, anchors, regression, classification, regressBoxes, clipBoxes, threshold, iou_threshold):
    transformed_anchors = regressBoxes(anchors, regression)
    transformed_anchors = clipBoxes(transformed_anchors, x)
    scores = torch.max(classification, dim=2, keepdim=True)[0]
    scores_over_thresh = (scores > threshold)[:, :, 0]
    out = []
    for i in range(x.shape[0]):
        if scores_over_thresh[i].sum() == 0:
            out.append({
                'rois': np.array(()),
                'class_ids': np.array(()),
                'scores': np.array(()),
            })
            continue

        classification_per = classification[i, scores_over_thresh[i, :], ...].permute(1, 0)
        transformed_anchors_per = transformed_anchors[i, scores_over_thresh[i, :], ...]
        scores_per = scores[i, scores_over_thresh[i, :], ...]
        scores_, classes_ = classification_per.max(dim=0)
        anchors_nms_idx = batched_nms(transformed_anchors_per, scores_per[:, 0], classes_, iou_threshold=iou_threshold)

        if anchors_nms_idx.shape[0] != 0:
            classes_ = classes_[anchors_nms_idx]
            scores_ = scores_[anchors_nms_idx]
            boxes_ = transformed_anchors_per[anchors_nms_idx, :]

            out.append({
                'rois': boxes_.detach().cpu().numpy(),
                'class_ids': classes_.detach().cpu().numpy(),
                'scores': scores_.detach().cpu().numpy(),
            })
        else:
            out.append({
                'rois': np.array(()),
                'class_ids': np.array(()),
                'scores': np.array(()),
            })

    return out


def display(preds, imgs, obj_list, imshow=True, imwrite=False):
    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            continue

        for j in range(len(preds[i]['rois'])):
            (x1, y1, x2, y2) = preds[i]['rois'][j].astype(np.int)
            cv2.rectangle(imgs[i], (x1, y1), (x2, y2), (255, 255, 0), 2)
            obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])

            cv2.putText(imgs[i], '{}, {:.3f}'.format(obj, score),
                        (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 0), 1)
        if imshow:
            cv2.imshow('img', imgs[i])
            cv2.waitKey(1)

        if imwrite:
            os.makedirs('test/', exist_ok=True)
            cv2.imwrite(f'test/{uuid.uuid4().hex}.jpg', imgs[i])


# def from_colorname_to_bgr(color):
#     rgb_color=webcolors.name_to_rgb(color)
#     result=(rgb_color.blue,rgb_color.green,rgb_color.red)
#     return result

def standard_to_bgr(list_color_name):
    standard= []
    for i in range(len(list_color_name)-36): #-36 used to match the len(obj_list)
        standard.append(from_colorname_to_bgr(list_color_name[i]))
    return standard

def get_index_label(label, obj_list):
    index = int(obj_list.index(label))
    return index

def plot_one_box(img, coord, label=None, score=None, color=None, line_thickness=None):
    tl = line_thickness or int(round(0.001 * max(img.shape[0:2])))  # line thickness
    color = color
    c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 2, 1)  # font thickness
        s_size = cv2.getTextSize(str('{:.0%}'.format(score)),0, fontScale=float(tl) / 3, thickness=tf)[0]
        t_size = cv2.getTextSize(label, 0, fontScale=float(tl) / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0]+s_size[0]+15, c1[1] - t_size[1] -3
        cv2.rectangle(img, c1, c2 , color, -1)  # filled
        cv2.putText(img, '{}: {:.0%}'.format(label, score), (c1[0],c1[1] - 2), 0, float(tl) / 3, [0, 0, 0], thickness=tf, lineType=cv2.FONT_HERSHEY_SIMPLEX)
