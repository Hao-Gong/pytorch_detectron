  
from random import shuffle
import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb


from ...detection_models.yolo.yolo_utils import bbox_iou, merge_bboxes,jaccard,bboxes_iou

#---------------------------------------------------#
#   平滑标签
#---------------------------------------------------#
def smooth_labels(y_true, label_smoothing,num_classes):
    return y_true * (1.0 - label_smoothing) + label_smoothing / num_classes

def box_ciou(b1, b2):
    """
    输入为：
    ----------
    b1: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
    b2: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh

    返回为：
    -------
    ciou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
    """
    # 求出预测框左上角右下角
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half
    # 求出真实框左上角右下角
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    # 求真实框和预测框所有的iou
    intersect_mins = torch.max(b1_mins, b2_mins)
    intersect_maxes = torch.min(b1_maxes, b2_maxes)
    intersect_wh = torch.max(intersect_maxes - intersect_mins, torch.zeros_like(intersect_maxes))
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    union_area = b1_area + b2_area - intersect_area
    iou = intersect_area / torch.clamp(union_area,min = 1e-6)

    # 计算中心的差距
    center_distance = torch.sum(torch.pow((b1_xy - b2_xy), 2), axis=-1)
    
    # 找到包裹两个框的最小框的左上角和右下角
    enclose_mins = torch.min(b1_mins, b2_mins)
    enclose_maxes = torch.max(b1_maxes, b2_maxes)
    enclose_wh = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(intersect_maxes))
    # 计算对角线距离
    enclose_diagonal = torch.sum(torch.pow(enclose_wh,2), axis=-1)
    ciou = iou - 1.0 * (center_distance) / torch.clamp(enclose_diagonal,min = 1e-6)
    
    v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(b1_wh[..., 0]/torch.clamp(b1_wh[..., 1],min = 1e-6)) - torch.atan(b2_wh[..., 0]/torch.clamp(b2_wh[..., 1],min = 1e-6))), 2)
    alpha = v / torch.clamp((1.0 - iou + v),min=1e-6)
    ciou = ciou - alpha * v
    return ciou
  
def clip_by_tensor(t,t_min,t_max):
    t=t.float()
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result

def MSELoss(pred,target):
    return (pred-target)**2

def BCELoss(pred,target):
    epsilon = 1e-7
    pred = clip_by_tensor(pred, epsilon, 1.0 - epsilon)
    output = -target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
    return output

class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes, img_size, label_smooth=0, cuda=True):
        super(YOLOLoss, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.img_size = img_size
        self.feature_length = [img_size[0]//32,img_size[0]//16,img_size[0]//8]
        self.label_smooth = label_smooth

        self.ignore_threshold = 0.5
        self.lambda_conf = 1.0
        self.lambda_cls = 1.0
        self.lambda_loc = 1.0
        self.iou_loss_mode="l2"
        self.l2_loss = nn.MSELoss(reduction='sum')
        self.bceloss = nn.BCELoss(reduction='sum') 
        self.iou_judgement_mode="jaccard" # "iou" "jaccard"
        self.cuda = cuda

    def forward(self, input, targets=None):
        # input为bs,3*(5+num_classes),13,13
        
        # 一共多少张图片
        bs = input.size(0)
        # 特征层的高
        in_h = input.size(2)
        # 特征层的宽
        in_w = input.size(3)

        # 计算步长
        # 每一个特征点对应原来的图片上多少个像素点
        # 如果特征层为13x13的话，一个特征点就对应原来的图片上的32个像素点
        stride_h = self.img_size[1] / in_h
        stride_w = self.img_size[0] / in_w

        # 把先验框的尺寸调整成特征层大小的形式
        # 计算出先验框在特征层上对应的宽高
        scaled_anchors = [(a_w / self.img_size[0], a_h / self.img_size[1] ) for a_w, a_h in self.anchors]
        # print(scaled_anchors)
        # bs,3*(5+num_classes),13,13 -> bs,3,13,13,(5+num_classes)
        prediction = input.view(bs, int(self.num_anchors/3),
                                self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()
        
        # 对prediction预测进行调整
        conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        # 找到哪些先验框内部包含物体
        obj_mask,coodn_mask,pred_boxes, t_box, tconf, tcls, box_loss_scale_x, box_loss_scale_y = self.build_target(prediction, targets, scaled_anchors,in_w, in_h,self.ignore_threshold)

        if self.cuda:
            obj_mask= obj_mask.cuda()
            coodn_mask=coodn_mask.cuda()
            box_loss_scale_x, box_loss_scale_y= box_loss_scale_x.cuda(), box_loss_scale_y.cuda()
            tconf, tcls = tconf.cuda(), tcls.cuda()
            pred_boxes = pred_boxes.cuda()
            t_box = t_box.cuda()

        box_loss_scale = 2-box_loss_scale_x*box_loss_scale_y

        if self.iou_loss_mode=="l2":
            # print("hello")
            pred_xy=torch.sigmoid(prediction[..., :2])
            # print(box_loss_scale.shape,pred_xy[...,0].shape)
            iou_loss_x=self.l2_loss(pred_xy[coodn_mask==1][...,0]*box_loss_scale[coodn_mask==1], t_box[coodn_mask==1][...,0]*box_loss_scale[coodn_mask==1])
            iou_loss_y=self.l2_loss(pred_xy[coodn_mask==1][...,1]*box_loss_scale[coodn_mask==1], t_box[coodn_mask==1][...,1]*box_loss_scale[coodn_mask==1])
            iou_loss_w=self.l2_loss(prediction[coodn_mask==1][...,2]*box_loss_scale[coodn_mask==1], t_box[coodn_mask==1][...,2]*box_loss_scale[coodn_mask==1])
            iou_loss_h=self.l2_loss(prediction[coodn_mask==1][...,3]*box_loss_scale[coodn_mask==1], t_box[coodn_mask==1][...,3]*box_loss_scale[coodn_mask==1])
            iou_loss=iou_loss_x+iou_loss_y+iou_loss_w+iou_loss_h

        elif self.iou_loss_mode=="ciou":
            iou_loss = (1 - box_ciou( pred_boxes[coodn_mask.bool()], t_box[coodn_mask.bool()]))* box_loss_scale[coodn_mask.bool()]

        loss_loc = torch.sum(iou_loss)
        loss_conf = torch.sum(self.bceloss(conf[obj_mask.bool()], tconf[obj_mask.bool()]))
        loss_cls = torch.sum(self.bceloss(pred_cls[coodn_mask.bool()], tcls[coodn_mask.bool()]))
#         print(loss_loc.item(),loss_conf.item(),loss_cls.item())
        loss = loss_conf * self.lambda_conf + loss_cls * self.lambda_cls + loss_loc * self.lambda_loc
        return loss, loss_conf.item(), loss_cls.item(), loss_loc.item()

    def build_target(self, prediction, target, scaled_anchors_total, in_w, in_h, ignore_threshold):
        # 计算一共有多少张图片
        bs = len(target)
        # 获得先验框
        anchor_index = [[0,1,2],[3,4,5],[6,7,8]][self.feature_length.index(in_w)]

        scaled_anchors = np.array(scaled_anchors_total)[anchor_index]
        # layer_anchor_index=[]
        # for index in anchor_index:
        #     layer_anchor_index.append(index-anchor_index[0])

        # 创建全是0或者全是1的阵列
        obj_mask = torch.ones(bs, int(self.num_anchors/3), in_h, in_w, requires_grad=False)
        coodn_mask=torch.zeros(bs, int(self.num_anchors/3), in_h, in_w,requires_grad=False)

        tx = torch.zeros(bs, int(self.num_anchors/3), in_h, in_w, requires_grad=False)
        ty = torch.zeros(bs, int(self.num_anchors/3), in_h, in_w, requires_grad=False)
        tw = torch.zeros(bs, int(self.num_anchors/3), in_h, in_w, requires_grad=False)
        th = torch.zeros(bs, int(self.num_anchors/3), in_h, in_w, requires_grad=False)
        t_box = torch.zeros(bs, int(self.num_anchors/3), in_h, in_w, 4, requires_grad=False)
        tconf = torch.zeros(bs, int(self.num_anchors/3), in_h, in_w, requires_grad=False)
        tcls = torch.zeros(bs, int(self.num_anchors/3), in_h, in_w, self.num_classes, requires_grad=False)

        ############################################ 第一阶段：ignore bbox update select ############################################
        # 先验框的中心位置的调整参数
        x = torch.sigmoid(prediction[..., 0])  
        y = torch.sigmoid(prediction[..., 1])
        # 先验框的宽高调整参数
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        # 生成网格，先验框中心，网格左上角
        grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_w, 1).repeat(
            int(bs*self.num_anchors/3), 1, 1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_h, 1).t().repeat(
            int(bs*self.num_anchors/3), 1, 1).view(y.shape).type(FloatTensor)

        # 生成先验框的宽高
        anchor_w = FloatTensor(scaled_anchors).index_select(dim=1, index=LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors).index_select(dim=1, index=LongTensor([1]))
        
        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)
        # print( x,grid_x)
        # 计算调整后的先验框中心与宽高
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = (x + grid_x)/in_w
        pred_boxes[..., 1] = (y + grid_y)/in_h
        pred_boxes[..., 2] = torch.exp(w) * anchor_w
        pred_boxes[..., 3] = torch.exp(h) * anchor_h

        # print(pred_boxes[..., 0])

        for i in range(bs):
            pred_boxes_bs = pred_boxes[i].view(-1, 4)
            target_bs=target[i][target[i][:,4]!=-1]

            if len(target_bs) > 0:
                gx = target_bs[:, 0:1]
                gy =target_bs[:, 1:2]
                gw = target_bs[:, 2:3]
                gh = target_bs[:, 3:4] 
                
                gt_box = torch.FloatTensor(np.concatenate([gx, gy, gw, gh],-1)).type(FloatTensor)
                # print(gt_box)
                if self.iou_judgement_mode=="iou":
                    anch_ious = bboxes_iou(gt_box, pred_boxes_bs,x1y1x2y2=False)
                elif self.iou_judgement_mode=="jaccard":
                    anch_ious = jaccard(gt_box, pred_boxes_bs)
                # print(gt_box, pred_boxes_bs)
                # print(anch_ious[anch_ious>0])
                for t in range(target_bs.shape[0]):
                    anch_iou = anch_ious[t].view(pred_boxes[i].size()[:3])
                    obj_mask[i][anch_iou>self.ignore_threshold] = 0

        ############################################ 第二阶段：not ignorebbox update select ############################################
        box_loss_scale_x = torch.zeros(bs, int(self.num_anchors/3), in_h, in_w, requires_grad=False)
        box_loss_scale_y = torch.zeros(bs, int(self.num_anchors/3), in_h, in_w, requires_grad=False)
        for b in range(bs):
            target_bs=target[b][target[b][:,4]!=-1]
            # print(target_bs)
            for t in range(target_bs.shape[0]):
                # print("target",t)
                # 计算出在特征层上的点位
                gx = target_bs[t, 0] * in_w
                gy = target_bs[t, 1] * in_h
                
                gw = target_bs[t, 2] * in_w
                gh = target_bs[t, 3] * in_h
                # 计算出属于哪个网格
                gi = int(gx)
                gj = int(gy)

                # print(gx,gy,gi,gj)

                # 计算真实框的位置
                gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)
                
                # 计算出所有先验框的位置
                anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((self.num_anchors, 2)),
                                                                  np.array(scaled_anchors_total)), 1))
                # print(anchor_shapes)
                anchor_shapes[...,2]*=  in_w
                anchor_shapes[...,3]*=  in_h
                # print(anchor_shapes)
                # 计算重合程度
                anch_ious = bbox_iou(gt_box, anchor_shapes)
                
                # print(gi,gj)
                # print(anch_ious[anch_ious>0])
                # Find the best matching anchor box
                best_n = np.argmax(anch_ious)
                # print(best_n,anchor_index)
                if best_n not in anchor_index:
                    continue
                # Masks
                # print(anch_ious,best_n)
                current_best_n=best_n-anchor_index[0]
                if (gj < in_h) and (gi < in_w) and (gj >=0) and (gi >=0):
                    # 判定哪些先验框内部真实的存在物体
                    obj_mask[b, current_best_n, gj, gi] = 1
                    coodn_mask[b, current_best_n, gj, gi] = 1
                    # # 计算先验框中心调整参数
                    if self.iou_loss_mode=="ciou":
                        tx[b, current_best_n, gj, gi] = gx
                        ty[b, current_best_n, gj, gi] = gy
                        # 计算先验框宽高调整参数
                        tw[b, current_best_n, gj, gi] = gw
                        th[b, current_best_n, gj, gi] = gh
                    elif self.iou_loss_mode=="l2":
                        # tx[b, best_n, gj, gi] = gx-gx.to(torch.int16).to(torch.float)
                        # ty[b, best_n, gj, gi] = gy-gy.to(torch.int16).to(torch.float)
                        tx[b, current_best_n, gj, gi] = gx-gi
                        ty[b, current_best_n, gj, gi] = gy-gj
                        # 计算先验框宽高调整参数
                        tw[b, current_best_n, gj, gi] = torch.log(gw/anchor_shapes[best_n,2]+ 1e-16)
                        th[b, current_best_n, gj, gi] = torch.log(gh/anchor_shapes[best_n,3]+ 1e-16)
                
                    # 用于获得xywh的比例
                    box_loss_scale_x[b, current_best_n, gj, gi] = target_bs[t, 2]
                    box_loss_scale_y[b, current_best_n, gj, gi] = target_bs[t, 3]
                    # 物体置信度
                    tconf[b, current_best_n, gj, gi] = 1
                    # 种类
                    tcls[b, current_best_n, gj, gi, int(target_bs[t, 4])] = 1
        if self.iou_loss_mode=="ciou":
            pred_boxes[..., 0] = pred_boxes[..., 0]*in_w
            pred_boxes[..., 1] = pred_boxes[..., 1]*in_h
            pred_boxes[..., 2] = pred_boxes[..., 2]*in_w
            pred_boxes[..., 3] = pred_boxes[..., 3]*in_w
        t_box[...,0] = tx
        t_box[...,1] = ty
        t_box[...,2] = tw
        t_box[...,3] = th
        return obj_mask,coodn_mask,pred_boxes, t_box, tconf, tcls, box_loss_scale_x, box_loss_scale_y

class YOLOTinyLoss(nn.Module):
    def __init__(self, anchors, num_classes, img_size, label_smooth=0, cuda=True):
        super(YOLOTinyLoss, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.img_size = img_size
        self.feature_length = [img_size[0]//32,img_size[0]//16]
        self.label_smooth = label_smooth

        self.ignore_threshold = 0.5
        self.lambda_conf = 1.0
        self.lambda_cls = 1.0
        self.lambda_loc = 1.0
        self.iou_loss_mode="l2"
        self.l2_loss = nn.MSELoss(reduction='sum')
        self.bceloss = nn.BCELoss(reduction='sum') 
        self.iou_judgement_mode="jaccard" # "iou" "jaccard"
        self.cuda = cuda

    def forward(self, input, targets=None):
        # input为bs,3*(5+num_classes),13,13
        
        # 一共多少张图片
        bs = input.size(0)
        # 特征层的高
        in_h = input.size(2)
        # 特征层的宽
        in_w = input.size(3)

        # 计算步长
        # 每一个特征点对应原来的图片上多少个像素点
        # 如果特征层为13x13的话，一个特征点就对应原来的图片上的32个像素点
        stride_h = self.img_size[1] / in_h
        stride_w = self.img_size[0] / in_w

        # 把先验框的尺寸调整成特征层大小的形式
        # 计算出先验框在特征层上对应的宽高
        scaled_anchors = [(a_w / self.img_size[0], a_h / self.img_size[1] ) for a_w, a_h in self.anchors]
        # bs,3*(5+num_classes),13,13 -> bs,3,13,13,(5+num_classes)
        prediction = input.view(bs, int(self.num_anchors/2),
                                self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()
        
        # 对prediction预测进行调整
        conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        # 找到哪些先验框内部包含物体
        obj_mask,coodn_mask,pred_boxes, t_box, tconf, tcls, box_loss_scale_x, box_loss_scale_y = self.build_target(prediction, targets, scaled_anchors,in_w, in_h,self.ignore_threshold)

        if self.cuda:
            obj_mask= obj_mask.cuda()
            coodn_mask=coodn_mask.cuda()
            box_loss_scale_x, box_loss_scale_y= box_loss_scale_x.cuda(), box_loss_scale_y.cuda()
            tconf, tcls = tconf.cuda(), tcls.cuda()
            pred_boxes = pred_boxes.cuda()
            t_box = t_box.cuda()

        box_loss_scale = 2-box_loss_scale_x*box_loss_scale_y
        if self.iou_loss_mode=="l2":
            pred_xy=torch.sigmoid(prediction[..., :2])
            iou_loss_x=self.l2_loss(pred_xy[coodn_mask==1][...,0]*box_loss_scale[coodn_mask==1], t_box[coodn_mask==1][...,0]*box_loss_scale[coodn_mask==1])
            iou_loss_y=self.l2_loss(pred_xy[coodn_mask==1][...,1]*box_loss_scale[coodn_mask==1], t_box[coodn_mask==1][...,1]*box_loss_scale[coodn_mask==1])
            iou_loss_w=self.l2_loss(prediction[coodn_mask==1][...,2]*box_loss_scale[coodn_mask==1], t_box[coodn_mask==1][...,2]*box_loss_scale[coodn_mask==1])
            iou_loss_h=self.l2_loss(prediction[coodn_mask==1][...,3]*box_loss_scale[coodn_mask==1], t_box[coodn_mask==1][...,3]*box_loss_scale[coodn_mask==1])
            iou_loss=iou_loss_x+iou_loss_y+iou_loss_w+iou_loss_h

        elif self.iou_loss_mode=="ciou":
            iou_loss = (1 - box_ciou( pred_boxes[coodn_mask.bool()], t_box[coodn_mask.bool()]))* box_loss_scale[coodn_mask.bool()]

        loss_loc = torch.sum(iou_loss)
        loss_conf = torch.sum(self.bceloss(conf[obj_mask.bool()], tconf[obj_mask.bool()]))
        loss_cls = torch.sum(self.bceloss(pred_cls[coodn_mask.bool()], tcls[coodn_mask.bool()]))
        loss = loss_conf * self.lambda_conf + loss_cls * self.lambda_cls + loss_loc * self.lambda_loc
        # print(loss_loc.item(),loss_conf.item(),loss_cls.item())
        return loss, loss_conf.item(), loss_cls.item(), loss_loc.item()

    def build_target(self, prediction, target, scaled_anchors_total, in_w, in_h, ignore_threshold):
        # 计算一共有多少张图片
        bs = len(target)
        # 获得先验框
        anchor_index = [[0,1,2],[3,4,5]][self.feature_length.index(in_w)]
        scaled_anchors = np.array(scaled_anchors_total)[anchor_index]

        # 创建全是0或者全是1的阵列
        obj_mask = torch.ones(bs, int(self.num_anchors/2), in_h, in_w, requires_grad=False)
        coodn_mask=torch.zeros(bs, int(self.num_anchors/2), in_h, in_w,requires_grad=False)

        tx = torch.zeros(bs, int(self.num_anchors/2), in_h, in_w, requires_grad=False)
        ty = torch.zeros(bs, int(self.num_anchors/2), in_h, in_w, requires_grad=False)
        tw = torch.zeros(bs, int(self.num_anchors/2), in_h, in_w, requires_grad=False)
        th = torch.zeros(bs, int(self.num_anchors/2), in_h, in_w, requires_grad=False)
        t_box = torch.zeros(bs, int(self.num_anchors/2), in_h, in_w, 4, requires_grad=False)
        tconf = torch.zeros(bs, int(self.num_anchors/2), in_h, in_w, requires_grad=False)
        tcls = torch.zeros(bs, int(self.num_anchors/2), in_h, in_w, self.num_classes, requires_grad=False)

        ############################################ 第一阶段：ignore bbox update select ############################################
        # 先验框的中心位置的调整参数
        x = torch.sigmoid(prediction[..., 0])  
        y = torch.sigmoid(prediction[..., 1])
        # 先验框的宽高调整参数
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        # 生成网格，先验框中心，网格左上角
        grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_w, 1).repeat(
            int(bs*self.num_anchors/2), 1, 1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_h, 1).t().repeat(
            int(bs*self.num_anchors/2), 1, 1).view(y.shape).type(FloatTensor)

        # 生成先验框的宽高
        anchor_w = FloatTensor(scaled_anchors).index_select(dim=1, index=LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors).index_select(dim=1, index=LongTensor([1]))
        
        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)
        # print( x,grid_x)
        # 计算调整后的先验框中心与宽高
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = (x + grid_x)/in_w
        pred_boxes[..., 1] = (y + grid_y)/in_h
        pred_boxes[..., 2] = torch.exp(w) * anchor_w
        pred_boxes[..., 3] = torch.exp(h) * anchor_h

        for i in range(bs):
            pred_boxes_bs = pred_boxes[i].view(-1, 4)
            target_bs=target[i][target[i][:,4]!=-1]

            if len(target_bs) > 0:
                gx = target_bs[:, 0:1]
                gy =target_bs[:, 1:2]
                gw = target_bs[:, 2:3]
                gh = target_bs[:, 3:4] 
                
                gt_box = torch.FloatTensor(np.concatenate([gx, gy, gw, gh],-1)).type(FloatTensor)
                # print(gt_box)
                if self.iou_judgement_mode=="iou":
                    anch_ious = bboxes_iou(gt_box, pred_boxes_bs,x1y1x2y2=False)
                elif self.iou_judgement_mode=="jaccard":
                    anch_ious = jaccard(gt_box, pred_boxes_bs)
                # print(gt_box, pred_boxes_bs)
                # print(anch_ious[anch_ious>0])
                for t in range(target_bs.shape[0]):
                    anch_iou = anch_ious[t].view(pred_boxes[i].size()[:3])
                    obj_mask[i][anch_iou>self.ignore_threshold] = 0

        ############################################ 第二阶段：not ignorebbox update select ############################################
        box_loss_scale_x = torch.zeros(bs, int(self.num_anchors/2), in_h, in_w, requires_grad=False)
        box_loss_scale_y = torch.zeros(bs, int(self.num_anchors/2), in_h, in_w, requires_grad=False)
        for b in range(bs):
            target_bs=target[b][target[b][:,4]!=-1]

            for t in range(target_bs.shape[0]):
                # print(target[b])
                # 计算出在特征层上的点位
                gx = target_bs[t, 0] * in_w
                gy = target_bs[t, 1] * in_h
                
                gw = target_bs[t, 2] * in_w
                gh = target_bs[t, 3] * in_h

                # 计算出属于哪个网格
                gi = int(gx)
                gj = int(gy)

                # 计算真实框的位置
                gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)
                
                # 计算出所有先验框的位置
                anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((self.num_anchors, 2)),
                                                                  np.array(scaled_anchors_total)), 1))
                # print(anchor_shapes)
                anchor_shapes[...,2]*=  in_w
                anchor_shapes[...,3]*=  in_h
                # print(anchor_shapes)
                # 计算重合程度
                anch_ious = bbox_iou(gt_box, anchor_shapes)
                # print(anch_ious[anch_ious>0])
                # Find the best matching anchor box
                best_n = np.argmax(anch_ious)
                if best_n not in anchor_index:
                    continue
                current_best_n=best_n-anchor_index[0]
                # Masks
                if (gj < in_h) and (gi < in_w) and (gj >=0) and (gi >=0):
                    # 判定哪些先验框内部真实的存在物体
                    obj_mask[b, current_best_n, gj, gi] = 1
                    coodn_mask[b, current_best_n, gj, gi] = 1
                    # # 计算先验框中心调整参数
                    if self.iou_loss_mode=="ciou":
                        tx[b, current_best_n, gj, gi] = gx
                        ty[b, current_best_n, gj, gi] = gy
                        # 计算先验框宽高调整参数
                        tw[b, current_best_n, gj, gi] = gw
                        th[b, current_best_n, gj, gi] = gh
                    elif self.iou_loss_mode=="l2":
                        # tx[b, best_n, gj, gi] = gx-gx.to(torch.int16).to(torch.float)
                        # ty[b, best_n, gj, gi] = gy-gy.to(torch.int16).to(torch.float)
                        tx[b, current_best_n, gj, gi] = gx-gi
                        ty[b, current_best_n, gj, gi] = gy-gj
                        # 计算先验框宽高调整参数
                        tw[b, current_best_n, gj, gi] = torch.log(gw/anchor_shapes[best_n,2]+ 1e-16)
                        th[b, current_best_n, gj, gi] = torch.log(gh/anchor_shapes[best_n,3]+ 1e-16)
                
                    # 用于获得xywh的比例
                    box_loss_scale_x[b, current_best_n, gj, gi] = target_bs[t, 2]
                    box_loss_scale_y[b, current_best_n, gj, gi] = target_bs[t, 3]
                    # 物体置信度
                    tconf[b, current_best_n, gj, gi] = 1
                    # 种类
                    tcls[b, current_best_n, gj, gi, int(target_bs[t, 4])] = 1
        if self.iou_loss_mode=="ciou":
            pred_boxes[..., 0] = pred_boxes[..., 0]*in_w
            pred_boxes[..., 1] = pred_boxes[..., 1]*in_h
            pred_boxes[..., 2] = pred_boxes[..., 2]*in_w
            pred_boxes[..., 3] = pred_boxes[..., 3]*in_w

        t_box[...,0] = tx
        t_box[...,1] = ty
        t_box[...,2] = tw
        t_box[...,3] = th
        return obj_mask,coodn_mask,pred_boxes, t_box, tconf, tcls, box_loss_scale_x, box_loss_scale_y

