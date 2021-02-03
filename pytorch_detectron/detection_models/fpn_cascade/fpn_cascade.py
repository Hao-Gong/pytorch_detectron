from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import os
# self defined packages
from ...backbone_models.resnet.resnet import  resnetSelection
from ...base_functions.rpn.rpn_fpn import _RPN_FPN
from ...lib_c.model.roi_layers import ROIAlign, ROIPool,ROIAlignAvg
from ...base_functions.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
from ...base_functions.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta
from ...base_functions.rpn.bbox_transform import bbox_transform_inv, clip_boxes, bbox_decode

class resnet_fpn_cascade(nn.Module):
    def __init__(self, cfg):
        super(resnet_fpn_cascade, self).__init__()
        backbone_model=cfg["model_cfg"]["backbone"]
        self.cfg=cfg
        self.pretrained = cfg["model_cfg"]["pretrained"]
        self.classes =cfg["datasets"]["voc_classes_list"]
        abspath=os.path.abspath(os.path.join(os.path.dirname(__file__),os.path.pardir,os.path.pardir,os.path.pardir,"pretrain_models"))
        self.model_path =abspath+'/'+backbone_model+'.pth'
        self.model_name=backbone_model
        self.dout_base_model = 256
        self.n_classes = len(self.classes)
        self.class_agnostic = False
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        self.maxpool2d = nn.MaxPool2d(1, stride=2)
        # define rpn
        self.RCNN_rpn = _RPN_FPN(self.dout_base_model,self.cfg)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes,self.cfg)

        # NOTE: the original paper used pool_size = 7 for cls branch, and 14 for mask branch, to save the
        # computation time, we first use 14 as the pool_size, and then do stride=2 pooling for cls branch.
        self.RCNN_roi_pool = ROIPool((self.cfg.POOLING_SIZE, self.cfg.POOLING_SIZE), 1.0/16.0)
        self.RCNN_roi_align = ROIAlignAvg((self.cfg.POOLING_SIZE+1, self.cfg.POOLING_SIZE+1), 1.0/16.0, 0)
        # self.RCNN_roi_pool = ROIPool(self.cfg.POOLING_SIZE, self.cfg.POOLING_SIZE, 1.0/16.0)
        # self.RCNN_roi_align = ROIAlignAvg(self.cfg.POOLING_SIZE, self.cfg.POOLING_SIZE, 1.0/16.0)
        self.grid_size = self.cfg.POOLING_SIZE * 2 if self.cfg.CROP_RESIZE_WITH_MAX_POOL else self.cfg.POOLING_SIZE
            
        resnet = resnetSelection(self.model_name)

        if self.pretrained == True:
            print("Loading pretrained weights from %s" %(self.model_path))
            state_dict = torch.load(self.model_path)
            resnet.load_state_dict({k:v for k,v in state_dict.items() if k in resnet.state_dict()})

        self.RCNN_layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.RCNN_layer1 = nn.Sequential(resnet.layer1)
        self.RCNN_layer2 = nn.Sequential(resnet.layer2)
        self.RCNN_layer3 = nn.Sequential(resnet.layer3)
        self.RCNN_layer4 = nn.Sequential(resnet.layer4)

        # Top layer
        if "34" in self.model_name or "18" in self.model_name:
            # Top layer
            self.RCNN_toplayer = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)  # reduce channel
            # Lateral layers
            self.RCNN_latlayer1 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
            self.RCNN_latlayer2 = nn.Conv2d( 128, 256, kernel_size=1, stride=1, padding=0)
            self.RCNN_latlayer3 = nn.Conv2d( 64, 256, kernel_size=1, stride=1, padding=0)
        else:
            # Top layer
            self.RCNN_toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # reduce channel
            # Lateral layers
            self.RCNN_latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
            self.RCNN_latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
            self.RCNN_latlayer3 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)
        # Smooth layers
        self.RCNN_smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.RCNN_smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.RCNN_smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # ROI Pool feature downsampling
        self.RCNN_roi_feat_ds = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)

        self.RCNN_top = nn.Sequential(
            nn.Conv2d(256, 1024, kernel_size=self.cfg.POOLING_SIZE, stride=self.cfg.POOLING_SIZE, padding=0),
            nn.ReLU(True),
            nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True)
        )

        self.RCNN_top_2nd = nn.Sequential(
            nn.Conv2d(256, 1024, kernel_size=self.cfg.POOLING_SIZE, stride=self.cfg.POOLING_SIZE, padding=0),
            nn.ReLU(True),
            nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True)
        )

        self.RCNN_top_3rd = nn.Sequential(
            nn.Conv2d(256, 1024, kernel_size=self.cfg.POOLING_SIZE, stride=self.cfg.POOLING_SIZE, padding=0),
            nn.ReLU(True),
            nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True)
        )

        self.RCNN_cls_score = nn.Linear(1024, self.n_classes)
        if self.class_agnostic:
            self.RCNN_bbox_pred = nn.Linear(1024, 4)
        else:
            self.RCNN_bbox_pred = nn.Linear(1024, 4 * self.n_classes)

        self.RCNN_cls_score_2nd = nn.Linear(1024, self.n_classes)
        if self.class_agnostic:
            self.RCNN_bbox_pred_2nd = nn.Linear(1024, 4)
        else:
            self.RCNN_bbox_pred_2nd = nn.Linear(1024, 4 * self.n_classes)

        self.RCNN_cls_score_3rd = nn.Linear(1024, self.n_classes)
        if self.class_agnostic:
            self.RCNN_bbox_pred_3rd = nn.Linear(1024, 4)
        else:
            self.RCNN_bbox_pred_3rd = nn.Linear(1024, 4 * self.n_classes)

        # Fix blocks
        for p in self.RCNN_layer0[0].parameters(): p.requires_grad=False
        for p in self.RCNN_layer0[1].parameters(): p.requires_grad=False

        assert (0 <= self.cfg.RESNET.FIXED_BLOCKS < 4)
        if self.cfg.RESNET.FIXED_BLOCKS >= 3:
            for p in self.RCNN_layer3.parameters(): p.requires_grad=False
        if self.cfg.RESNET.FIXED_BLOCKS >= 2:
            for p in self.RCNN_layer2.parameters(): p.requires_grad=False
        if self.cfg.RESNET.FIXED_BLOCKS >= 1:
            for p in self.RCNN_layer1.parameters(): p.requires_grad=False

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters(): p.requires_grad=False

        self.RCNN_layer0.apply(set_bn_fix)
        self.RCNN_layer1.apply(set_bn_fix)
        self.RCNN_layer2.apply(set_bn_fix)
        self.RCNN_layer3.apply(set_bn_fix)
        self.RCNN_layer4.apply(set_bn_fix)
        self.create_architecture()

    def train(self, mode=True):
      # Override train so that the training mode is set as we want
      nn.Module.train(self, mode)
      if mode:
        # Set fixed blocks to be in eval mode
        self.RCNN_layer0.eval()
        self.RCNN_layer1.eval()
        self.RCNN_layer2.train()
        self.RCNN_layer3.train()
        self.RCNN_layer4.train()

        self.RCNN_smooth1.train()
        self.RCNN_smooth2.train()
        self.RCNN_smooth3.train()

        self.RCNN_latlayer1.train()
        self.RCNN_latlayer2.train()
        self.RCNN_latlayer3.train()

        self.RCNN_toplayer.train()

        def set_bn_eval(m):
          classname = m.__class__.__name__
          if classname.find('BatchNorm') != -1:
            m.eval()

        self.RCNN_layer0.apply(set_bn_eval)
        self.RCNN_layer1.apply(set_bn_eval)
        self.RCNN_layer2.apply(set_bn_eval)
        self.RCNN_layer3.apply(set_bn_eval)
        self.RCNN_layer4.apply(set_bn_eval)

    def _head_to_tail(self, pool5):
        block5 = self.RCNN_top(pool5)
        fc7 = block5.mean(3).mean(2)
        return fc7

    def _head_to_tail_2nd(self, pool5):
        block5 = self.RCNN_top_2nd(pool5)
        fc7 = block5.mean(3).mean(2)
        return fc7

    def _head_to_tail_3rd(self, pool5):
        block5 = self.RCNN_top_3rd(pool5)
        fc7 = block5.mean(3).mean(2)
        return fc7

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        # custom weights initialization called on netG and netD
        def weights_init(m, mean, stddev, truncated=False):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

        normal_init(self.RCNN_toplayer, 0, 0.01, self.cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_smooth1, 0, 0.01, self.cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_smooth2, 0, 0.01, self.cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_smooth3, 0, 0.01, self.cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_latlayer1, 0, 0.01, self.cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_latlayer2, 0, 0.01, self.cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_latlayer3, 0, 0.01, self.cfg.TRAIN.TRUNCATED)

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, self.cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, self.cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, self.cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, self.cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, self.cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score_2nd, 0, 0.01, self.cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred_2nd, 0, 0.001, self.cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score_3rd, 0, 0.01, self.cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred_3rd, 0, 0.001, self.cfg.TRAIN.TRUNCATED)
        weights_init(self.RCNN_top, 0, 0.01, self.cfg.TRAIN.TRUNCATED)
        weights_init(self.RCNN_top_2nd, 0, 0.01, self.cfg.TRAIN.TRUNCATED)
        weights_init(self.RCNN_top_3rd, 0, 0.01, self.cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_weights()

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
        x: (Variable) top feature map to be upsampled.
        y: (Variable) lateral feature map.
        Returns:
        (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear',align_corners=True) + y

    def _PyramidRoI_Feat(self, feat_maps, rois, im_info):
        ''' roi pool on pyramid feature maps'''
        # do roi pooling based on predicted rois
        # print("rois shape",rois.shape)
        # print("feat_maps",feat_maps.shape)
        img_area = im_info[0][0] * im_info[0][1]
        h = rois.data[:, 4] - rois.data[:, 2] + 1
        w = rois.data[:, 3] - rois.data[:, 1] + 1
        # print(h)
        # print(w)
        
        roi_level = torch.log2(torch.sqrt(h * w) / 224.0)
        roi_level = torch.round(roi_level + 4)
        roi_level[roi_level < 2] = 2
        roi_level[roi_level > 5] = 5
        # roi_level.fill_(5)
        # print("roi_level",roi_level)
        if self.cfg.POOLING_MODE == 'align':
            roi_pool_feats = []
            box_to_levels = []
            for i, l in enumerate(range(2, 6)):
                # print(i, l)
                # print(roi_level)
                if (roi_level == l).sum() == 0:
                    continue
                
                idx_l = (roi_level == l).nonzero().squeeze()
                # print(idx_l.dim())
                # print((idx_l.cpu().numpy()))
                if(idx_l.dim()==0):
                    idx_l=idx_l.unsqueeze(0)
                    # continue
                    # print("^^^^^^^^^^^^^^^^^^^^^^",idx_l.dim())
                box_to_levels.append(idx_l)
                scale = feat_maps[i].size(2) / im_info[0][0]
                # self.RCNN_roi_align.scale=scale
                feat = self.RCNN_roi_align(feat_maps[i], rois[idx_l],scale)
                roi_pool_feats.append(feat)

            # print("box_to_levels")
            # print(box_to_levels)
            roi_pool_feat = torch.cat(roi_pool_feats, 0)
            box_to_level = torch.cat(box_to_levels, 0)
            idx_sorted, order = torch.sort(box_to_level)
            roi_pool_feat = roi_pool_feat[order]

        elif self.cfg.POOLING_MODE == 'pool':
            roi_pool_feats = []
            box_to_levels = []
            for i, l in enumerate(range(2, 6)):
                if (roi_level == l).sum() == 0:
                    continue
                idx_l = (roi_level == l).nonzero().squeeze()
                box_to_levels.append(idx_l)
                scale = feat_maps[i].size(2) / im_info[0][0]
                self.RCNN_roi_pool.scale=scale
                feat = self.RCNN_roi_pool(feat_maps[i], rois[idx_l])
                roi_pool_feats.append(feat)
            roi_pool_feat = torch.cat(roi_pool_feats, 0)
            box_to_level = torch.cat(box_to_levels, 0)
            idx_sorted, order = torch.sort(box_to_level)
            roi_pool_feat = roi_pool_feat[order]
            
        return roi_pool_feat

    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        # Bottom-up
        c1 = self.RCNN_layer0(im_data)
        c2 = self.RCNN_layer1(c1)
        c3 = self.RCNN_layer2(c2)
        c4 = self.RCNN_layer3(c3)
        c5 = self.RCNN_layer4(c4)
        # Top-down
        p5 = self.RCNN_toplayer(c5)
        p4 = self._upsample_add(p5, self.RCNN_latlayer1(c4))
        p4 = self.RCNN_smooth1(p4)
        p3 = self._upsample_add(p4, self.RCNN_latlayer2(c3))
        p3 = self.RCNN_smooth2(p3)
        p2 = self._upsample_add(p3, self.RCNN_latlayer3(c2))
        p2 = self.RCNN_smooth3(p2)

        p6 = self.maxpool2d(p5)

        rpn_feature_maps = [p2, p3, p4, p5, p6]
        mrcnn_feature_maps = [p2, p3, p4, p5]

        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(rpn_feature_maps, im_info, gt_boxes, num_boxes)
        # print("rois shape stage1:",rois.shape)
        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, gt_assign, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            ## NOTE: additionally, normalize proposals to range [0, 1],
            #        this is necessary so that the following roi pooling
            #        is correct on different feature maps
            # rois[:, :, 1::2] /= im_info[0][1]
            # rois[:, :, 2::2] /= im_info[0][0]

            rois = rois.view(-1, 5)
            rois_label = rois_label.view(-1).long()
            gt_assign = gt_assign.view(-1).long()
            pos_id = rois_label.nonzero().squeeze()
            gt_assign_pos = gt_assign[pos_id]
            rois_label_pos = rois_label[pos_id]
            rois_label_pos_ids = pos_id

            rois_pos = Variable(rois[pos_id])
            rois = Variable(rois)
            rois_label = Variable(rois_label)

            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            ## NOTE: additionally, normalize proposals to range [0, 1],
            #        this is necessary so that the following roi pooling
            #        is correct on different feature maps
            # rois[:, :, 1::2] /= im_info[0][1]
            # rois[:, :, 2::2] /= im_info[0][0]

            rois_label = None
            gt_assign = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0
            rois = rois.view(-1, 5)
            pos_id = torch.arange(0, rois.size(0)).long().type_as(rois).long()
            rois_label_pos_ids = pos_id
            rois_pos = Variable(rois[pos_id])
            rois = Variable(rois)

        roi_pool_feat = self._PyramidRoI_Feat(mrcnn_feature_maps, rois, im_info)

        # feed pooled features to top model
        pooled_feat = self._head_to_tail(roi_pool_feat)

        bbox_pred = self.RCNN_bbox_pred(pooled_feat)

        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.long().view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score,1)
        # print(cls_prob)
        # print("*******************cls prob shape",cls_prob.shape)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)
            # loss (l1-norm) for bounding box regression
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

        rois = rois.view(batch_size, -1, rois.size(1))
        bbox_pred = bbox_pred.view(batch_size, -1, bbox_pred.size(1))
        
        if self.training:
            rois_label = rois_label.view(batch_size, -1)

        # 2nd-----------------------------
        # decode
        rois = bbox_decode(rois, bbox_pred, batch_size, self.class_agnostic, self.n_classes, im_info, self.training,cls_prob,self.cfg)

        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes, stage=2)
            rois, rois_label, gt_assign, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois = rois.view(-1, 5)
            rois_label = rois_label.view(-1).long()
            gt_assign = gt_assign.view(-1).long()
            pos_id = rois_label.nonzero().squeeze()
            gt_assign_pos = gt_assign[pos_id]
            rois_label_pos = rois_label[pos_id]
            rois_label_pos_ids = pos_id

            rois_pos = Variable(rois[pos_id])
            rois = Variable(rois)
            rois_label = Variable(rois_label)

            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            gt_assign = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0
            rois = rois.view(-1, 5)
            pos_id = torch.arange(0, rois.size(0)).long().type_as(rois).long()
            # print(pos_id)
            rois_label_pos_ids = pos_id
            rois_pos = Variable(rois[pos_id])
            rois = Variable(rois)

        roi_pool_feat = self._PyramidRoI_Feat(mrcnn_feature_maps, rois, im_info)
        # feed pooled features to top model
        pooled_feat = self._head_to_tail_2nd(roi_pool_feat)
        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred_2nd(pooled_feat)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1,
                                            rois_label.long().view(rois_label.size(0), 1, 1).expand(rois_label.size(0),
                                                                                                    1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score_2nd(pooled_feat)
        cls_prob_2nd = F.softmax(cls_score,1) 

        RCNN_loss_cls_2nd = 0
        RCNN_loss_bbox_2nd = 0

        if self.training:
            # loss (cross entropy) for object classification
            RCNN_loss_cls_2nd = F.cross_entropy(cls_score, rois_label)
            # loss (l1-norm) for bounding box regression
            RCNN_loss_bbox_2nd = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

        rois = rois.view(batch_size, -1, rois.size(1))
        # cls_prob_2nd = cls_prob_2nd.view(batch_size, -1, cls_prob_2nd.size(1))  ----------------not be used ---------
        bbox_pred_2nd = bbox_pred.view(batch_size, -1, bbox_pred.size(1))

        if self.training:
            rois_label = rois_label.view(batch_size, -1)

        # 3rd---------------
        # decode
        rois = bbox_decode(rois, bbox_pred_2nd, batch_size, self.class_agnostic, self.n_classes, im_info, self.training,cls_prob_2nd,self.cfg)

        # proposal_target
        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes, stage=3)
            rois, rois_label, gt_assign, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois = rois.view(-1, 5)
            rois_label = rois_label.view(-1).long()
            gt_assign = gt_assign.view(-1).long()
            pos_id = rois_label.nonzero().squeeze()
            gt_assign_pos = gt_assign[pos_id]
            rois_label_pos = rois_label[pos_id]
            rois_label_pos_ids = pos_id

            rois_pos = Variable(rois[pos_id])
            rois = Variable(rois)
            rois_label = Variable(rois_label)

            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:

            rois_label = None
            gt_assign = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0
            rois = rois.view(-1, 5)
            pos_id = torch.arange(0, rois.size(0)).long().type_as(rois).long()
            rois_label_pos_ids = pos_id
            rois_pos = Variable(rois[pos_id])
            rois = Variable(rois)

        roi_pool_feat = self._PyramidRoI_Feat(mrcnn_feature_maps, rois, im_info)

        # feed pooled features to top model
        pooled_feat = self._head_to_tail_3rd(roi_pool_feat)

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred_3rd(pooled_feat)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1,
                                            rois_label.long().view(rois_label.size(0), 1, 1).expand(
                                                rois_label.size(0),
                                                1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score_3rd(pooled_feat)
        cls_prob_3rd = F.softmax(cls_score, 1)

        RCNN_loss_cls_3rd = 0
        RCNN_loss_bbox_3rd = 0

        if self.training:
            # loss (cross entropy) for object classification
            RCNN_loss_cls_3rd = F.cross_entropy(cls_score, rois_label)
            # loss (l1-norm) for bounding box regression
            RCNN_loss_bbox_3rd = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

        rois = rois.view(batch_size, -1, rois.size(1))
        cls_prob_3rd = cls_prob_3rd.view(batch_size, -1, cls_prob_3rd.size(1))
        bbox_pred_3rd = bbox_pred.view(batch_size, -1, bbox_pred.size(1))

        if self.training:
            rois_label = rois_label.view(batch_size, -1)
        if not self.training:
            # 3rd_avg
            # 1st_3rd
            pooled_feat_1st_3rd = self._head_to_tail(roi_pool_feat)
            cls_score_1st_3rd = self.RCNN_cls_score(pooled_feat_1st_3rd)
            cls_prob_1st_3rd = F.softmax(cls_score_1st_3rd, 1)
            cls_prob_1st_3rd = cls_prob_1st_3rd.view(batch_size, -1, cls_prob_1st_3rd.size(1))
            # 2nd_3rd
            pooled_feat_2nd_3rd = self._head_to_tail_2nd(roi_pool_feat)
            cls_score_2nd_3rd = self.RCNN_cls_score_2nd(pooled_feat_2nd_3rd)
            cls_prob_2nd_3rd = F.softmax(cls_score_2nd_3rd, 1)
            cls_prob_2nd_3rd = cls_prob_2nd_3rd.view(batch_size, -1, cls_prob_2nd_3rd.size(1))

            cls_prob_3rd_avg = (cls_prob_1st_3rd + cls_prob_2nd_3rd + cls_prob_3rd) / 3
        else:
            cls_prob_3rd_avg = cls_prob_3rd

        return rois, cls_prob_3rd_avg, bbox_pred_3rd, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, RCNN_loss_cls_2nd, RCNN_loss_bbox_2nd, RCNN_loss_cls_3rd, RCNN_loss_bbox_3rd, rois_label