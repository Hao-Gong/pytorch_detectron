from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
import pdb
import time


# from ...base_functions.utils.config import self.cfg
from ...base_functions.rpn.proposal_layer_fpn import _ProposalLayer_FPN
from ...base_functions.rpn.anchor_target_layer_fpn import _AnchorTargetLayer_FPN
from ...base_functions.utils.net_utils import _smooth_l1_loss

class _RPN_FPN(nn.Module):
    """ region proposal network """
    def __init__(self, din,cfg):
        super(_RPN_FPN, self).__init__()
        self.cfg=cfg
        self.din = din  # get depth of input feature map, e.g., 512
        self.anchor_ratios = self.cfg.ANCHOR_RATIOS
        self.anchor_scales = self.cfg.ANCHOR_SCALES
        self.feat_stride = self.cfg.FEAT_STRIDE
        # print(self.cfg.FEAT_STRIDE[0])

        # define the convrelu layers processing input feature map
        self.RPN_Conv = nn.Conv2d(self.din, 512, 3, 1, 1, bias=True)

        # define bg/fg classifcation score layer
        # self.nc_score_out = len(self.anchor_scales) * len(self.anchor_ratios) * 2 # 2(bg/fg) * 9 (anchors)
        self.nc_score_out = 1 * len(self.anchor_ratios) * 2 # 2(bg/fg) * 3 (anchor ratios) * 1 (anchor scale)
        self.RPN_cls_score = nn.Conv2d(512, self.nc_score_out, 1, 1, 0)

        # define anchor box offset prediction layer
        # self.nc_bbox_out = len(self.anchor_scales) * len(self.anchor_ratios) * 4 # 4(coords) * 9 (anchors)
        self.nc_bbox_out = 1 * len(self.anchor_ratios) * 4 # 4(coords) * 3 (anchors) * 1 (anchor scale)
        self.RPN_bbox_pred = nn.Conv2d(512, self.nc_bbox_out, 1, 1, 0)

        # define proposal layer
        self.RPN_proposal = _ProposalLayer_FPN(self.feat_stride, self.anchor_scales, self.anchor_ratios,self.cfg)

        # define anchor target layer
        self.RPN_anchor_target = _AnchorTargetLayer_FPN(self.feat_stride, self.anchor_scales, self.anchor_ratios,self.cfg)

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        x = x.reshape(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        return x

    def forward(self, rpn_feature_maps, im_info, gt_boxes, num_boxes):        

        # print(self.cfg.FEAT_STRIDE[0])
        # print(self.cfg.FEAT_STRIDE)
        n_feat_maps = len(rpn_feature_maps)

        rpn_cls_scores = []
        rpn_cls_probs = []
        rpn_bbox_preds = []
        rpn_shapes = []

        for i in range(n_feat_maps):
            feat_map = rpn_feature_maps[i]
            batch_size = feat_map.size(0)
            
            # return feature map after convrelu layer
            rpn_conv1 = F.relu(self.RPN_Conv(feat_map), inplace=True)
            # get rpn classification score
            rpn_cls_score = self.RPN_cls_score(rpn_conv1)

            rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)
            rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, 1)
            rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.nc_score_out)

            # get rpn offsets to the anchor boxes
            rpn_bbox_pred = self.RPN_bbox_pred(rpn_conv1)

            rpn_shapes.append([rpn_cls_score.size()[2], rpn_cls_score.size()[3]])
            rpn_cls_scores.append(rpn_cls_score.permute(0, 2, 3, 1).contiguous().reshape(batch_size, -1, 2))
            rpn_cls_probs.append(rpn_cls_prob.permute(0, 2, 3, 1).contiguous().reshape(batch_size, -1, 2))
            rpn_bbox_preds.append(rpn_bbox_pred.permute(0, 2, 3, 1).contiguous().reshape(batch_size, -1, 4))

        rpn_cls_score_alls = torch.cat(rpn_cls_scores, 1)
        rpn_cls_prob_alls = torch.cat(rpn_cls_probs, 1)
        rpn_bbox_pred_alls = torch.cat(rpn_bbox_preds, 1)

        n_rpn_pred = rpn_cls_score_alls.size(1)

        # proposal layer
        self.cfg_key = 'TRAIN' if self.training else 'TEST'

        rois = self.RPN_proposal((rpn_cls_prob_alls.data, rpn_bbox_pred_alls.data,
                                 im_info, self.cfg_key, rpn_shapes))

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

        # generating training labels and build the rpn loss
        if self.training:
            assert gt_boxes is not None

            rpn_data = self.RPN_anchor_target((rpn_cls_score_alls.data, gt_boxes, im_info, num_boxes, rpn_shapes))

            # compute classification loss
            rpn_label = rpn_data[0].reshape(batch_size, -1)
            rpn_keep = Variable(rpn_label.reshape(-1).ne(-1).nonzero().reshape(-1))
            rpn_cls_score = torch.index_select(rpn_cls_score_alls.reshape(-1,2), 0, rpn_keep)
            rpn_label = torch.index_select(rpn_label.reshape(-1), 0, rpn_keep.data)
            rpn_label = Variable(rpn_label.long())
            self.rpn_loss_cls = F.cross_entropy(rpn_cls_score, rpn_label)
            fg_cnt = torch.sum(rpn_label.data.ne(0))

            rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]
            # print(rpn_bbox_targets.shape)
            # compute bbox regression loss
            rpn_bbox_inside_weights = Variable(rpn_bbox_inside_weights.unsqueeze(2) \
                    .expand(batch_size, rpn_bbox_inside_weights.size(1), 4))
            rpn_bbox_outside_weights = Variable(rpn_bbox_outside_weights.unsqueeze(2) \
                    .expand(batch_size, rpn_bbox_outside_weights.size(1), 4))
            rpn_bbox_targets = Variable(rpn_bbox_targets)
            
            self.rpn_loss_box = _smooth_l1_loss(rpn_bbox_pred_alls, rpn_bbox_targets, rpn_bbox_inside_weights, 
                            rpn_bbox_outside_weights, sigma=3)

        return rois, self.rpn_loss_cls, self.rpn_loss_box
