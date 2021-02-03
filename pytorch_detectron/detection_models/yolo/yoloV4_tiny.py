import torch
import torch.nn as nn
from collections import OrderedDict
from ...backbone_models.darknet.CSPdarknet_tiny import CSPDarkNet_tiny
import os

#-------------------------------------------------#
#   卷积块
#   CONV+BATCHNORM+LeakyReLU
#-------------------------------------------------#
class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


#---------------------------------------------------#
#   卷积 + 上采样
#---------------------------------------------------#
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x,):
        x = self.upsample(x)
        return x


#---------------------------------------------------#
#   最后获得yolov4的输出
#---------------------------------------------------#
def yolo_head(filters_list, in_filters):
    m = nn.Sequential(
        BasicConv(in_filters, filters_list[0], 3),
        nn.Conv2d(filters_list[0], filters_list[1], 1),
    )
    return m

#---------------------------------------------------#
#   yolo_body
#---------------------------------------------------#
class YoloV4TinyBody(nn.Module):
    def __init__(self, cfg):
        super(YoloV4TinyBody, self).__init__()
        self.cfg=cfg
        self.pretrain = cfg["model_cfg"]["pretrained"]
        num_classes =len(cfg["datasets"]["voc_classes_list"])
        num_anchors=len(cfg["model_cfg"]["anchors"][0])
        #  backbone
        self.backbone = CSPDarkNet_tiny()

        if self.pretrain:
            print("========Loading pretrained model========")
            abspath=os.path.abspath(os.path.join(os.path.dirname(__file__),os.path.pardir,os.path.pardir,os.path.pardir,"pretrain_models"))
            self.model_path =abspath+'/yolov4_tiny_pretrain.pth'
            pretrained_dict = torch.load(self.model_path)
            model_dict = self.backbone.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k1: v for (k, v), k1 in zip(pretrained_dict.items(), model_dict)}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            self.backbone.load_state_dict(model_dict)

        self.conv_for_P5 = BasicConv(512,256,1)
        self.yolo_headP5 = yolo_head([512, num_anchors * (5 + num_classes)],256)

        self.upsample = Upsample(256,128)
        self.yolo_headP4 = yolo_head([256, num_anchors * (5 + num_classes)],384)

        self._init_weights()

    def _init_weights(self):

        def weights_init(m):
            for key in m.state_dict():
                if key.split('.')[-1] == 'weight':
                    if 'conv' in key:
                        torch.nn.init.kaiming_normal_(m.state_dict()[key], mode='fan_out')
                    if 'bn' in key:
                        m.state_dict()[key][...] = 1
                elif key.split('.')[-1] == 'bias':
                    m.state_dict()[key][...] = 0

        weights_init(self.conv_for_P5)
        weights_init(self.yolo_headP5)
        weights_init(self.upsample)
        weights_init(self.yolo_headP4)

    def forward(self, x):
        #  backbone
        feat1, feat2 = self.backbone(x)
        P5 = self.conv_for_P5(feat2)
        out0 = self.yolo_headP5(P5) 

        P5_Upsample = self.upsample(P5)
        P4 = torch.cat([feat1,P5_Upsample],axis=1)

        out1 = self.yolo_headP4(P4)
        
        return out0, out1

