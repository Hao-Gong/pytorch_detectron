import torch
import torch.nn as nn
from collections import OrderedDict
from ...backbone_models.darknet.darknet53 import darknet53
import os
def conv2d(filter_in, filter_out, kernel_size):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=1, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))

def make_last_layers(filters_list, in_filters, out_filter):
    m = nn.ModuleList([
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        nn.Conv2d(filters_list[1], out_filter, kernel_size=1,
                                        stride=1, padding=0, bias=True)
    ])
    return m

class YoloV3Body(nn.Module):
    def __init__(self, cfg):
        super(YoloV3Body, self).__init__()
        self.cfg=cfg
        self.pretrain = cfg["model_cfg"]["pretrained"]
        num_classes =len(cfg["datasets"]["voc_classes_list"])
        num_anchors=len(cfg["model_cfg"]["anchors"][0])
        #  backbone
        self.backbone = darknet53()
        if self.pretrain:
            print("========Loading pretrained model========")
            abspath=os.path.abspath(os.path.join(os.path.dirname(__file__),os.path.pardir,os.path.pardir,os.path.pardir,"pretrain_models"))
            self.model_path =abspath+'/yoloV3_pretrain.pth'
            pretrained_dict = torch.load(self.model_path)
            model_dict = self.backbone.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k1: v for (k, v), k1 in zip(pretrained_dict.items(), model_dict)}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            self.backbone.load_state_dict(model_dict)
        out_filters = self.backbone.layers_out_filters
        #  last_layer0
        final_out_filter0 = num_anchors* (5 + num_classes)
        self.last_layer0 = make_last_layers([512, 1024], out_filters[-1], final_out_filter0)

        #  embedding1
        final_out_filter1 =num_anchors* (5 + num_classes)
        self.last_layer1_conv = conv2d(512, 256, 1)
        self.last_layer1_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer1 = make_last_layers([256, 512], out_filters[-2] + 256, final_out_filter1)

        #  embedding2
        final_out_filter2 = num_anchors* (5 + num_classes)
        self.last_layer2_conv = conv2d(256, 128, 1)
        self.last_layer2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer2 = make_last_layers([128, 256], out_filters[-3] + 128, final_out_filter2)


    def forward(self, x):
        def _branch(last_layer, layer_in):
            for i, e in enumerate(last_layer):
                layer_in = e(layer_in)
                if i == 4:
                    out_branch = layer_in
            return layer_in, out_branch
        #  backbone
        x2, x1, x0 = self.backbone(x)
        #  yolo branch 0
        out0, out0_branch = _branch(self.last_layer0, x0)

        #  yolo branch 1
        x1_in = self.last_layer1_conv(out0_branch)
        x1_in = self.last_layer1_upsample(x1_in)
        x1_in = torch.cat([x1_in, x1], 1)
        out1, out1_branch = _branch(self.last_layer1, x1_in)

        #  yolo branch 2
        x2_in = self.last_layer2_conv(out1_branch)
        x2_in = self.last_layer2_upsample(x2_in)
        x2_in = torch.cat([x2_in, x2], 1)
        out2, _ = _branch(self.last_layer2, x2_in)
        return out0, out1, out2

