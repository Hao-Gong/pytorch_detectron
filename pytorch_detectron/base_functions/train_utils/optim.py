# Author: Zylo117

import os

import cv2
import numpy as np
import torch

def burnin_schedule(i):
    if i < 100:
        factor = pow(i / 100, 4)
    elif i < 400000:
        factor = 1.0
    elif i < 450000:
        factor = 0.1
    else:
        factor = 0.01
    return factor

class optimizerSchedulerController(object):
    def __init__(self,model,cfg):
        self.cfg=cfg
        if cfg["model_cfg"]['optim']== 'adamw':
            self.optimizer= torch.optim.AdamW(model.parameters(), cfg["model_cfg"]["lr"])
        elif cfg["model_cfg"]['optim'] == "adam":
            self.optimizer= torch.optim.Adam(model.parameters(), lr=cfg["model_cfg"]["lr"],weight_decay=cfg["model_cfg"]["weight_decay"])
        else:
            self.optimizer= torch.optim.SGD(model.parameters(), lr=cfg["model_cfg"]["lr"], momentum=cfg["model_cfg"]["momentum"])

        if cfg["model_cfg"]["lr_scheduler"]=='LambdaLR':
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, burnin_schedule)
        elif cfg["model_cfg"]["lr_scheduler"]=='ReduceLROnPlateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=3, verbose=True)
        else:
            self.scheduler =  torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=cfg["model_cfg"]["lr_scheduler_step_decay"], gamma=cfg["model_cfg"]["lr_scheduler_step_gamma"])

    def schedulerStep(self,lossMean=0.0):
        if self.cfg["model_cfg"]["lr_scheduler"]=='ReduceLROnPlateau':
            self.scheduler.step(lossMean)
        else:
            self.scheduler.step()
            print("learning rate: %f" % (self.scheduler.get_last_lr()[0]))
        
    
    # learning rate setup for yolo



