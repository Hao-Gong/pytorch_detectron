
# coding: utf-8
from pytorch_detectron.detector import detector
import cv2

# 加载一个训练好的模型，可以初始化的时候输入模型的路径,可以是本系统导出的任意的一种网络
# 加载一个fpn_resnet101训练好的模型来预测
uaesDetector=detector('/home/gong/fpn_resnet101_VOC_epoch79_mAP_0.7178.pth')
# 加载一个yolov4_tiny训练好的模型来预测
# uaesDetector=detector('/home/gong/yoloV4_tiny_VOC_epoch495_mAP_0.5451.pth')

dets,drawImage=uaesDetector.predict_one_image("/home/gong/demo.jpg")

# 输出的dets是一个框的list [classname,xmin,ymin,xmax,ymax,score]
# ['person', 742.6838, 369.11438, 988.8707, 1080.4215, 0.9985789], ['person', 969.66345, 386.49753, 1223.5818, 1077.6532, 0.99698955]
print(dets)
imagergb = drawImage[:,:,::-1] 	# transform image to rgb
plt.figure(dpi=200)
plt.imshow(imagergb)
plt.show()


# 在工业上应用的时候，可以用训练好的网路自动生成标注，大大减少工作量
# 调用export_xml_annotation()函数，第一个参数是图片路径，第二个是输出xml标注的目录
uaesDetector.export_xml_annotation("/home/gong/demo.jpg","/home/gong")