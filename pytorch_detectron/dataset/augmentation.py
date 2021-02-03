# import albumentations as albu
# from albumentations.pytorch.transforms import ToTensor
import torch
import numpy as np
import cv2
import math

transformerDict={"BrightJitter":False,"HueSaturationJitter":False,"RandomCroper":False, "HorizontalFliper":False,"VerticalFliper":False, "RandomBlur":False,"NoiseAdder":False,"RandomAffiner":False,"Resizer":False,"Normalizer":False,"ToTensor":False,"TransformAnnotXYWH":False}

def parseCfgTransformer(cfg):
    augTransformerObjList=[]
    toTensorTransformerObjList=[]
    for transformsName in cfg["train_dataset_transforms"]:
        transformerDict[transformsName]=True

    for transformsName in transformerDict:
        if transformsName=="Resizer" and transformerDict[transformsName] ==True:
            augTransformerObjList.append(Resizer(common_size=cfg["train_image_resize"],padding_to_rect=cfg["padding_to_rect"]))
        elif transformsName=="HorizontalFliper" and transformerDict[transformsName] ==True:
            augTransformerObjList.append(HorizontalFliper())
        elif transformsName=="BrightJitter" and transformerDict[transformsName] ==True:
            augTransformerObjList.append(BrightJitter())
        elif transformsName=="VerticalFliper" and transformerDict[transformsName] ==True:
            augTransformerObjList.append(VerticalFliper())
        elif transformsName=="HueSaturationJitter" and transformerDict[transformsName] ==True:
            augTransformerObjList.append(HueSaturationJitter())
        elif transformsName=="RandomCroper" and transformerDict[transformsName] ==True:
            augTransformerObjList.append(RandomCroper())
        elif transformsName=="RandomBlur" and transformerDict[transformsName] ==True:
            augTransformerObjList.append(RandomBlur())
        elif transformsName=="NoiseAdder" and transformerDict[transformsName] ==True:
            augTransformerObjList.append(NoiseAdder())
        elif transformsName=="NoiseAdder" and transformerDict[transformsName] ==True:
            augTransformerObjList.append(NoiseAdder())
        # elif transformsName=="RandomSelfMosaic" and transformerDict[transformsName] ==True:
        #     augTransformerObjList.append(RandomSelfMosaic())
        elif transformsName=="Normalizer" and transformerDict[transformsName] ==True:
            toTensorTransformerObjList.append(Normalizer(mean=cfg["mean"],std=cfg["std"]))
        elif transformsName=="TransformAnnotXYWH" and transformerDict[transformsName] ==True:
            toTensorTransformerObjList.append(TransformAnnotXYWH())
        elif transformsName=="ToTensor" and transformerDict[transformsName] ==True:
            toTensorTransformerObjList.append(ToTensor())
    return augTransformerObjList,toTensorTransformerObjList


def showImageWithAnnots(annots,image,name="images"):
    num_boxes=annots.shape[0]
    image=np.array(image,dtype=np.uint8)
    for i in range(num_boxes):
        x_min,y_min,x_max,y_max,c=annots[i]
        x_min=x_min
        y_min=y_min
        x_max=x_max+1
        y_max=y_max+1
        image=cv2.rectangle(image,(int(x_min),int(y_min)),(int(x_max),int(y_max)),(0,255,0),1,8)
    cv2.imshow(name,image)
    cv2.waitKey(10)

def detection_collate(batch):
    imgs = [s['image'] for s in batch]
    annots = [s['bboxes'] for s in batch]
    labels = [s['category_id'] for s in batch]

    max_num_annots = max(len(annot) for annot in annots)
    annot_padded = np.ones((len(annots), max_num_annots, 5))*-1

    if max_num_annots > 0:
        for idx, (annot, lab) in enumerate(zip(annots, labels)):
            if len(annot) > 0:
                annot_padded[idx, :len(annot), :4] = annot
                annot_padded[idx, :len(annot), 4] = lab
    return (torch.stack(imgs, 0), torch.FloatTensor(annot_padded))

def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]
    #  im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)
    image_info= [[s['img'].shape[0],s['img'].shape[1],s['scale']] for s in data] 
    box_num= [s['annot'].shape[0] for s in data]
    imgs = torch.from_numpy(np.stack(imgs, axis=0))

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:
        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    imgs = imgs.permute(0, 3, 1, 2)

    return {"images":imgs, "annot":torch.FloatTensor(annot_padded),"image_info":torch.FloatTensor(image_info),"num_boxes":torch.FloatTensor(box_num)}

# after normalizer
class Resizer(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, common_size=768,padding_to_rect=True,single_img=False,padding_mode="leftTop",posibility=0.3):
        self.common_size=common_size
        self.base_common_size=common_size
        self.padding_to_rect=padding_to_rect
        self.single_img=single_img
        self.padding_mode=padding_mode
        self.posibility=posibility

    def __call__(self, sample):
            
        if self.single_img:
            image= sample['img']
            height, width, _ = image.shape
            
            if height > width:
                scale = self.common_size / height
                resized_height = self.common_size
                resized_width = int(width * scale)
            else:
                scale = self.common_size / width
                resized_height = int(height * scale)
                resized_width = self.common_size

            image = cv2.resize(image, (resized_width, resized_height))

            if self.padding_to_rect is False:
                return {'img':image, 'scale': scale}
            else:
                new_image = np.zeros((self.common_size, self.common_size, 3))
                new_image[0:resized_height, 0:resized_width] = image
                return {'img': new_image,'scale': scale}

        else:
            image, annots = sample['img'], sample['annot']
            originImage=sample["origin_image"]
            addition_info=sample["addition_info"]
            height, width, _ = image.shape

            if height  > width:
                scale = self.common_size / height
                resized_height = self.common_size
                resized_width = int(width * scale)
            else:
                scale = self.common_size / width
                resized_height = int(height * scale)
                resized_width = self.common_size

            image = cv2.resize(image, (resized_width, resized_height))
            annots[:, :4] *= scale

            if self.padding_to_rect is False:
                return {'img': image, 'annot': annots, 'scale': scale,"origin_image":originImage}
            else:
                new_image = np.zeros((self.common_size, self.common_size, 3))
                self.padding_mode=addition_info["padding_mode"]
                if self.padding_mode=="leftTop":
                    
                    new_image[0:resized_height, 0:resized_width] = image
                    return {'img': new_image, 'annot': annots, 'scale': scale,"origin_image":originImage}

                elif  self.padding_mode=="leftBottom":
                    height, width, _ = image.shape
                    random_scale=np.random.uniform(0.9,1.0)
                    if np.random.rand() < self.posibility:
                        random_height=int(np.random.uniform(height*3/4, height)*random_scale)
                        random_width=int(np.random.uniform(width*3/4, width)*random_scale)
                    else:
                        random_height=int(height*random_scale)
                        random_width=int(width*random_scale)
                    image = cv2.resize(image, (random_width, random_height))

                    annots[:, 0]=annots[:, 0]*random_width/width
                    annots[:, 2]=annots[:, 2]*random_width/width
                    annots[:, 1]=annots[:, 1]*random_height/height+self.common_size-random_height
                    annots[:, 3]=annots[:, 3]*random_height/height+self.common_size-random_height

                    new_image[self.common_size-random_height:, 0:random_width] = image
                    # showImageWithAnnots(annots,new_image)
                    return {'img': new_image, 'annot': annots, 'scale': scale,"origin_image":originImage}

                elif  self.padding_mode=="rightTop":
                    height, width, _ = image.shape
                    random_scale=np.random.uniform(0.9,1.0)
                    if np.random.rand() < self.posibility:
                        random_height=int(np.random.uniform(height*3/4, height)*random_scale)
                        random_width=int(np.random.uniform(width*3/4, width)*random_scale)
                    else:
                        random_height=int(height*random_scale)
                        random_width=int(width*random_scale)
                    image = cv2.resize(image, (random_width, random_height))

                    annots[:, 0]=annots[:, 0]*random_width/width+self.common_size-random_width
                    annots[:, 2]=annots[:, 2]*random_width/width+self.common_size-random_width
                    annots[:, 1]=annots[:, 1]*random_height/height
                    annots[:, 3]=annots[:, 3]*random_height/height
                    new_image[0:random_height, self.common_size-random_width:] = image
                    # showImageWithAnnots(annots,new_image)
                    return {'img': new_image, 'annot': annots, 'scale': scale,"origin_image":originImage}

                elif  self.padding_mode=="rightBottom":
                    height, width, _ = image.shape
                    random_scale=np.random.uniform(0.9,1.0)
                    if np.random.rand() < self.posibility:
                        random_height=int(np.random.uniform(height*3/4, height)*random_scale)
                        random_width=int(np.random.uniform(width*3/4, width)*random_scale)
                    else:
                        random_height=int(height*random_scale)
                        random_width=int(width*random_scale)
                    image = cv2.resize(image, (random_width, random_height))
                    annots[:, 0]=annots[:, 0]*random_width/width+self.common_size-random_width
                    annots[:, 2]=annots[:, 2]*random_width/width+self.common_size-random_width
                    annots[:, 1]=annots[:, 1]*random_height/height+self.common_size-random_height
                    annots[:, 3]=annots[:, 3]*random_height/height+self.common_size-random_height
                    new_image[self.common_size-random_height:, self.common_size-random_width:] = image
                    # showImageWithAnnots(annots,new_image)
                    return {'img': new_image, 'annot': annots, 'scale': scale,"origin_image":originImage}

                elif  self.padding_mode=="randomPadding":
                    
                    # random resize and padding
                    height, width, _ = image.shape
                    random_scale=np.random.uniform(0.6,1.0)
                    if np.random.rand() < self.posibility:
                        random_height=int(np.random.uniform(height*3/4, height)*random_scale)
                        random_width=int(np.random.uniform(width*3/4, width)*random_scale)
                    else:
                        random_height=int(height*random_scale)
                        random_width=int(width*random_scale)

                    random_width_shift=int(np.random.uniform(0,(width-random_width)))
                    random_height_shift=int(np.random.uniform(0,(height-random_height)))
                    annots[:, 0]=annots[:, 0]*random_width/width+random_width_shift
                    annots[:, 2]=annots[:, 2]*random_width/width+random_width_shift
                    annots[:, 1]=annots[:, 1]*random_height/height+random_height_shift
                    annots[:, 3]=annots[:, 3]*random_height/height+random_height_shift

                    image = cv2.resize(image, (random_width, random_height))
                    new_image[random_height_shift:(random_height+random_height_shift), random_width_shift:(random_width+random_width_shift)] = image
                    return {'img': new_image, 'annot': annots, 'scale': scale,"origin_image":originImage,"addition_info":addition_info}


# after normalizer
class RandomAffiner(object):
    def __init__(self, posibility = 0.5, degrees=10, translate=.05, scale=.1, shear=10, border=0):
        self.posibility = 0.5
        self.degree = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.border = border
    def __call__(self,sample):
        if  np.random.rand() < self.posibility:
            img, annots, origin_img = sample['img'], sample['annot'], sample['origin_image']
            height = img.shape[0] + self.border * 2
            width = img.shape[1] + self.border * 2
            # Rotation and Scale
            R = np.eye(3)
            a = np.random.uniform(-self.degree, self.degree)
            # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
            s = np.random.uniform(1 - self.scale, 1 + self.scale)
            # s = 2 ** random.uniform(-scale, scale)
            R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

            # Translation
            T = np.eye(3)
            T[0, 2] = np.random.uniform(-self.translate, self.translate) * img.shape[1] + self.border  # x translation (pixels)
            T[1, 2] = np.random.uniform(-self.translate, self.translate) * img.shape[0] + self.border  # y translation (pixels)

            # Shear
            S = np.eye(3)
            S[0, 1] = math.tan(np.random.uniform(-self.shear, self.shear) * math.pi / 180)  # x shear (deg)
            S[1, 0] = math.tan(np.random.uniform(-self.shear, self.shear) * math.pi / 180)  # y shear (deg)

            # Combined rotation matrix
            M = S @ T @ R  # ORDER IS IMPORTANT HERE!!
            if (self.border != 0) or (self.border != 0) or (M != np.eye(3)).any():  # image changed
                img = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))
            
            n = annots.shape[0]
            if n > 0:
                # warp points
                xy = np.ones((n * 4, 3))
                xy[:, :2] = annots[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
                xy = (xy @ M.T)[:, :2].reshape(n, 8)

                # create new boxes
                x = xy[:, [0, 2, 4, 6]]
                y = xy[:, [1, 3, 5, 7]]
                xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

                # reject warped points outside of image
                xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
                xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
                w = xy[:, 2] - xy[:, 0]
                h = xy[:, 3] - xy[:, 1]
                area = w * h
                area0 = (annots[:, 2] - annots[:, 0]) * (annots[:, 3] - annots[:, 1])
                ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))  # aspect ratio
                i = (w > 2) & (h > 2) & (area / (area0 * s + 1e-16) > 0.2) & (ar < 20)

                annots_new = annots[i]
                annots_new[:, 0:4] = xy[i]

            sample = {'img': img, 'annot': annots_new,"origin_image":origin_img}

            # self.showImageWithAnnots(annots,origin_img,annots,img)

        return sample


                
# after normalizer
class HorizontalFliper(object):
    def __init__(self, flip_posibility=0.2,single_img=False):
        self.flip_posibility=flip_posibility
        self.single_img=single_img
    def __call__(self, sample):
        if np.random.rand() < self.flip_posibility:
            image, annots = sample['img'], sample['annot']
            originImage=sample["origin_image"]
            addition_info=sample["addition_info"]
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {'img': image, 'annot': annots,"origin_image":originImage,"addition_info":addition_info}
        return sample

# after normalizer
class VerticalFliper(object):
    def __init__(self, flip_posibility=0.2,single_img=False):
        self.flip_posibility=flip_posibility
        self.single_img=single_img
    def __call__(self, sample):
        if np.random.rand() < self.flip_posibility:
            image, annots = sample['img'], sample['annot']
            originImage=sample["origin_image"]
            addition_info=sample["addition_info"]
            image = image[::-1, :, :]
            rows, cols, channels = image.shape

            y1 = annots[:, 1].copy()
            y2 = annots[:, 3].copy()

            y_tmp = y1.copy()

            annots[:, 1] = rows - y2
            annots[:, 3] = rows - y_tmp

            sample = {'img': image, 'annot': annots,"origin_image":originImage,"addition_info":addition_info}
        return sample

# before normalizer
class BrightJitter(object):
    def __init__(self, posibility=0.2,delta=0.5,single_img=False):
        self.posibility=posibility
        self.delta=delta
        self.single_img=single_img
    
    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        originImage=sample["origin_image"]
        addition_info=sample["addition_info"]
        if  np.random.rand() < self.posibility:
            delta = np.random.uniform(-self.delta, self.delta)
            image += delta
            image = image.clip(min=0, max=255)

        sample = {'img': image, 'annot': annots,"origin_image":originImage,"addition_info":addition_info}
        return sample


# before normalizer
class HueSaturationJitter(object):
    def __init__(self, posibility=0.2,single_img=False):
        self.posibility=posibility
        self.single_img=single_img

    def random_hue(self, im, delta=18.0):
        if  np.random.rand() < self.posibility:
            im[:, :, 0] += np.random.uniform(-delta, delta)
            im[:, :, 0][im[:, :, 0] > 360.0] -= 360.0
            im[:, :, 0][im[:, :, 0] < 0.0] += 360.0
        return im

    def random_saturation(self, im, lower=0.7, upper=0.9):
        if  np.random.rand() < self.posibility:
            im[:, :, 1] *= np.random.uniform(lower, upper)
        return im

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        originImage=sample["origin_image"]
        addition_info=sample["addition_info"]

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image = self.random_saturation(image)
        image = self.random_hue(image)
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        sample = {'img': image, 'annot': annots,"origin_image":originImage,"addition_info":addition_info}
        return sample

# before normalizer
class RandomCroper(object):
    def __init__(self, posibility=0.2,single_img=False):
        self.posibility=posibility
        self.single_img=single_img
    
    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        originImage=sample["origin_image"]
        addition_info=sample["addition_info"]

        if  np.random.rand() < self.posibility:
            height, width, _ = image.shape
            left=min(annots[:, 0])
            right=max(annots[:, 2])
            top=min(annots[:, 1])
            down=max(annots[:, 3])
            x_min=int(np.random.uniform(0,left))
            y_min=int(np.random.uniform(0,top))
            x_max=int(np.random.uniform(right,width))
            y_max=int(np.random.uniform(down,height))
            image=image[y_min:y_max,x_min:x_max,:]
            annots[:, 0]-=x_min
            annots[:, 2]-=x_min
            annots[:, 1]-=y_min
            annots[:, 3]-=y_min

        sample = {'img': image, 'annot': annots,"origin_image":originImage,"addition_info":addition_info}
        return sample


# before normalizer
class RandomBlur(object):
    def __init__(self, posibility=0.2,single_img=False):
        self.posibility=posibility
        self.single_img=single_img
    
    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        originImage=sample["origin_image"]
        addition_info=sample["addition_info"]

        if  np.random.rand() < self.posibility:
            wsize=int(np.random.uniform(1,3))*2+1
            image=cv2.GaussianBlur(image, (wsize,wsize), 0)
        elif np.random.rand() < self.posibility/(1-self.posibility):
            wsize=pow(int(np.random.uniform(2,4)),2)
            std=int(np.random.uniform(5,50))
            image=cv2.bilateralFilter(image, wsize, std, std)

        sample = {'img': image, 'annot': annots,"origin_image":originImage,"addition_info":addition_info}
        return sample

# noise adder
class NoiseAdder(object):
    def __init__(self, posibility=0.2,single_img=False):
        self.posibility=posibility
        self.single_img=single_img
    
    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        originImage=sample["origin_image"]
        addition_info=sample["addition_info"]
        if np.random.rand() < self.posibility:
            size =  image.shape[0]*image.shape[1]
            snr=np.random.uniform(0.95,0.99)
            # 因为信噪比是 SNR ，所以噪声占据百分之10，所以需要对这百分之10加噪声
            noiseSize = int(size * (1 - snr))
            # 对这些点加噪声
            for k in range(0, noiseSize):
                # 随机获取 某个点
                xi = int(np.random.uniform(0, image.shape[1]))
                xj = int(np.random.uniform(0, image.shape[0]))
                # 增加噪声
                if np.random.rand() < 0.5:
                    image[xj, xi,:] = 0
                else:
                    image[xj, xi,:] = 255

        sample = {'img': image, 'annot': annots,"origin_image":originImage,"addition_info":addition_info}
        return sample


def grubAnnot(img,annots):
        h,w=img.shape[:2]
        img=img.astype(np.uint8)
        roiList=[]
        for annot in annots:
            xmin,ymin,xmax,ymax,c=annot

            xmin=int(xmin)
            ymin=int(ymin)
            xmax=int(xmax)
            ymax=int(ymax)

            aw=xmax-xmin
            ah=ymax-ymin
            area=aw*ah

            xmin=max(xmin-5,0)
            xmax=min(w,xmax+5)
            ymin=max(ymin-5,0)
            ymax=min(h,ymax+5)

            xmin_lagger=max(xmin-aw//2,0)
            xmax_lagger=min(w,xmax+aw//2)
            ymin_lagger=max(ymin-ah//2,0)
            ymax_lagger=min(h,ymax+ah//2)

            roi_lagger=img[ymin_lagger:ymax_lagger,xmin_lagger:xmax_lagger,:]
            mask = np.zeros(roi_lagger.shape[:2], np.uint8)

            bgdModel = np.zeros((1,65), np.float64)
            fgdModel = np.zeros((1,65), np.float64)

            rect = (xmin-xmin_lagger, ymin-ymin_lagger, xmax - xmin, ymax - ymin)
            # print(rect,roi_lagger.shape)
            iterCount = 5
            cv2.grabCut(roi_lagger, mask, rect, bgdModel, fgdModel, iterCount, cv2.GC_INIT_WITH_RECT)
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

            mask_loc=np.where(mask2>0)
            mask_area=len(mask_loc[0])
        
            roi_lagger = roi_lagger * mask2[:, :, np.newaxis]

            if mask_area/area>0.5:
                shrink_xmin=min(mask_loc[1])
                shrink_ymin=min(mask_loc[0])
                shrink_xmax=max(mask_loc[1])
                shrink_ymax=max(mask_loc[0])
                roiList.append([roi_lagger[shrink_ymin:shrink_ymax,shrink_xmin:shrink_xmax,:],c,shrink_xmin+xmin_lagger,shrink_ymin+ymin_lagger])

        return roiList

# grubPadding
def GrubRandomPadding(sample1,sample2=None):
        img1,annot1=sample1["img"],sample1["annot"]
        originImage=sample1["origin_image"]
        # img2,annot2=sample2["img"],sample2["annot"]
        try:
            roiList= grubAnnot(img1,annot1)
        except:
            return {"img":img1,"annot":annot1,"origin_image":originImage}
        # roiList.extend(grubAnnot(img2,annot2))

        if np.random.rand() > 0.2:
            return {"img":img1,"annot":annot1,"origin_image":originImage}

        h,w,_=img1.shape
        # h2,w2,_=img2.shape
        # w=max(w1,w2)
        # h=max(h1,h2)
        new_img=np.random.rand(h,w,3)
        new_img=(new_img-np.min(new_img))*np.random.uniform(100,200)
        newAnnots=[]

        for roi in roiList:
            imgRoi,c,padding_x,padding_y= roi
            roi_h,roi_w,_=imgRoi.shape
            # rand_scale=np.random.uniform(0.5,1.0)
            # imgRoi=cv2.resize(imgRoi,(int(roi_w*rand_scale),int(roi_h*rand_scale)))
            # roi_h,roi_w,_=imgRoi.shape
            # padding_x=int(np.random.uniform(0,w-roi_w-1))
            # padding_y=int(np.random.uniform(0,h-roi_h-1))
            new_img_roi=new_img[padding_y:padding_y+roi_h,padding_x:padding_x+roi_w,:]
            new_img[padding_y:padding_y+roi_h,padding_x:padding_x+roi_w,:]=np.where(imgRoi==0,new_img_roi,imgRoi)
            newAnnots.append([padding_x,padding_y,padding_x+roi_w,padding_y+roi_h,c])
        if len(newAnnots)==0:
            return {"img":img1,"annot":annot1,"origin_image":originImage}
        else:
            return {"img":new_img.astype(np.float32),"annot":np.array(newAnnots),"origin_image":originImage}


def Mosaic4Pics(sample1,sample2,sample3,sample4):
        originImage=sample1["origin_image"]
        scale=sample1["scale"]
        img1,annot1=sample1["img"],sample1["annot"]
        img2,annot2=sample2["img"],sample2["annot"]
        img3,annot3=sample3["img"],sample3["annot"]
        img4,annot4=sample4["img"],sample4["annot"]
        height, width, _ = img1.shape
        split_x=int(np.random.uniform(width//3,width//3*2))
        split_y=int(np.random.uniform(height//3,height//3*2))
        split_x_mirror=width-split_x
        split_y_mirror=height-split_y

        leftTopImage = img1[:split_y,:split_x,]
        leftBottomImage = img2[split_y:,:split_x,]
        rightTopImage = img3[:split_y,split_x:,]
        rightBottomImage = img4[split_y:,split_x:,]

        leftTopAnnots=[]
        leftBottomAnnots=[]
        rightTopAnnots=[]
        rightBottomAnnots=[]
        newAnnots=[]
        tolScale=5

        for annot in annot1:
            x_min,y_min,x_max,y_max,c=annot
            x_tolerance=(x_max-x_min)/tolScale
            y_tolerance=(y_max-y_min)/tolScale
            # annot has enough area in each mosaic
            if y_min+y_tolerance<split_y and x_min+x_tolerance<split_x:
                _x_max=min(split_x-1,x_max)
                _y_max=min(split_y-1,y_max)
                leftTopAnnots.append([x_min,y_min,_x_max,_y_max,c])

        for annot in annot2:
            x_min,y_min,x_max,y_max,c=annot
            x_tolerance=(x_max-x_min)/tolScale
            y_tolerance=(y_max-y_min)/tolScale
            # annot has enough area in each mosaic
            if y_max-y_tolerance>split_y and x_min+x_tolerance<split_x:
                _x_max=min(split_x-1,x_max)
                _y_min=max(split_y,y_min)
                leftBottomAnnots.append([x_min,_y_min,_x_max,y_max,c])

        for annot in annot3:
            x_min,y_min,x_max,y_max,c=annot
            x_tolerance=(x_max-x_min)/tolScale
            y_tolerance=(y_max-y_min)/tolScale
            # annot has enough area in each mosaic
            if y_min+y_tolerance<split_y and x_max-x_tolerance>split_x:
                _x_min=max(split_x,x_min)
                _y_max=min(split_y-1,y_max)
                rightTopAnnots.append([_x_min,y_min,x_max,_y_max,c])

        for annot in annot4:
            x_min,y_min,x_max,y_max,c=annot
            x_tolerance=(x_max-x_min)/tolScale
            y_tolerance=(y_max-y_min)/tolScale
            # annot has enough area in each mosaic
            if y_max-y_tolerance>split_y and x_max-x_tolerance>split_x:
                _x_min=max(split_x,x_min)
                _y_min=max(split_y,y_min)
                rightBottomAnnots.append([_x_min,_y_min,x_max,y_max,c])

        if len(leftTopAnnots)>0:    newAnnots.extend(leftTopAnnots)
        if len(leftBottomAnnots)>0:   newAnnots.extend(leftBottomAnnots)
        if len(rightTopAnnots)>0:   newAnnots.extend(rightTopAnnots)
        if len(rightBottomAnnots)>0:   newAnnots.extend(rightBottomAnnots) 

        leftImage=np.concatenate((leftTopImage,leftBottomImage),axis=0)
        rightImage=np.concatenate((rightTopImage,rightBottomImage),axis=0)
        mosaicImage=np.concatenate((leftImage,rightImage),axis=1)

        if len(newAnnots)>0:
            return {"img":mosaicImage,"annot":np.array(newAnnots),'scale': scale,"origin_image":originImage}
        else:
            return {"img":img1,"annot":annot1,'scale': scale,"origin_image":originImage}

# before normalizer
class RandomSelfMosaic(object):
    def __init__(self, posibility=0.2,single_img=False):
        self.posibility=posibility
        self.single_img=single_img
    
    def centerPointSplit(self,image,annots):
        height, width, _ = image.shape
        split_x=int(np.random.uniform(width//3,width//3*2))
        split_y=int(np.random.uniform(height//3,height//3*2))
        split_x_mirror=width-split_x
        split_y_mirror=height-split_y

        topLeftImage = image[:split_y,:split_x,]
        topRightImage = image[split_y:,:split_x,]
        downLeftImage = image[:split_y,split_x:,]
        downRightImage = image[split_y:,split_x:,]

        topLeftAnnots=[]
        topRightAnnots=[]
        downLeftAnnots=[]
        downRightAnnots=[]
        newAnnots=[]
        for annot in annots:
            x_min,y_min,x_max,y_max,c=annot
            x_tolerance=(x_max-x_min)/4
            y_tolerance=(y_max-y_min)/4

            # annot has enough area in each mosaic
            if y_min+y_tolerance<split_y and x_min+x_tolerance<split_x:
                _x_max=min(split_x-1,x_max)
                _y_max=min(split_y-1,y_max)
                topLeftAnnots.append([x_min+split_x_mirror,y_min+split_y_mirror,_x_max+split_x_mirror,_y_max+split_y_mirror,c])

            if y_min+y_tolerance<split_y and x_max-x_tolerance>=split_x:
                _x_min=max(split_x,x_min)
                _y_max=min(split_y-1,y_max)
                topRightAnnots.append([_x_min-split_x,y_min+split_y_mirror,x_max-split_x,_y_max+split_y_mirror,c])

            if y_max-y_tolerance>=split_y and x_min+x_tolerance<split_x:
                _x_max=min(split_x-1,x_max)
                _y_min=max(split_y,y_min)
                downLeftAnnots.append([x_min+split_x_mirror,_y_min-split_y,_x_max+split_x_mirror,y_max-split_y,c])

            if y_max-y_tolerance>=split_y and x_max-x_tolerance>=split_x:
                _x_min=max(split_x,x_min)
                _y_min=max(split_y,y_min)
                downRightAnnots.append([_x_min-split_x,_y_min-split_y,x_max-split_x,y_max-split_y,c])

        newAnnots.extend(topLeftAnnots)
        newAnnots.extend(topRightAnnots)
        newAnnots.extend(downLeftAnnots)
        newAnnots.extend(downRightAnnots) 

        leftImage=np.concatenate((downRightImage,topRightImage),axis=1)
        rightImage=np.concatenate((downLeftImage,topLeftImage),axis=1)
        mosaicImage=np.concatenate((leftImage,rightImage),axis=0)
        return mosaicImage,np.array(newAnnots)

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        originImage=sample["origin_image"]
        addition_info=sample["addition_info"]

        if  np.random.rand() < self.posibility:
            height, width, _ = image.shape
            # if height<width:
            #     image_ext,annots_ext=self.centerPointSplit(image, annots)

            #     annots_ext[:, 1]+=height
            #     annots_ext[:, 3]+=height
            #     image=np.concatenate((image,image_ext),axis=0)
            #     annots=np.concatenate((annots,annots_ext),axis=0)
            # else:
            #     image_ext,annots_ext=self.centerPointSplit(image, annots)

            #     annots_ext[:, 0]+=width
            #     annots_ext[:, 2]+=width
            #     image=np.concatenate((image,image_ext),axis=1)
            #     annots=np.concatenate((annots,annots_ext),axis=0)
            if height<width:
                image_ext,annots_ext=self.centerPointSplit(image, annots)

                annots_ext[:, 1]+=height
                annots_ext[:, 3]+=height
                image=image_ext
                annots=annots_ext
            else:
                image_ext,annots_ext=self.centerPointSplit(image, annots)

                annots_ext[:, 0]+=width
                annots_ext[:, 2]+=width
                image=image_ext
                annots=annots_ext
            # showImageWithAnnots(annots,image)
        sample = {'img': image, 'annot': annots,"origin_image":originImage,"addition_info":addition_info}
        return sample


class Normalizer(object):
    def __init__(self,mean,std,single_img=False,color_mode="rgb"):
        self.mean=np.array([[mean]])
        self.std=np.array([[std]])
        self.single_img=single_img
        self.color_mode=color_mode
    def __call__(self, sample):

        if self.single_img:
            image= sample['img']
            scale= sample['scale']
            if self.color_mode =="rgb":
                image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
            return {'img': ((image.astype(np.float32)/255. - self.mean) / self.std), 'scale': scale}
        else:
            originImage=sample["origin_image"]
            image, annots = sample['img'], sample['annot']
            scale= sample['scale']
            if self.color_mode =="rgb":
                image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
            return {'img': ((image.astype(np.float32)/255. - self.mean) / self.std), 'annot': annots, 'scale': scale,"origin_image":originImage}

class ToTensor(object):
    def __init__(self,single_img=False):
        self.single_img=single_img

    def __call__(self, sample):
        if self.single_img:
            image = sample['img']
            scale= sample['scale']
            return {'img': torch.from_numpy(image), 'scale': scale}
        else:
            image, annots = sample['img'], sample['annot']
            scale= sample['scale']
            originImage=sample["origin_image"]
            return {'img': torch.from_numpy(image), 'annot': torch.from_numpy(annots), 'scale': scale,"origin_image":originImage}

# final one
class TransformAnnotXYWH(object):
    def __init__(self, single_img=False):
        self.single_img=single_img
    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        originImage=sample["origin_image"]
        scale=sample["scale"]
        height, width, _ = image.shape
        newAnnots=annots.clone()
        newAnnots[:,0]=(annots[:,0]+annots[:,2])/width/2
        newAnnots[:,1]=(annots[:,1]+annots[:,3])/height/2
        newAnnots[:,2]=(annots[:,2]-annots[:,0])/width
        newAnnots[:,3]=(annots[:,3]-annots[:,1])/height

        return {'img':image, 'annot': newAnnots, 'scale': scale,"origin_image":originImage}


class SingleDatasetGenerator(object):
    def __call__(self, sample):
        img =sample["img"]
        scales =sample['scale']
        image_info= [[img.shape[0],img.shape[1],scales]]
        box_num=torch.ones((1, 1, 5))
        imgs = img.unsqueeze(0)
        annot_padded = torch.ones((1, 1, 5))
        imgs = imgs.permute(0, 3, 1, 2)
        return {"images":imgs, "annot":torch.FloatTensor(annot_padded),"image_info":torch.FloatTensor(image_info),"num_boxes":torch.FloatTensor(box_num)}

def segmentation_collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]
    segfeas=[s['segs'] for s in data]
    image_info= [[s['img'].shape[0],s['img'].shape[1],s['scale']] for s in data] 
    box_num= [s['annot'].shape[0] for s in data]

    imgs = torch.from_numpy(np.stack(imgs, axis=0))
    segfeas = torch.from_numpy(np.stack(segfeas, axis=0))
    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:
        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    imgs = imgs.permute(0, 3, 1, 2)

    return {"images":imgs, "annot":torch.FloatTensor(annot_padded),"image_info":torch.FloatTensor(image_info),"num_boxes":torch.FloatTensor(box_num),"segs":torch.FloatTensor(segfeas)}


class segTargetGenerator(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, classes=20):
        self.classes=classes

    def __call__(self, sample):
        image =sample["img"]
        scales =sample['scale']
        annots = sample["annot"]
        originImage=sample["origin_image"]

        height, width, _ = image.shape
#         print(np.min(image),np.max(image))
        image=np.array(image,dtype=np.uint8)
        grayImage=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        scharrResize= np.zeros((height, width),dtype=np.uint8)
        laplaceResize= np.zeros((height, width),dtype=np.uint8)
#         blankResize= np.zeros((height, width),dtype=np.uint8)
        h,w=grayImage.shape

        # scharr
        gx = cv2.Scharr(grayImage, ddepth=cv2.CV_16S, dx=1, dy=0)
        gy = cv2.Scharr(grayImage, ddepth=cv2.CV_16S, dx=0, dy=1)
        gx_abs = cv2.convertScaleAbs(gx)
        gy_abs = cv2.convertScaleAbs(gy)
        scharr = cv2.addWeighted(src1=gx_abs, alpha=0.5, src2=gy_abs, beta=0.5, gamma=0)

        scharrResize[0:h, 0:w] =scharr
        
        # laplace
        laplace=cv2.Laplacian(grayImage,cv2.CV_16S,ksize=3)
        laplace=cv2.convertScaleAbs(laplace)
        laplaceResize[0:h, 0:w]=laplace

        num_boxes=annots.shape[0]

#         segfeas=  np.zeros((self.classes ,height, width),dtype=np.uint8)
        segfeas=  np.zeros((3,height, width),dtype=np.uint8)
        for i in range(num_boxes):
            x_min,y_min,x_max,y_max,c=annots[i]
            x_min=x_min
            y_min=y_min
            x_max=x_max
            y_max=y_max

            segfeas[0]=cv2.rectangle(segfeas[0],(int(x_min),int(y_min)),(int(x_max)+1,int(y_max)+1),255,-1,8)
            segfeas[0]=cv2.bitwise_and(segfeas[0],scharrResize)
            
            segfeas[1]=cv2.rectangle(segfeas[1],(int(x_min),int(y_min)),(int(x_max)+1,int(y_max)+1),255,-1,8)
            segfeas[1]=cv2.bitwise_and(segfeas[1],laplaceResize)
            
            segfeas[2]=cv2.rectangle(segfeas[2],(int(x_min),int(y_min)),(int(x_max)+1,int(y_max)+1),255,-1,8)
            
        segfeas=np.array(segfeas,dtype=np.float32)/255.0
        # print(type(image))
        
        self.mean=np.array([[0.485, 0.456, 0.406]])
        self.std=np.array([[0.229, 0.224, 0.225]])
        return  {'img': torch.from_numpy(((image.astype(np.float32)/255. - self.mean) / self.std)), 'annot': torch.from_numpy(annots),'scale':scales, 'segs':torch.from_numpy(segfeas)}
