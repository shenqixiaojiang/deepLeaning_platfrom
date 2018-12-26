from __future__ import division
import cv2
import numpy as np
from numpy import random
import math
from sklearn.utils import shuffle
import torch

__all__ = ['Compose','RandomHflip', 'RandomUpperCrop', 'Resize', 'UpperCrop', 'RandomBottomCrop',"RandomErasing",
           'BottomCrop', 'Normalize', 'RandomSwapChannels', 'RandomRotate', 'RandomHShift',"CenterCrop","RandomVflip",
           'ExpandBorder', 'RandomResizedCrop','RandomDownCrop', 'DownCrop', 'ResizedCrop',"FixRandomRotate"]

class Compose(object): ##transforms for each image, then 
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, img):
        img = np.array(img)
        img = img.transpose((1,2,0))
        #print img.shape
        for t in self.transforms:
            img = t(img)
        img = img.transpose((2,0,1))
        #print(img.shape)
        return torch.from_numpy(img.copy()).float()

class RandomHflip(object): ##
    def __call__(self, image):
        if random.random() < 0.5: ## 0 or 1
            return np.flip(image,axis=1) 
        else:
            return image

class RandomVflip(object): ##OK
    def __call__(self, image):
        if random.random() < 0.5: ## 0 or 1
            return np.flip(image,axis=0)
        else:
            return image

def resize_image(image,size,inter=cv2.INTER_CUBIC):
    if image.shape[0] == size[0]:
        return image
    times = image.shape[-1] // 3 + 1
    ff_array = np.zeros((size[0],size[1],image.shape[-1]))
    for ii in range(times):
        if ii == times - 1:
            cur_img = image[:,:,image.shape[-1] - 3:]
        else:
            cur_img = image[:,:,ii * 3 : ii * 3 + 3]
        cur_img = cv2.resize(cur_img, (size[0], size[1]), interpolation=inter)
        if ii == times - 1:
            ff_array[:,:,image.shape[-1] - 3:] = cur_img
        else:
            ff_array[:,:,ii * 3 : ii * 3 + 3] = cur_img
    return ff_array

def center_crop(img, size, center_rate = (0.15,0.15),inter=cv2.INTER_CUBIC):
    h, w, _ = img.shape
    left = int(w * center_rate[1])
    top = int(h * center_rate[0])
    right = w - left
    bottom = h - top
    out = resize_image(img[top:bottom,left:right,:],size,inter)
    return out

def rotate_nobound(image, angle, center=None, scale=1.):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated

class RandomRotate(object):
    def __init__(self, angles, bound=False):
        self.angles = angles
        self.bound = bound

    def __call__(self,img):
        if random.random() < 0.5:
            angle = np.random.uniform(self.angles[0], self.angles[1])
            if self.bound:
                img = rotate_nobound(img, angle)
            else:
                img = rotate_nobound(img, angle)
        return img

class CenterCrop(object):
    def __init__(self, size, center_rate = (0.15,0.15), inter=cv2.INTER_CUBIC):
        self.size = size
        self.center_rate = center_rate
        self.inter = inter
    def __call__(self, image):
        return center_crop(image, self.size,self.center_rate,self.inter)

class Resize(object):
    def __init__(self, size, inter=cv2.INTER_CUBIC):
        self.size = size
        self.inter = inter

    def __call__(self, image):
        return resize_image(image,self.size,self.inter)

class RandomResizedCrop(object):
    def __init__(self, size, scale=(0.16, 1.0), ratio=(0.75, 1.25),inter=cv2.INTER_CUBIC):
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.inter = inter

    def __call__(self,img):
        if random.random() < 0.2:
            return resize_image(img,self.size,self.inter)
        h, w, cn = img.shape
        area = h * w
        d = 1
        for attempt in range(10):
            target_area = random.uniform(self.scale[0], self.scale[1]) * area
            aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])

            new_w = int(round(math.sqrt(target_area * aspect_ratio)))
            new_h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                new_h, new_w = new_w, new_h

            if new_w < w and new_h < h:
                x0 = random.randint(0, w - new_w)
                y0 = (random.randint(0, h - new_h)) // d
                out = resize_image(img[y0 : y0+new_h,x0 : x0+new_w,:],self.size,self.inter)
                return out
        return center_crop(img, self.size, inter=self.inter)

class RandomErasing(object):
    def __init__(self, sl=0.02, sh=0.09, r1=0.3, mean=[0.485, 0.456, 0.406]):
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        if random.uniform(0, 1) < 0.3:
            return img
        if len(self.mean) != img.shape[-1]:
            cur_img = img.reshape((-1,img.shape[-1]))
            self.mean = cur_img.mean(axis=0)
        for attempt in range(10):
            area = img.shape[0] * img.shape[1]
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w <= img.shape[0] and h <= img.shape[1]:
                x1 = random.randint(0, img.shape[0] - h)
                y1 = random.randint(0, img.shape[1] - w)
                for ii in range(len(self.mean)):
                    img[ii, x1:x1 + h, y1:y1 + w] = self.mean[ii]
                break
        return img
   
class Normalize(object):
    def __init__(self,mean, std):
        '''
        :param mean: RGB order
        :param std:  RGB order
        '''
        self.mean = np.array(mean).reshape(3,1,1)
        self.std = np.array(std).reshape(3,1,1)
    def __call__(self, image):
        '''
        :param image:  (H,W,3)  RGB
        :return:
        '''
        return (image.transpose((2, 0, 1)) / 255. - self.mean) / self.std



'''
if __name__ == '__main__':
    import matplotlib.pyplot as plt


    class FSAug(object):
        def __init__(self):
            self.augment = Compose([
                AstypeToFloat(),
                # RandomHShift(scale=(0.,0.2),select=range(8)),
                # RandomRotate(angles=(-20., 20.), bound=True),
                ExpandBorder(select=range(8), mode='symmetric'),# symmetric
                # Resize(size=(336, 336), select=[ 2, 7]),
                AstypeToInt()
            ])

        def __call__(self, spct,attr_idx):
            return self.augment(spct,attr_idx)


    trans = FSAug()

    img_path = '/media/gserver/data/FashionAI/round2/train/Images/coat_length_labels/0b6b4a2146fc8616a19fcf2026d61d50.jpg'
    img = cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB)
    img_trans,_ = trans(img,5)
    # img_trans2,_ = trans(img,6)
    print img_trans.max(), img_trans.min()
    print img_trans.dtype

    plt.figure()
    plt.subplot(221)
    plt.imshow(img)

    plt.subplot(222)
    plt.imshow(img_trans)

    # plt.subplot(223)
    # plt.imshow(img_trans2)
    # plt.imshow(img_trans2)
    plt.show()
'''
