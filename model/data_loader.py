import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
import cv2
import glob
import random



def get_data_direct(img_size, imgs_dir, texture_size = None, textures_dir = None,
                    format = '*.png',
                    mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
    imgs = glob.glob(os.path.join(imgs_dir, format))
    imgs = sorted(imgs)
    if textures_dir is not None:
        textures = glob.glob(os.path.join(textures_dir, format))
        textures = sorted(textures)

    batch_size = len(imgs) * len(textures) if textures_dir is not None else len(imgs)
    imgs_data = []
    textures_data = []
    if textures_dir is not None:
        assert texture_size is not None
        assert len(imgs) == len(textures)
        # for img_index in range(len(imgs)):
        for index in range(len(imgs)):
            # for texture_index in range(len(textures)):
            img_cur = Image.open(imgs[index])
            img_cur = img_cur.resize([img_size, img_size])
            # it could be rgba
            img_cur = (np.asarray(img_cur)[..., :3] / 255.0 - mean) / std
            imgs_data.append(img_cur)
            texture_cur = Image.open(textures[index])
            texture_cur = texture_cur.resize([texture_size, texture_size])
            # it could be rgba
            texture_cur = (np.asarray(texture_cur)[..., :3] / 255.0 - mean) / std
            textures_data.append(texture_cur)
    else:
        for img_index in range(len(imgs)):
            img_cur = Image.open(imgs[img_index])
            img_cur = img_cur.resize([img_size, img_size])
            # it could be rgba
            img_cur = (np.asarray(img_cur)[...,:3] / 255.0 - mean) / std
            imgs_data.append(img_cur)

    imgs_data = np.array(imgs_data).reshape([batch_size, img_size, img_size, 3])
    imgs_data = np.transpose(imgs_data, [0, 3, 1, 2])

    if textures_dir is not None:
        textures_data = np.array(textures_data).reshape([batch_size, texture_size, texture_size, 3])
        textures_data = np.transpose(textures_data, [0, 3, 1, 2])
    return imgs_data, textures_data



class texture_seg_dataset(object):
    def __init__(self, data_path, img_size, segmentation_regions, texture_size,
                        shuffle = True, use_same_from = True,
                        mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]): # from torch normalize
        self.shuffle = shuffle
        self.img_size = img_size
        # self.segmentation_regions = segmentation_regions
        self.segmentation_regions = random.randint(2, segmentation_regions) # generate collages with 2~seg_regions regions
        self.texture_size = texture_size
        self.folders = glob.glob(os.path.join(data_path, '*/')) # 返回一个list of string
        self.use_same_from = use_same_from
        self.mean = mean
        self.std = std
        # num_seg must be smaller than scene_num
        assert (len(self.folders) >= self.segmentation_regions)

    def generate_random_masks(self, points = None):
        # use batch_size = 1
        # return [size, size, segmentation_regions]
        batch_size = 1
        xs, ys = np.meshgrid(np.arange(0, self.img_size), np.arange(0, self.img_size))

        if points is None:
            n_points = [self.segmentation_regions]
            points = [np.random.randint(0, self.img_size, size=(n_points[i], 2)) for i in range(batch_size)]

        masks = []
        for b in range(batch_size):
            dists_b = [np.sqrt((xs - p[0])**2 + (ys - p[1])**2) for p in points[b]]
            voronoi = np.argmin(dists_b, axis=0)
            masks_b = np.zeros((self.img_size, self.img_size, self.segmentation_regions))
            for m in range(self.segmentation_regions):
                masks_b[:,:,m][voronoi == m] = 1
            masks.append(masks_b)
        return masks[0]


    def random_crop(self, image, crop_height, crop_width):
        if (crop_width <= image.shape[1]) and (crop_height <= image.shape[0]):
            x = np.random.randint(0, image.shape[1]-crop_width)
            y = np.random.randint(0, image.shape[0]-crop_height)
            return image[y:y+crop_height, x:x+crop_width, :]
        else:
            raise Exception('Crop shape exceeds image dimensions!')

    def get_data(self, format = '*.jpg', mode = 'train'):
        mask = self.generate_random_masks()
        choose_from = []
        img = np.zeros([self.img_size, self.img_size, 3])
        # mode = 'validation'
        if mode == 'train':
            sampled_folders = random.sample(self.folders[0:34], self.segmentation_regions)
        else:
            sampled_folders = random.sample(self.folders[35:], self.segmentation_regions)
        texture_mask = []
        for index, folder in enumerate(sampled_folders):
            files = glob.glob(os.path.join(folder, format))
            file_cur = random.choice(files)
            # print (file_cur)
            img_cur = Image.open(file_cur)
            img_cur = img_cur.resize([self.img_size, self.img_size])
            img_cur = (np.asarray(img_cur) / 255.0 - self.mean) / self.std
            img_cur = np.asarray(img_cur)
            img[mask[..., index] == 1] = img_cur[mask[..., index] == 1]
            if self.use_same_from:
                texture_cur = img_cur
            else:
                file_cur = random.choice(files)
                texture_cur = np.asarray(Image.open(file_cur))
            texture = self.random_crop(texture_cur, self.texture_size, self.texture_size)
            texture_mask.append({'mask': mask[...,index], 'texture':texture})
        return img, texture_mask


    def feed(self, batch_size = None, mode='train'):
        if batch_size is None:
            return self.get_data(mode=mode)
        else:
            img_texture_mask = []
            # add alls in one img iput
            # for _ in range(batch_size // self.segmentation_regions + 1):
            #     img, texture_mask = self.get_data()
            #     for index in range(self.segmentation_regions):
            #         patch = {}
            #         patch['img'] = img
            #         patch['texture'] = texture_mask[index]['texture']
            #         patch['mask'] = texture_mask[index]['mask']
            #         img_texture_mask.append(patch)

            # add each one separatly
            for _ in range(batch_size):
                img, texture_mask = self.get_data(mode=mode)
                # random choice one from cluster
                index = np.random.choice(self.segmentation_regions, 1)[0]
                patch = {}
                patch['img'] = img
                patch['texture'] = texture_mask[index]['texture']
                patch['mask'] = texture_mask[index]['mask']
                img_texture_mask.append(patch)

            img_texture_mask = img_texture_mask[:batch_size]
            if self.shuffle:
                random.shuffle(img_texture_mask)
                imgs = [item['img'] for item in img_texture_mask]
                textures = [item['texture'] for item in img_texture_mask]
                masks = [item['mask'] for item in img_texture_mask]
            imgs = np.array(imgs).reshape([batch_size, self.img_size, self.img_size, 3])
            textures = np.array(textures).reshape([batch_size, self.texture_size, self.texture_size, 3])
            masks = np.array(masks).reshape([batch_size, self.img_size, self.img_size, 1])
            imgs = np.transpose(imgs, [0, 3, 1, 2])
            textures = np.transpose(textures, [0, 3, 1, 2])
            masks = np.transpose(masks, [0, 3, 1, 2])
            return imgs, textures, masks

if __name__ == '__main__':
    data_set = texture_seg_dataset('./dataset/dtd/images', img_size=256, segmentation_regions=10, texture_size=64)
    imgs, textures, masks = data_set.feed(batch_size=1)
    print('img shape: ', imgs.shape, 'dtype:', imgs.dtype)
    print('texture shape: ', textures.shape, 'dtype:', textures.dtype)
    print('masks shape: ', masks.shape, 'dtype:', masks.dtype)
    # raise
    # img, texture_mask = data_set.get_data()
    # print(img.shape)
    # print(len(texture_mask))
    # img = cv2.cvtColor(np.uint8(img), cv2.COLOR_BGR2RGB)
    # cv2.imwrite('test_img.png', img)
    # cv2.imshow('img', img/255.0)
    # for i in range(3):
    #     texture_mask[i]['texture'] = cv2.cvtColor(np.uint8(texture_mask[i]['texture']) , cv2.COLOR_BGR2RGB)
    #     # cv2.imwrite('test_texture_%d.png'%(i), texture_mask[i]['texture']
    #     cv2.imshow('mask_%d'%(i), texture_mask[i]['mask'])
    #     cv2.imshow('texture_%d'%(i), texture_mask[i]['texture'])
    # cv2.waitKey(0)
