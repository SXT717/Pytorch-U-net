
from os import listdir
from os.path import splitext
from PIL import Image
import cv2
import numpy as np

class Data_loader():
    def __init__(self, imgs_dir, masks_dir, scale=1):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        #取出文件名
        self.file_names = [splitext(file)[0] for file in listdir(imgs_dir) if not file.startswith('.')]
        #总数量
        self.file_lens = len(self.file_names)
        #随机分开训练集和测试机
        self.file_names = np.random.choice(self.file_names, self.file_lens)
        #训练集数量
        self.train_file_lens = int(self.file_lens*0.9)
        self.train_file_names = self.file_names[:self.train_file_lens]
        #测试集数量
        self.test_file_lens = self.file_lens - self.train_file_lens
        self.test_file_names = self.file_names[self.train_file_lens:]
        print('file number = ', self.file_lens, 'train file number = ', self.train_file_lens, 'test file number = ', self.test_file_lens)
        self.train_index = 0
        self.test_index = 0

    def train_next(self):
        if self.train_index >= self.train_file_lens: raise ValueError('train_next if self.train_index >= self.train_file_lens is True')
        # mask的最大值是1，不需要归一化
        img = np.array(self.scale_img(Image.open(self.imgs_dir+self.train_file_names[self.train_index]+'.jpg'), self.scale), dtype=np.float32) / 255.0
        mask = np.array(self.scale_img(Image.open(self.masks_dir+self.train_file_names[self.train_index]+'_mask.gif'), self.scale), dtype=np.float32)
        mask = np.expand_dims(mask, axis=-1)
        #交换维度
        img = img.transpose((2, 0, 1))
        mask = mask.transpose((2, 0, 1))
        # print(img.shape)
        # print(mask.shape)
        #累加索引值
        self.train_index += 1
        return img, mask

    def test_next(self):
        if self.test_index >= self.test_file_lens: raise ValueError('test_next if self.test_index >= self.test_file_lens is True')
        # mask的最大值是1，不需要归一化
        img = np.array(self.scale_img(Image.open(self.imgs_dir+self.test_file_names[self.test_index]+'.jpg'), self.scale), dtype=np.float32) / 255.0
        mask = np.array(self.scale_img(Image.open(self.masks_dir+self.test_file_names[self.test_index]+'_mask.gif'), self.scale), dtype=np.float32)
        mask = np.expand_dims(mask, axis=-1)
        # 交换维度
        img = img.transpose((2, 0, 1))
        mask = mask.transpose((2, 0, 1))
        # print(img.shape)
        # print(mask.shape)
        # 累加索引值
        self.test_index += 1
        return img, mask

    def scale_img(self, img, scale):
        w, h = img.size
        return img.resize((int(scale * w), int(scale * h)))




