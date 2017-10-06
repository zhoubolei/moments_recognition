# implement the data loader for video classification

import torch
import os
from PIL import Image

import torch
from torch.utils.data.dataset import Dataset
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

class MomentsDataset(Dataset):
    def __init__(self, root_dataset, file_videos, transform=None, size_frame=256):
        self.root_dataset = root_dataset
        self.transform = transform
        self.size_frame = size_frame
        with open(file_videos) as f:
            lines = f.readlines()
        self.imgfiles = [line.split()[0] for line in lines]
        y = np.array([int(line.split()[1]) for line in lines])
        self.y = torch.from_numpy(y)

    def __getitem__(self, index):
        # TD: use a better way to handle the image name (maybe input file in the format: xxx/img1.jpg classIDX)
        path_img = os.path.join(self.root_dataset, self.imgfiles[index],'images_0.jpg')
        img = Image.open(path_img)
        img_frame = self.cropframe(img)
        img_frame = img_frame.convert('RGB')
        if self.transform is not None:
            img_frame = self.transform(img_frame)
        label = self.y[index]

        return img_frame, label

    def __len__(self):
        return len(self.imgfiles)

    def cropframe(self, img):
        # random crop one frame
        num_frames = img.size[0] / self.size_frame
        idx_random = np.random.randint(num_frames)
        return img.crop((idx_random*self.size_frame, 0, (idx_random+1)*self.size_frame, img.size[1]))

class KineticsFrameDataset(Dataset):
    # load the frames of kinetics
    def __init__(self, root_dataset, file_frames, transform=None, size_frame=256):
        self.root_dataset = root_dataset
        self.transform = transform
        self.size_frame = size_frame
        with open(file_frames) as f:
            lines = f.readlines()
        self.videofolders = [line.split()[0] for line in lines]
        y = np.array([int(line.split()[1]) for line in lines])
        self.y = torch.from_numpy(y)
        self.num_videos = len(self.videofolders)

    def __getitem__(self, index):
        # todo: input a imglist file with the number of frames predefined
        randIDX = np.random.randint(150) + 1
        file_frame = os.path.join(self.root_dataset, self.videofolders[index], '%06d.jpg'%randIDX)
        while not os.path.exists(file_frame):
            # if the video frame is not readable
            #print 'not existed', file_frame
            index = np.random.randint(self.num_videos)
            randIDX = np.random.randint(250) + 1
            file_frame = os.path.join(self.root_dataset, self.videofolders[index], '%06d.jpg'%randIDX)
        img = Image.open(file_frame)
        img = img.convert('RGB')
        label = self.y[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.videofolders)
