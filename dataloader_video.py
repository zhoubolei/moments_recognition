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

class VideoDataset(Dataset):
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
