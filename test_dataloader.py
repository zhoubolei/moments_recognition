from dataloader_video import VideoDataset
from torch.utils.data import DataLoader
from torchvision import transforms

root_dataset = '/data/vision/oliva/scratch/moments_collage_0715'
file_list = 'split/train_oct4.txt'
dataset_train = VideoDataset(root_dataset, file_list)

train_loader = DataLoader(dataset_train, batch_size=256, shuffle=True, num_workers=4)
