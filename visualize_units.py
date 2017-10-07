# the example script to generate the unit segmentation visualization using pyTorch
# Bolei Zhou

import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
import pdb
import numpy as np
from scipy.misc import imresize as imresize
import cv2
from PIL import Image
import torch.utils.data as data
import torchvision.models as models
import dataloader_video

# dataset setup
img_size = (224, 224) # input image size
batch_size = 64
num_workers = 6
segment_size = (100,100)
num_top = 10 # how many top activated images to extract
num_topunit_class = 3 # how many top class-specific unit to plot for each class
threshold_scale = 0.2 # the scale used to segment the feature map. Smaller the segmentation will be tighter.


# load the pre-trained weights
id_model = 2
if id_model == 1:
    model_name = 'resnet18_kinetics_fromscratch'
    file_category = 'kinetics/categories.txt'
    with open(file_category) as f:
        lines = f.readlines()
    categories = [item.rstrip() for item in lines]
    model = models.resnet18(num_classes=len(categories)) # moment class 328, kinetics class 400
    checkpoint = torch.load('model/resnet18_kinetics_fromscratch_best.pth.tar')
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].iteritems()}
    model.load_state_dict(state_dict)
    features_names = ['layer4']
elif id_model == 2:
    model_name = 'resnet18_moments_fromscratch'
    file_category = 'moments/categories_0803.txt'
    with open(file_category) as f:
        lines = f.readlines()
    categories = [item.rstrip() for item in lines]
    model = models.resnet18(num_classes=len(categories)) # moment class 328, kinetics class 400
    checkpoint = torch.load('model/resnet18_moments_fromscratch_best.pth.tar')
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].iteritems()}
    model.load_state_dict(state_dict)
    features_names = ['layer4']

model.eval()
model.cuda()

# get the softmax weight
params = list(model.parameters())
weight_softmax = np.squeeze(params[-2].data.cpu().numpy())

# image datasest to be processed
name_dataset = 'sun+imagenetval'
root_image = '/data/vision/torralba/gigaSUN/www/unit_annotation/data/images'
with open('/data/vision/torralba/gigaSUN/www/unit_annotation/data/images/imagelist.txt') as f:
    lines = f.readlines()
imglist = []
for line in lines:
    line = line.rstrip()
    imglist.append(root_image + '/' + line)

imglist = imglist[::2]


features_blobs = []
def hook_feature(module, input, output):
    # hook the feature extractor
    features_blobs.append(np.squeeze(output.data.cpu().numpy()))

for name in features_names:
    model._modules.get(name).register_forward_hook(hook_feature)

# image transformer
tf = trn.Compose([
        trn.Scale(img_size),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = dataloader_video.SimpleDataset(imglist, tf)
loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False)

# extract the max value activaiton for each image
imglist_results = []
maxfeatures = [None] * len(features_names)
num_batches = len(dataset) / batch_size
for batch_idx, (input, paths) in enumerate(loader):
    del features_blobs[:]
    print 'extracting feature: batch %d / %d' % (batch_idx+1, num_batches)
    input = input.cuda()
    input_var = V(input, volatile=True)
    logit = model.forward(input_var)
    imglist_results = imglist_results + list(paths)
    if maxfeatures[0] is None:
        # initialize the feature variable
        for i, feat_batch in enumerate(features_blobs):
            size_features = (len(dataset), feat_batch.shape[1])
            maxfeatures[i] = np.zeros(size_features)
    start_idx = batch_idx*batch_size
    end_idx = min((batch_idx+1)*batch_size, len(dataset))
    for i, feat_batch in enumerate(features_blobs):
        maxfeatures[i][start_idx:end_idx] = np.max(np.max(feat_batch,3),2)

# generate the top activated images
output_folder = 'result_visualization/%s' % model_name
if not os.path.exists(output_folder):
    os.makedirs(output_folder + '/image')

# output the html first
for layerID, layer in enumerate(features_names):
    file_html = os.path.join(output_folder, layer + '.html')
    with open(file_html, 'w') as f:
        num_units = maxfeatures[layerID].shape[1]
        lines_units = ['%s-unit%03d.jpg' % (layer, unitID) for unitID in range(num_units)]
        lines_units = ['unit%03d<br><img src="image/%s">'%(unitID, lines_units[unitID]) for unitID in range(num_units)]
        f.write('\n<br>'.join(lines_units))
        print(file_html)

# output the class specific unit to category
layer_lastconv = features_names[-1]
file_html = os.path.join(output_folder, 'class_specific_unit.html')
output_lines = []
for classID in range(len(categories)):
    line = '<h3>%s</h3>' % categories[classID]
    idx_units_sorted = np.argsort(np.squeeze(weight_softmax[classID]))[::-1]
    for sortID in range(num_topunit_class):
        unitID = idx_units_sorted[sortID]
        weight_unit = weight_softmax[classID][unitID]
        line += 'weight=%.3f %s<br>' % (weight_unit, lines_units[unitID])
    line = '<p>%s</p>' % line
    output_lines.append(line)

with open(file_html, 'w') as f:
    f.write('\n'.join(output_lines))

# generate the unit visualization
for layerID, layer in enumerate(features_names):
    num_units = maxfeatures[layerID].shape[1]
    imglist_sorted = []
    # load the top actiatied image list into one list
    for unitID in range(num_units):
        activations_unit = np.squeeze(maxfeatures[layerID][:, unitID])
        idx_sorted = np.argsort(activations_unit)[::-1]
        imglist_sorted += [imglist[item] for item in idx_sorted[:num_top]]

    # data loader for the top activated images
    loader_top = data.DataLoader(
        dataloader_video.SimpleDataset(imglist_sorted, tf),
        batch_size=num_top,
        num_workers=num_workers,
        shuffle=False)

    for unitID, (input, paths) in enumerate(loader_top):
        del features_blobs[:]
        print 'visualize unit %d / %d' % (unitID+1, num_units)
        input = input.cuda()
        input_var = V(input, volatile=True)
        logit = model.forward(input_var)
        feature_maps = features_blobs[layerID]

        images_input = input.cpu().numpy()
        max_value = 0
        output_unit = []
        for i in range(num_top):
            feature_map = feature_maps[i][unitID]
            if max_value == 0:
                max_value = np.max(feature_map)
            feature_map = feature_map / max_value
            mask = cv2.resize(feature_map, segment_size)
            mask[mask < threshold_scale] = 0.0 # binarize the mask
            mask[mask > threshold_scale] = 1.0

            img = cv2.imread(paths[i])
            img = cv2.resize(img, segment_size)
            img = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
            img_mask = np.multiply(img, mask[:,:, np.newaxis])
            img_mask = np.uint8(img_mask * 255)
            output_unit.append(img_mask)
            output_unit.append(np.uint8(np.ones((segment_size[0],4,3))*255))
        montage_unit = np.concatenate(output_unit, axis=1)
        cv2.imwrite(os.path.join(output_folder, 'image', '%s-unit%03d.jpg'%(layer, unitID)), montage_unit)

