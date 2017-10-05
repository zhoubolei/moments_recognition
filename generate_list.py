import os
import random

setID = 2

if setID == 1:
    split_type = 'train'
    split_name = '0803'
    flag_categories = 0
    raw_list = '/data/vision/oliva/atomic-videos/data_lists/big_train_list_0803.txt'
    root_folder = '/data/vision/oliva/scratch/moments_collage_0715'
    threshold = 300
elif setID == 2:
    split_type = 'val'
    split_name = '0803'
    flag_categories = 1
    raw_list = '/data/vision/oliva/atomic-videos/data_lists/big_val_list_0803.txt'
    threshold = 0
    root_folder = '/data/vision/oliva/scratch/moments_collage_0715'

with open(raw_list) as f:
    lines = f.readlines()

num_valid = 0
num_total = 0

dict_categories = {}
for line in lines:
    items = line.split()
    category, videoname = items[0].split('/')
    if category not in dict_categories:
        dict_categories[category] = []
    filename = os.path.join(root_folder,  items[0], 'images_0.jpg')
    if os.path.isfile(filename):
        num_valid += 1
        dict_categories[category].append(items[0])

output_histogram = []
categories = dict_categories.keys()
categories = sorted(categories)
for i, category in enumerate(categories):
    files_category = dict_categories[category]
    output_histogram.append((category, len(files_category)))

# only include the categories with enough samples
categories = [item[0] for item in output_histogram if item[1]>=threshold]
output_histogram = [item for item in output_histogram if item[1]>=threshold]

if flag_categories == 0:
    categories = sorted(categories)
    with open('split/categories_'+split_name + '.txt','w') as f:
        f.write('\n'.join(categories))
else:
    with open('split/categories_' + split_name + '.txt') as f:
        lines = f.readlines()
    categories = [item.rstrip() for item in lines]

if split_type == 'train':
    with open('split/hist_' + split_name + '.txt','w') as f:
        f.write('\n'.join(['%s %d'%(item[0], item[1]) for item in output_histogram]))

output = []
for i, category in enumerate(categories):
    files_category = dict_categories[category]
    output_categories = ['%s %d'%(item, i) for item in files_category]
    output += output_categories

random.shuffle(output)
with open('split/%s_%s.txt'%(split_type, split_name), 'w') as f:
    f.write('\n'.join(output))


