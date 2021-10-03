import os
from collections import Counter
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from deepcell.utils.plot_utils import create_rgb_image
from deepcell.utils.plot_utils import make_outline_overlay
from dataExtra import *
### Load the test split


npz_dir = ''
test_dict = np.load(os.path.join(npz_dir, 'tissuenet_v1.0_train.npz'))

test_X, test_y = test_dict['X'], test_dict['y']

tissue_list, platform_list = test_dict['tissue_list'], test_dict['platform_list']
# print(np.array(test_X).shape,np.array(test_y).shape)
# temp = Counter(np.array(test_X[1]).flatten())
# print(temp)
# plt.figure()
# plt.bar(range(len(temp)),temp)
# plt.show()
# print(np.array(tissue_list).shape,np.array(platform_list).shape)
for i in test_dict.keys():
    print(i)
valid_tissues = np.unique(tissue_list)
print(valid_tissues)
valid_platforms = np.unique(platform_list)
selected_tissue = 'breast'
selected_platform = 'all'
if selected_tissue == 'all':
    tissue_idx = np.repeat(True, len(tissue_list))
else:
    tissue_idx = tissue_list == selected_tissue

if selected_platform == 'all':
    platform_idx = np.repeat(True, len(platform_list))
else:
    platform_idx = platform_list == selected_platform

combined_idx = tissue_idx * platform_idx

if sum(combined_idx) == 0:
    raise ValueError("The specified combination of image platform and tissue type does not exist")

selected_X, selected_y = test_X[combined_idx, ...], test_y[combined_idx, ...]

rgb_images = create_rgb_image(selected_X, channel_colors=['red', 'green', 'blue'])
overlay_data_cell = make_outline_overlay(rgb_data=rgb_images, predictions=selected_y[..., 0:1])
overlay_data_nuc = make_outline_overlay(rgb_data=rgb_images, predictions=selected_y[..., 1:2])
# plot_idx = 136
# plot_idx = np.random.randint(0, selected_X.shape[0])
# plt.figure()
# io.imshow(selected_X[plot_idx,:,:,0:1])
# plt.figure()
# io.imshow(selected_y[plot_idx,:,:,0:1])
# plt.figure()
# io.imshow(selected_X[plot_idx,:,:,1:2])
# plt.figure()
# io.imshow(selected_y[plot_idx,:,:,1:2])
# plt.figure()
# io.imshow(overlay_data_nuc[plot_idx])
# plt.figure()
# io.imshow(overlay_data_nuc[plot_idx])
# plt.show()