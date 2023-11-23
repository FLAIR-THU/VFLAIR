import torch
import numpy as np
from PIL import Image
import sys, os
import matplotlib.pyplot as plt

attack_name = 'ressfl'
# attack_name = 'grn'
file_name = ['None', 'GaussianDP', 'LaplaceDP', 'DistanceCorrelation', 'MID_Passive']

image = [torch.load(f'./exp_result/{attack_name}/{file_name[i]}.pkl',map_location='cpu') for i in range(len(file_name))]

print(image[0].shape)
# image = [item.reshape(-1,14,28) for item in image]
image = [item.reshape(-1,3,16,32) for item in image]
print(image[0].shape)

image = [item.numpy() for item in image]

# image_index_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
# for _i, _images in enumerate(image):
#     save_dir = f'./exp_result/{attack_name}/figure/{file_name[_i]}/'
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#     for _index in image_index_list:
#         save_path = f'{save_dir}{_index}.png'
#         img=Image.fromarray(np.uint8(_images[_index]*255))
#         img.save(save_path,quality=95)

image_index_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
fig, axes = plt.subplots(5, 16, figsize=(12, 3))  # Change 3 to the number of images
# image_index_list = [0,1,6,12,15]
# fig, axes = plt.subplots(5, 5, figsize=(12, 7.5))  # Change 3 to the number of images
# image_index = 0
for _i, _images in enumerate(image):
    for _j, _index in enumerate(image_index_list):
        # print(axes[_i])
        # print(axes[_i][_j])
        # Display images in subplots
        # axes[_i][_j].imshow(np.uint8(_images[_index]*255), cmap='gray')
        axes[_i][_j].imshow(np.uint8(_images[_index]*255).transpose(1, 2, 0))
        axes[_i][_j].axis('off')  # Turn off axis labels and ticks
        # axes[image_index].set_title('Image 1')
        # image_index += 1

# Adjust layout for better spacing
plt.tight_layout()
# Show the figure
plt.show()
plt.savefig(f'./exp_result/{attack_name}/figure/total.png', dpi=200, bbox_inches='tight')
