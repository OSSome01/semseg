# from torch._C import int32
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob
class RUGD(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform


        # Define the classes and color map
        self.classes = ['void', 'dirt', 'sand', 'grass', 'tree', 'pole', 'water', 'sky', 'vehicle', 'container/generic-object', 'asphalt', 'gravel', 'building', 'mulch', 'rock-bed',
                        'log', 'bicycle', 'person', 'fence', 'bush', 'sign', 'rock', 'bridge', 'concrete', 'picnic-table']
        self.color_map = {
            (0, 0, 0): 0,   
            (108, 64, 20): 1,    
            (255, 229, 204): 2,   
            (0, 102, 0): 3, #'grass'      
            (0, 255, 0): 4, #'tree'
            (0, 153, 153): 5,
            (0, 128, 255): 6,
            (0, 0, 255): 7, #'sky'
            (255, 255, 0): 8,
            (255, 0, 127): 9,
            (64, 64, 64): 10,
            (255, 128, 0): 11, #'gravel'
            (255, 0, 0): 12,
            (153, 76, 0): 13, #'mulch'
            (102, 102, 0): 14,
            (102, 0, 0): 15,
            (0, 255, 128): 16,
            (204, 153, 255): 17,
            (102, 0, 204): 18,
            (255, 153, 204): 19,
            (0, 102, 102): 20,
            (153, 204, 255): 21,
            (102, 255, 255): 22,
            (101, 101, 11): 23,
            (114, 85, 47): 24,
        }

        # Load the list of image and label filename
        self.image_filenames = sorted(glob.glob(os.path.join(self.root_dir, self.split, '*.png')))
        self.label_filenames = sorted(glob.glob(os.path.join(self.root_dir, self.split+'_labels', '*.png')))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Load the image
        image = Image.open(self.image_filenames[idx]).convert('RGB')
        # Apply the transforms to the image
        if self.transform is not None:
            image = self.transform(image)

        # Load the label
        label = Image.open(self.label_filenames[idx]).convert('RGB')
        label = label.resize((256,256), resample=Image.NEAREST)
        label_classes = np.zeros((256*256), dtype=np.uint8)
        label = np.array(label)
        label_pixels = label.reshape(-1,3)
        for i in range(label_pixels.shape[0]):
          rgb = tuple(label_pixels[i])
          if rgb in self.color_map:
            label_classes[i] = self.color_map[rgb]
        label = label_classes.reshape(256,256)        

        return image, label
