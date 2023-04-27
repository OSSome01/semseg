import torch
import numpy as np
from tqdm import tqdm
import sys
import pandas as pd
import multiprocessing
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import glob
import os
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from datasets import load_metric
import torch.nn as nn


class RUGD(Dataset):
    def __init__(self, root_dir, feature_extractor, color_map, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.feature_extractor = feature_extractor
        self.color_map = color_map

        # Load the list of image and label filenames

        self.image_filenames = sorted(glob.glob(os.path.join(self.root_dir, '*.png')))
        self.label_filenames = sorted(glob.glob(os.path.join(self.root_dir+'_labels', '*.png')))

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
        label_size = np.array(label).shape
        label_classes = np.zeros((label_size[0]*label_size[1]), dtype=np.uint8)
        label = np.array(label)
        label_pixels = label.reshape(-1,3)
        for i in range(label_pixels.shape[0]):
          rgb = tuple(label_pixels[i])
          if rgb in self.color_map:
            label_classes[i] = self.color_map[rgb]
        label = label_classes.reshape(label_size[0],label_size[1])    
        
        
        encoded_inputs = self.feature_extractor(image, Image.fromarray(label), return_tensors="pt")

        for k,v in encoded_inputs.items():
          encoded_inputs[k].squeeze_() # remove batch dimension

        return encoded_inputs


feature_extractor = SegformerFeatureExtractor(reduce_labels=True)

color_map = pd.read_csv('/home/ossome/semantic_segmentation/semantic-segmentation/data/RUGD_sample-data/RUGD_annotation-colormap.txt', sep=" ", header=None)
color_map.columns = ["label_idx", "label", "R", "G", "B"]
color_map.head()
label2id = {label: id for id, label in enumerate(color_map.label)}
id2label = {id: label for id, label in enumerate(color_map.label)}
id2color = {id: [R,G,B] for id, (R,G,B) in enumerate(zip(color_map.R, color_map.G, color_map.B))}
color2id = {(R,G,B): id for id, (R,G,B) in enumerate(zip(color_map.R, color_map.G, color_map.B))}

del id2color[0]
id2color = {id-1: color for id, color in id2color.items()}
del id2label[0]
label2id = {label: id-1 for id, label in id2label.items()}
id2label = {id-1: label for id, label in id2label.items()}

test_dir = '/home/ossome/semantic_segmentation/sem_seg/RUGD_small/test'
test_dataset = RUGD(root_dir=test_dir, feature_extractor=feature_extractor, color_map=color2id)
testloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Define the device to be used for inference
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0", num_labels=24)
model.load_state_dict(torch.load('segformer_best_model_weights.pth', map_location=device))
model.to(device)
model.eval()

test_metric = load_metric("mean_iou")
# metric = load_metric("f1")

for idx, batch in enumerate((testloader)):
    pixel_values = batch["pixel_values"].to(device)
    labels = batch["labels"].to(device)
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values, labels=labels)
        test_loss, logits = outputs.loss, outputs.logits

        upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        predicted = upsampled_logits.argmax(dim=1)
        
        test_metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=labels.detach().cpu().numpy())
        # metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=labels.detach().cpu().numpy())  

test_metrics = test_metric.compute(num_labels=len(id2label), ignore_index=255, reduce_labels=False)
# metrics = metric.compute(num_labels=len(id2label), ignore_index=255, reduce_labels=False)
print("Val Loss:", test_loss.item())
print("Val Mean_iou:", test_metrics["mean_iou"])
print("Val Mean accuracy:", test_metrics["mean_accuracy"])
# print("F1", metrics['f1'])
