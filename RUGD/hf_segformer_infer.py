import torch
import torchvision.transforms as transforms
from PIL import Image
from models import UNet, AttnUNet
import numpy as np
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0", num_labels=24)
model.load_state_dict(torch.load('segformer_best_model_weights.pth', map_location=device))
model.to(device)
feature_extractor = SegformerFeatureExtractor(reduce_labels=True)
model.eval()

color_map = pd.read_csv('RUGD_annotation-colormap.txt', sep=" ", header=None)
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
    
image = Image.open('/home/ossome/semantic_segmentation/sem_seg/creek_04071.png')
encoding = feature_extractor(image, return_tensors="pt")
pixel_values = encoding.pixel_values.to(device)
outputs = model(pixel_values=pixel_values)
logits = outputs.logits.cpu()

upsampled_logits = nn.functional.interpolate(logits,
                size=image.size[::-1], 
                mode='bilinear',
                align_corners=False)
seg = upsampled_logits.argmax(dim=1)[0]
color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
for label, color in id2color.items():
    color_seg[seg == label, :] = color
img = np.array(image) * 0.5 + color_seg * 0.5
img = img.astype(np.uint8)

plt.figure(figsize=(15, 10))
plt.imshow(color_seg)
plt.show()