import torch
import torchvision.transforms as transforms
from PIL import Image
from final_project_attention import UNet
# from final_project import UNet
import numpy as np

# Define the device to be used for inference
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the transforms to be applied to the input image
image_file = 'attention_'
# train_image_name = 'frame000204-1581624095_650'
# image_name = 'frame000002-1581623790_549'
# image_name = 'frame000100-1581624662_749'
# image_name = 'frame000204-1581624095_650'
height = 256
width = 256
model_name = 'rellis_2.31298527373181_.pth'
input_image = Image.open('E:/Dataset/Rellis-3D/test/images/'+image_name+'.jpg').convert('RGB')
# input_image = Image.open('E:/Dataset/Rellis-3D/train/images/'+train_image_name+'.jpg').convert('RGB')
reference_image = Image.open('E:/Dataset/Rellis-3D/test/labels/' + image_name + '.png').convert('RGB')
# reference_image = Image.open('E:/Dataset/Rellis-3D/train/labels/' + train_image_name + '.png').convert('RGB')
transform = transforms.Compose([
    transforms.Resize((height, width)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
# Load the saved model weights
model = UNet(num_channels=3, num_classes=22)
model.load_state_dict(torch.load('E:/DL/Models/retrain/'+model_name, map_location=device))
# model.load_state_dict(torch.load('E:/DL/Models/New folder1/best_model_weights_22.pth', map_location=device))


# unet_dict = torch.load('E:/DL/Models/unet_carvana_scale0.5_epoch2.pth',map_location=device)
# class_weights = torch.load('E:/DL/Models/best_model_weights_final.pth', map_location=device)

# print((list(unet_dict.keys())))
# print((list(class_weights.keys())))

# unet_dict['outc.conv.weight'] = class_weights['outc.conv.weight']
# unet_dict['outc.conv.bias'] = class_weights['outc.conv.bias']
# model.load_state_dict(unet_dict)
# # with torch.no_grad():
# #     model.outc.conv.weight.copy_(class_weights['outc.conv.weight'])
# #     model.outc.conv.bias.copy_(class_weights['outc.conv.bias'])
model.to(device)
#image_name = 'frame000100-1581624662_749'
# image_name = 'frame000000-1581624652_750'
# Load the input image

# input_image = Image.open('E:/Dataset/Rellis-3D/test/download.jpeg').convert('RGB')

# input_image = Image.open('E:/DL/Final Project/semseg-master/test_img.jpg').convert('RGB')
input_tensor = transform(input_image).unsqueeze(0).to(device)

# Perform inference on the input image
model.eval()
with torch.no_grad():
    output_tensor = model(input_tensor)
    output_tensor = torch.softmax(output_tensor, dim=1)
    _, predicted = torch.max(output_tensor, dim=1)
    predicted_image = predicted[0].cpu().numpy()
color_map = {
            (0, 0, 0): 0,   
            (108, 64, 20): 1,       
            (0, 102, 0): 2,       
            (0, 255, 0): 3,
            (0, 153, 153): 4,
            (0, 128, 255): 5,
            (0, 0, 255): 6,
            (255, 255, 0): 7,
            (255, 0, 127): 8,
            (64, 64, 64): 9,
            (255, 0, 0): 10,
            (102, 0, 0): 11,
            (204, 153, 255): 12,
            (102, 0, 204): 13,
            (255, 153, 204): 14,
            (170, 170, 170): 15,
            (41, 121, 255): 16,
            (101, 31, 255): 17,
            (137, 149, 9): 18,
            (134, 255, 239): 19,
            (99, 66, 34): 20,
            (110, 22, 138): 21,
        }
# color_map = {
#             (0, 0, 0): 0,   
#             (108, 64, 20): 1,       
#             (0, 102, 0): 3,       
#             (0, 255, 0): 4,
#             (0, 153, 153): 5,
#             (0, 128, 255): 6,
#             (0, 0, 255): 7,
#             (255, 255, 0): 8,
#             (255, 0, 127): 9,
#             (64, 64, 64): 10,
#             (255, 128, 0): 11,
#             (255, 0, 0): 12,
#             (102, 0, 0): 15,
#             (204, 153, 255): 17,
#             (102, 0, 204): 18,
#             (101, 101, 11): 23,
#             (41, 121, 255): 27,
#             (101, 31, 255): 29,
#             (137, 149, 9): 30,
#             (134, 255, 239): 31,
#             (99, 66, 34): 33,
#             (110, 22, 138): 34,
#         }
inverted_dict = {v: k for k, v in color_map.items()}
print(inverted_dict)

# print('predicted_image', predicted_image)
predicted_image_pixels = predicted_image.reshape(height*width)
image = np.zeros(height*width*3)
for i in range(predicted_image_pixels.shape[0]):
    rgb = inverted_dict[predicted_image_pixels[i]]
    image[i*3:i*3 + 3] = rgb
predicted_image = image.reshape(height,width,3)
# Display the predicted output image
predicted_image = predicted_image.astype('uint8')
predicted_image = Image.fromarray(predicted_image)
predicted_image.save("E:/DL/Report Images/"+image_file+image_name+'.jpg')
reference_image.save("E:/DL/Report Images/"+image_file+image_name+'_mask.jpg')
predicted_image.show()
reference_image.show()
input_image.show()
