from PIL import Image

#Your original image
InputImg = Image.open("/home/hetzer/Pictures/valley.jpg")

import matplotlib.pyplot as plt

#plt.imshow(InputImg)

from torchvision import transforms
transform = transforms.Compose([
     transforms.Resize(256),
     transforms.CenterCrop (224),
     transforms.ToTensor(),
     transforms.Normalize(
         mean=[0.485, 0.456, 0.406],
         std=[0.229, 0.224, 0.225]
     )])

InputImg_t = transform(InputImg)

print(InputImg_t.shape)

import torch
InputImg_bt = torch.unsqueeze(InputImg_t,0)
print(InputImg_bt.shape)

from torchvision import models
dir(models)

resnet = models.resnet152(pretrained=True)
resnet.eval()

output = resnet(InputImg_bt)

#alexnet = models.alexnet(pretrained=True)
#alexnet.eval()

#output = alexnet(InputImg_bt)

with open('/home/hetzer/Downloads/imagenet1000Classes.txt') as classesfile:
  ImageNetclasses = [line.strip() for line in classesfile.readlines()]

_, predictedLabels = torch.sort(output, descending = True)
percentage = torch.sigmoid (output) [0] * 100
print ( [(ImageNetclasses[index], percentage[index].item() ) for index in predictedLabels[0][:5] ] )