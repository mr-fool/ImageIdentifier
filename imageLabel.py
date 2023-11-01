# Import the Image class from the Pillow (PIL) library
from PIL import Image

# Load the original image from the specified path
InputImg = Image.open("/home/hetzer/Pictures/valley.jpg")

# Import the matplotlib library for image visualization
import matplotlib.pyplot as plt

# Create a series of image transformations using torchvision.transforms
# These transformations will resize the image, center crop it, convert it to a tensor,
# and normalize it with specified mean and standard deviation values.
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Apply the defined transformations to the original image
InputImg_t = transform(InputImg)

# Print the shape of the transformed image tensor
print(InputImg_t.shape)

# Import the torch library for deep learning operations
import torch

# Add a batch dimension to the image tensor (expected by the model)
InputImg_bt = torch.unsqueeze(InputImg_t, 0)

# Print the shape of the batched image tensor
print(InputImg_bt.shape)

# Import the torchvision models module to access pre-trained models
from torchvision import models

# List the available models in torchvision
dir(models)

# Load a pre-trained ResNet-152 model and set it to evaluation mode
resnet = models.resnet152(pretrained=True)
resnet.eval()

# Pass the batched image tensor through the ResNet-152 model to obtain output
output = resnet(InputImg_bt)

# Uncomment the following lines to use a pre-trained AlexNet model instead
# alexnet = models.alexnet(pretrained=True)
# alexnet.eval()
# output = alexnet(InputImg_bt)

# Open a file containing ImageNet class labels
with open('/home/hetzer/Downloads/imagenet1000Classes.txt') as classesfile:
    # Read and store the class labels from the file
    ImageNetclasses = [line.strip() for line in classesfile.readlines()]

# Sort the output tensor to get the top predicted class indices in descending order
_, predictedLabels = torch.sort(output, descending=True)

# Calculate the percentages by applying a sigmoid function to the output tensor
percentage = torch.sigmoid(output)[0] * 100

# Print the top 5 predicted classes and their associated probabilities
print([(ImageNetclasses[index], percentage[index].item()) for index in predictedLabels[0][:5]])
