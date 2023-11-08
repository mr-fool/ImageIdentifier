import tkinter as tk
from tkinter import filedialog
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import torch
from torchvision import models

# Create a Tkinter application
app = tk.Tk()
app.title("Image Classification")

# Function to classify the image
def classify_image():
    # Get the image location from the user
    image_location = entry_image_location.get()

    # Load the original image from the specified path
    InputImg = Image.open(image_location)

    # Apply the defined transformations to the original image
    InputImg_t = transform(InputImg)

    # Add a batch dimension to the image tensor
    InputImg_bt = torch.unsqueeze(InputImg_t, 0)

    # Pass the batched image tensor through the ResNet-152 model to obtain output
    output = resnet(InputImg_bt)

    # Sort the output tensor to get the top predicted class indices in descending order
    _, predictedLabels = torch.sort(output, descending=True)

    # Calculate the percentages by applying a sigmoid function to the output tensor
    percentage = torch.sigmoid(output)[0] * 100

    # Get the user-specified number of labels to display
    num_labels_to_display = int(entry_num_labels.get())

    # Display the top 'num_labels_to_display' predicted classes and their associated probabilities
    results_text.delete(1.0, tk.END)  # Clear the previous results
    top_predictions = [(ImageNetclasses[index], percentage[index].item()) for index in predictedLabels[0][:num_labels_to_display]]
    for label, prob in top_predictions:
        results_text.insert(tk.END, f"{label}: {prob:.2f}%\n")

# Create a series of image transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load a pre-trained ResNet-152 model and set it to evaluation mode
resnet = models.resnet152(pretrained=True)
resnet.eval()

# Open a file containing ImageNet class labels
with open('imagenet1000Classes.txt') as classesfile:
    # Read and store the class labels from the file
    ImageNetclasses = [line.strip() for line in classesfile.readlines()]

# Create and pack widgets
label_image_location = tk.Label(app, text="Image Location:")
label_image_location.pack()

entry_image_location = tk.Entry(app)
entry_image_location.pack()

label_num_labels = tk.Label(app, text="Number of Labels to Display:")
label_num_labels.pack()

entry_num_labels = tk.Entry(app)
entry_num_labels.pack()
entry_num_labels.insert(0, "5")  # Default value

button_browse = tk.Button(app, text="Browse", command=lambda: entry_image_location.insert(tk.END, filedialog.askopenfilename()))
button_browse.pack()

button_classify = tk.Button(app, text="Classify", command=classify_image)
button_classify.pack()

results_text = tk.Text(app, height=10, width=50)
results_text.pack()

app.mainloop()
