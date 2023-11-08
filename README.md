# Image Classification with Pre-trained Deep Learning Models

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)

This repository contains a Python script for performing image classification using pre-trained deep learning models. The code leverages the power of deep neural networks to predict the top classes for an input image, along with their associated probabilities. It uses PyTorch and torchvision to achieve this.

## Purpose 
This software leverages artificial intelligence to automatically generate descriptive labels for images. These generated labels can then be seamlessly integrated into website's alt tags, significantly improving accessibility for users with visual impairments. By making web content more inclusive, it ensures that everyone can access and understand the images displayed on the site.

## Features

- Loads an image and preprocesses it for model input.
- Utilizes a pre-trained deep learning model (ResNet-152 or AlexNet) for image classification.
- Displays the top predicted classes and their probabilities based on the ImageNet dataset.

## Usage

1. Clone this repository to your local machine.
2. Install the necessary dependencies (PyTorch, torchvision, Pillow, matplotlib).
3. Replace the sample image with your own image in the script.
4. Run the script to obtain image classification results.
5. Use the classification as an alt-tag for websites to enhance accessibility.

## Prerequisites

- Python 3.10
- PyTorch
- torchvision
- Pillow (PIL)
- matplotlib
- Tkinter

## Models

- ResNet-152 (pre-trained on ImageNet)
- AlexNet (pre-trained on ImageNet) (deprecated)

## Example

Here's an example of how to use the code:

```bash
python3 imageLabel.py
```

## GUI
![gui](https://github.com/mr-fool/ImageIdentifier/assets/6241984/0f78da84-858e-4643-b310-15eaad9175a7)



