# Download an example image from the pytorch website
from PIL import Image
from torchvision import transforms
import torch
import models.swin_transformer as swin_transformer

filename = "data/dog.jpg"

input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

swin_model = swin_transformer.swin_model(pretrained=True)
with torch.no_grad():
    output = swin_model(input_batch)
print("SWIN EXECUTION SUCCESSFUL!")
# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
# print(output[0])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
# print(torch.nn.functional.softmax(output[0], dim=0))