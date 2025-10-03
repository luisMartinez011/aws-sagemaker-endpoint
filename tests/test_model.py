import torch
import os
import json
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile
from torchvision import transforms


MODEL_NAME = 'model_deteccion_imagenes.pth'

class SimpleNet(nn.Module):

    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(12288, 84)
        self.fc2 = nn.Linear(84, 50)
        self.fc3 = nn.Linear(50,2)

    def forward(self, x):
        x = x.view(-1, 12288)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def model_fn(model_dir):
    model = SimpleNet()
    with open(os.path.join(model_dir, MODEL_NAME), 'rb') as f:
        model.load_state_dict(torch.load(f))
    return model


def test_simple_model():
    model = model_fn("model_deteccion_imagenes")


    img_transforms = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225] )
        ])
    img = Image.open("pescado.jpg")
    img = img_transforms(img)
    input_batch = torch.unsqueeze(img, 0)
    input_batch = input_batch

    # print(input_batch)

    output_batch = model(input_batch)
    output_tensor = output_batch[0]
    output_array = output_tensor.detach().cpu().numpy().argmax()
    print(output_array)
    return output_array

output_array = test_simple_model()

