import torch
import os
import json
import torch.nn as nn
import torch.nn.functional as F

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


def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        data = json.loads(request_body)
        input_tensor = torch.tensor(data)
        return input_tensor
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        input_data = input_data.to(device)
        predictions = model(input_data)
    return predictions


def output_fn(prediction, response_content_type):
    if response_content_type == 'application/json':
        return prediction.cpu().numpy().argmax().tolist()

    else:
        raise ValueError(f"Unsupported content type: {response_content_type}")
