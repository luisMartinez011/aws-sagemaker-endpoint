import torch
import os
import json
import logging
import time
import torch.nn as nn
import torch.nn.functional as F

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = 'model_deteccion_imagenes.pth'

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(12288, 84)
        self.fc2 = nn.Linear(84, 50)
        self.fc3 = nn.Linear(50, 2)

    def forward(self, x):
        x = x.view(-1, 12288)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def model_fn(model_dir):
    """Carga el modelo desde el directorio especificado"""
    logger.info(f"Loading model from {model_dir}")
    start_time = time.time()

    try:
        model = SimpleNet()
        model_path = os.path.join(model_dir, MODEL_NAME)

        with open(model_path, 'rb') as f:
            model.load_state_dict(torch.load(f, map_location=torch.device('cpu')))

        model.eval()
        load_time = time.time() - start_time
        logger.info(f"Model loaded successfully in {load_time:.2f}s")

        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


def input_fn(request_body, request_content_type):
    """Procesa la entrada del request"""
    logger.info(f"Processing input with content type: {request_content_type}")

    if request_content_type == 'application/json':
        try:
            data = json.loads(request_body)
            input_tensor = torch.tensor(data)
            logger.info(f"Input tensor shape: {input_tensor.shape}")
            return input_tensor
        except Exception as e:
            logger.error(f"Error processing input: {str(e)}")
            raise ValueError(f"Error parsing JSON: {str(e)}")
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    """Realiza la predicción"""
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    try:
        model.to(device)
        input_data = input_data.to(device)

        with torch.no_grad():
            predictions = model(input_data)

        inference_time = time.time() - start_time
        predicted_class = predictions.cpu().numpy().argmax()

        logger.info(f"Inference completed in {inference_time:.4f}s")
        logger.info(f"Predicted class: {predicted_class}")
        logger.info(f"Prediction scores: {predictions.cpu().numpy().tolist()}")

        return predictions

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise


def output_fn(prediction, response_content_type):
    """Formatea la salida de la predicción"""
    if response_content_type == 'application/json':
        result = {
            'predicted_class': prediction.cpu().numpy().argmax().tolist(),
            'scores': prediction.cpu().numpy().tolist(),
            'timestamp': time.time()
        }
        logger.info(f"Returning prediction: {result}")
        return json.dumps(result)
    else:
        raise ValueError(f"Unsupported content type: {response_content_type}")
