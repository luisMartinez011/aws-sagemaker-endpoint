import os
import sys
import boto3
import json
import torch
from PIL import Image
from torchvision import transforms
import io
import logging

# Agregar el path del config
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from config.config_loader import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EndpointInvoker:
    def __init__(self, endpoint_name=None):
        self.config = config
        self.s3_client = boto3.client('s3', region_name=self.config.AWS_REGION)
        self.runtime_client = boto3.client('sagemaker-runtime', region_name=self.config.AWS_REGION)
        self.endpoint_name = endpoint_name or self.config.ENDPOINT_NAME

    def get_image_from_s3(self, bucket_name, filename):
        """Descarga imagen desde S3"""
        logger.info(f"Downloading image: s3://{bucket_name}/{filename}")

        try:
            response = self.s3_client.get_object(Bucket=bucket_name, Key=filename)
            image_data = response['Body'].read()
            logger.info("Image downloaded successfully")
            return image_data
        except Exception as e:
            logger.error(f"Error downloading image: {str(e)}")
            raise

    def get_image_from_local(self, filepath):
        """Lee imagen desde archivo local"""
        logger.info(f"Reading local image: {filepath}")

        try:
            with open(filepath, 'rb') as f:
                image_data = f.read()
            logger.info("Image read successfully")
            return image_data
        except Exception as e:
            logger.error(f"Error reading image: {str(e)}")
            raise

    def preprocess_image(self, image_data):
        """Preprocesa la imagen para el modelo"""
        logger.info("Preprocessing image...")

        img_transforms = transforms.Compose([
            transforms.Resize((self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        img = Image.open(io.BytesIO(image_data))
        img = img_transforms(img)
        input_image = torch.unsqueeze(img, 0).numpy().tolist()

        logger.info(f"Image preprocessed - shape: {len(input_image)}x{len(input_image[0])}")
        return input_image

    def invoke_endpoint(self, input_data):
        """Invoca el endpoint de SageMaker"""
        logger.info(f"Invoking endpoint: {self.endpoint_name}")

        try:
            payload = json.dumps(input_data)

            response = self.runtime_client.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType='application/json',
                Body=payload
            )

            result = json.loads(response['Body'].read().decode())
            logger.info(f"Endpoint response: {result}")
            return result

        except self.runtime_client.exceptions.ModelError as e:
            logger.error(f"Model error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error invoking endpoint: {str(e)}")
            raise

    def postprocess_output(self, prediction_result):
        """Procesa la salida del modelo"""
        logger.info("Processing model output...")

        # Si el resultado tiene la estructura nueva con metadata
        if isinstance(prediction_result, dict):
            predicted_class = prediction_result.get('predicted_class', prediction_result)
        else:
            predicted_class = prediction_result

        label = self.config.LABELS[predicted_class]

        result = {
            'predicted_class': predicted_class,
            'predicted_label': label,
            'raw_output': prediction_result
        }

        logger.info(f"Prediction: {label} (class {predicted_class})")
        return result

    def predict_from_s3(self, bucket_name, filename):
        """Pipeline completo de predicción desde S3"""
        logger.info("="*50)
        logger.info("Starting Prediction Pipeline (S3)")
        logger.info("="*50)

        try:
            # 1. Descargar imagen
            image_data = self.get_image_from_s3(bucket_name, filename)

            # 2. Preprocesar
            input_data = self.preprocess_image(image_data)

            # 3. Invocar endpoint
            raw_prediction = self.invoke_endpoint(input_data)

            # 4. Postprocesar
            result = self.postprocess_output(raw_prediction)

            logger.info("="*50)
            logger.info("Prediction completed successfully!")
            logger.info("="*50)

            return result

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise

    def predict_from_local(self, filepath):
        """Pipeline completo de predicción desde archivo local"""
        logger.info("="*50)
        logger.info("Starting Prediction Pipeline (Local)")
        logger.info("="*50)

        try:
            # 1. Leer imagen
            image_data = self.get_image_from_local(filepath)

            # 2. Preprocesar
            input_data = self.preprocess_image(image_data)

            # 3. Invocar endpoint
            raw_prediction = self.invoke_endpoint(input_data)

            # 4. Postprocesar
            result = self.postprocess_output(raw_prediction)

            logger.info("="*50)
            logger.info("Prediction completed successfully!")
            logger.info("="*50)

            return result

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise


if __name__ == "__main__":
    # Ejemplo de uso
    invoker = EndpointInvoker()

    # Opción 1: Desde S3
    result = invoker.predict_from_s3(
        bucket_name=config.BUCKET_NAME,
        filename='gato.jpg'
    )

    # Opción 2: Desde archivo local
    # result = invoker.predict_from_local('pescado.jpg')

    print(json.dumps(result, indent=2))
