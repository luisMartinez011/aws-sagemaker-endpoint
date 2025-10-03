import os
import sys
import boto3
import tarfile
import sagemaker
from datetime import datetime
from sagemaker.pytorch.model import PyTorchModel

# Agregar el path del config
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from config.config_loader import config

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelDeployer:
    def __init__(self):
        self.config = config
        self.s3_client = boto3.client('s3', region_name=self.config.AWS_REGION)
        self.timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')

    def compress_model(self):
        """Comprime el modelo y el código en un tar.gz"""
        logger.info("Compressing model...")

        current_path = os.getcwd()
        model_folder = os.path.join(current_path, self.config.MODEL_NAME)
        code_folder = os.path.join(model_folder, 'code')
        model_pth = os.path.join(model_folder, f'{self.config.MODEL_NAME}.pth')

        # Nombre del archivo con versión y timestamp
        tar_filename = f'{self.config.MODEL_NAME}-v{self.config.MODEL_VERSION}-{self.timestamp}.tar.gz'

        if not os.path.exists(code_folder):
            raise FileNotFoundError(f"Code folder not found: {code_folder}")
        if not os.path.exists(model_pth):
            raise FileNotFoundError(f"Model file not found: {model_pth}")

        with tarfile.open(tar_filename, "w:gz") as tar:
            tar.add(code_folder, arcname='code')
            tar.add(model_pth, arcname=os.path.basename(model_pth))

        logger.info(f"Model compressed: {tar_filename}")
        return tar_filename

    def push_model_to_s3(self):
        """Sube el modelo comprimido a S3"""
        logger.info("Uploading model to S3...")

        filename = self.compress_model()
        s3_key = f'models/{self.config.MODEL_NAME}/v{self.config.MODEL_VERSION}/{filename}'

        try:
            self.s3_client.upload_file(
                Filename=filename,
                Bucket=self.config.BUCKET_NAME,
                Key=s3_key
            )

            model_s3_uri = f"s3://{self.config.BUCKET_NAME}/{s3_key}"
            logger.info(f"Model uploaded to: {model_s3_uri}")

            # Guardar metadatos
            self._save_model_metadata(s3_key, model_s3_uri)

            # Limpiar archivo local
            os.remove(filename)

            return model_s3_uri

        except Exception as e:
            logger.error(f"Error uploading model: {str(e)}")
            raise

    def _save_model_metadata(self, s3_key, model_uri):
        """Guarda metadatos del modelo en S3"""
        metadata = {
            'model_name': self.config.MODEL_NAME,
            'version': self.config.MODEL_VERSION,
            'timestamp': self.timestamp,
            'framework': 'pytorch',
            'framework_version': self.config.FRAMEWORK_VERSION,
            's3_uri': model_uri
        }

        import json
        metadata_key = f'models/{self.config.MODEL_NAME}/v{self.config.MODEL_VERSION}/metadata.json'

        self.s3_client.put_object(
            Bucket=self.config.BUCKET_NAME,
            Key=metadata_key,
            Body=json.dumps(metadata, indent=2)
        )
        logger.info(f"Metadata saved to: s3://{self.config.BUCKET_NAME}/{metadata_key}")

    def deploy_model(self, model_uri):
        """Despliega el modelo en SageMaker"""
        logger.info("Deploying model to SageMaker...")

        pytorch_model = PyTorchModel(
            model_data=model_uri,
            role=self.config.ROLE_ARN,
            framework_version=self.config.FRAMEWORK_VERSION,
            py_version=self.config.PY_VERSION,
            entry_point='inference.py',
            source_dir=os.path.join(self.config.MODEL_NAME, 'code'),
        )

        endpoint_name = f"{self.config.ENDPOINT_NAME}-v{self.config.MODEL_VERSION.replace('.', '-')}"

        try:
            predictor = pytorch_model.deploy(
                endpoint_name=endpoint_name,
                instance_type=self.config.INSTANCE_TYPE,
                initial_instance_count=self.config.INITIAL_INSTANCE_COUNT,
                wait=True
            )

            logger.info(f"Model deployed successfully to endpoint: {endpoint_name}")
            return predictor, endpoint_name

        except Exception as e:
            logger.error(f"Error deploying model: {str(e)}")
            raise

    def run_deployment(self):
        """Ejecuta el pipeline completo de deployment"""
        try:
            logger.info("="*50)
            logger.info("Starting Model Deployment Pipeline")
            logger.info("="*50)

            # 1. Comprimir y subir modelo
            model_uri = self.push_model_to_s3()

            # 2. Desplegar modelo
            predictor, endpoint_name = self.deploy_model(model_uri)

            logger.info("="*50)
            logger.info("Deployment completed successfully!")
            logger.info(f"Endpoint name: {endpoint_name}")
            logger.info(f"Model URI: {model_uri}")
            logger.info("="*50)

            return endpoint_name, model_uri

        except Exception as e:
            logger.error(f"Deployment failed: {str(e)}")
            raise


if __name__ == "__main__":
    deployer = ModelDeployer()
    deployer.run_deployment()
