import os
import sys
import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from datetime import datetime
import yaml
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from config.config_loader import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SageMakerTrainingJob:
    """
    Lanza jobs de entrenamiento en SageMaker
    """

    def __init__(self, config_path='config/training_config.yaml'):
        # Cargar configuración
        with open(config_path, 'r') as f:
            self.training_config = yaml.safe_load(f)['training']

        self.config = config
        self.session = sagemaker.Session()
        self.role = self.config.ROLE_ARN
        self.timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')

    def launch_training(self):
        """Lanza el job de entrenamiento en SageMaker"""

        job_name = f"{self.training_config['model_name']}-{self.timestamp}"

        logger.info("="*50)
        logger.info(f"Launching SageMaker Training Job: {job_name}")
        logger.info("="*50)

        # Configurar el estimator de PyTorch
        estimator = PyTorch(
            entry_point='train.py',
            source_dir='src/training',
            role=self.role,
            instance_type=self.training_config['instance_type'],
            instance_count=self.training_config['instance_count'],
            framework_version=self.training_config['framework_version'],
            py_version=self.training_config['py_version'],
            hyperparameters={
                'epochs': self.training_config['epochs'],
                'batch-size': self.training_config['batch_size'],
                'learning-rate': self.training_config['learning_rate'],
                'image-size': self.training_config['image_size'],
                'num-classes': self.training_config['num_classes'],
                'bucket-name': self.training_config['bucket_name']
            },
            output_path=f"s3://{self.config.BUCKET_NAME}/models/training-output",
            base_job_name=self.training_config['model_name'],
            metric_definitions=[
                {'Name': 'train:loss', 'Regex': 'Train.*Loss: ([0-9\\.]+)'},
                {'Name': 'train:accuracy', 'Regex': 'Train.*Accuracy: ([0-9\\.]+)'},
                {'Name': 'validation:loss', 'Regex': 'Validation.*Loss: ([0-9\\.]+)'},
                {'Name': 'validation:accuracy', 'Regex': 'Validation.*Accuracy: ([0-9\\.]+)'}
            ],
            enable_sagemaker_metrics=True
        )

        # Lanzar entrenamiento
        # No necesitamos pasar datos porque los cargamos directamente de S3
        logger.info("Starting training job...")
        estimator.fit(wait=False)

        logger.info("="*50)
        logger.info(f"✓ Training job launched successfully!")
        logger.info(f"Job name: {estimator.latest_training_job.name}")
        logger.info(f"Check status in AWS Console or run:")
        logger.info(f"aws sagemaker describe-training-job --training-job-name {estimator.latest_training_job.name}")
        logger.info("="*50)

        # Guardar info del job
        self._save_job_info(estimator)

        return estimator

    def _save_job_info(self, estimator):
        """Guarda información del training job"""
        import json

        job_info = {
            'job_name': estimator.latest_training_job.name,
            'timestamp': self.timestamp,
            'hyperparameters': self.training_config,
            'model_artifacts': f"s3://{self.config.BUCKET_NAME}/models/training-output/{estimator.latest_training_job.name}/output/model.tar.gz"
        }

        filename = f"training_job_{self.timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(job_info, f, indent=2)

        logger.info(f"Job info saved to: {filename}")


if __name__ == "__main__":
    launcher = SageMakerTrainingJob()
    launcher.launch_training()
