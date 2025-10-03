import os
import yaml
from dotenv import load_dotenv

load_dotenv()

class Config:
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # AWS Config
        self.AWS_REGION = os.getenv('AWS_REGION', config['aws']['region'])
        self.BUCKET_NAME = os.getenv('BUCKET_NAME')
        self.ROLE_ARN = os.getenv('SAGEMAKER_ROLE')

        # Model Config
        self.MODEL_NAME = config['model']['name']
        self.MODEL_VERSION = config['model']['version']
        self.FRAMEWORK_VERSION = config['model']['framework_version']
        self.PY_VERSION = config['model']['py_version']

        # Endpoint Config
        self.ENDPOINT_NAME = config['endpoint']['name']
        self.INSTANCE_TYPE = config['endpoint']['instance_type']
        self.INITIAL_INSTANCE_COUNT = config['endpoint']['initial_instance_count']

        # Inference Config
        self.IMAGE_SIZE = config['inference']['image_size']
        self.LABELS = config['inference']['labels']

        self._validate()

    def _validate(self):
        if not self.BUCKET_NAME:
            raise ValueError("BUCKET_NAME must be set in environment variables")
        if not self.ROLE_ARN:
            raise ValueError("SAGEMAKER_ROLE must be set in environment variables")

config = Config()
