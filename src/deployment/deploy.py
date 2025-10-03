import os
import boto3
import torch
import tarfile
import sagemaker
from sagemaker.pytorch.model import PyTorchModel

#* Cambiar las variables
MODEL_NAME = 'model_deteccion_imagenes'
BUCKET_NAME='objetos-random-tutoriales'
ROLE_ARN = "arn:aws:iam::675863513298:role/SageMaker_Pytorch_Tutorial"

def compress_model():
    current_path = os.getcwd()

    model_folder = os.path.join(current_path, MODEL_NAME)
    code_folder = os.path.join(model_folder, 'code')
    model_pth = os.path.join(model_folder, MODEL_NAME + '.pth')
    tar_file =  MODEL_NAME + '.tar.gz'

    with tarfile.open(tar_file, "w:gz") as tar:
        tar.add(code_folder, arcname=os.path.basename(code_folder))
        tar.add(model_pth, arcname=os.path.basename(model_pth))

    return tar_file

def push_model_to_s3():
    s3 = boto3.client('s3')
    filename = compress_model()
    print('key ', filename)
    s3.upload_file(
        Filename=filename,
        Bucket=BUCKET_NAME,
        Key=filename
    )

    model_s3_uri = f"s3://{BUCKET_NAME}/{filename}"
    return model_s3_uri

model_uri = push_model_to_s3()

pytorch_model = PyTorchModel(
    model_data=model_uri,
    role=ROLE_ARN,
    framework_version="1.12.0",
    py_version='py38',
    entry_point= os.path.join(MODEL_NAME,'code', 'inference.py'),
)

pytorch_model.deploy(
        endpoint_name="pytorch-model-endpoint",
        instance_type='ml.r5.large',
        initial_instance_count=1
)
