import boto3
import json
import torch
from PIL import Image, ImageFile
from torchvision import transforms
import io

BUCKET_NAME='objetos-random-tutoriales'
FILENAME = 'gato.jpg'

def get_image():

    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=BUCKET_NAME, Key=FILENAME)
    image_data = response['Body'].read()  # Leer el contenido del archivo
    return image_data

def process_input(imagen):
    img_transforms = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225] )
        ])
    img = Image.open(io.BytesIO(imagen))
    img = img_transforms(img)
    input_image = torch.unsqueeze(img, 0).numpy().tolist()
    return input_image

def invoke_sagemaker_endpoint(endpoint_name, input_data):
    runtime = boto3.client('sagemaker-runtime')
    payload = json.dumps(input_data)
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=payload
    )
    result = json.loads(response['Body'].read().decode())
    print("Resultado: ", result)
    return result


def process_output(prediction):
    labels = ['cat','fish']
    result = labels[prediction]
    print(result)
    return result

imagen = get_image()
input_image = process_input(imagen)
output_tensor = invoke_sagemaker_endpoint("pytorch-model-endpoint",input_image)
prediction = process_output(output_tensor)


