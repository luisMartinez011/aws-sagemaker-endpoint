import os
import boto3
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import io
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class S3ImageDataset(Dataset):
    """
    Dataset que carga imágenes directamente desde S3
    """

    def __init__(self, bucket_name, prefix, transform=None):
        """
        Args:
            bucket_name: Nombre del bucket S3
            prefix: Prefijo del path (ej: 'train/cat')
            transform: Transformaciones de torchvision
        """
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.transform = transform
        self.s3_client = boto3.client('s3')

        # Obtener lista de imágenes
        self.image_keys = self._list_images()
        logger.info(f"Found {len(self.image_keys)} images in s3://{bucket_name}/{prefix}")

    def _list_images(self):
        """Lista todas las imágenes en el prefijo de S3"""
        image_keys = []
        paginator = self.s3_client.get_paginator('list_objects_v2')

        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=self.prefix):
            if 'Contents' not in page:
                continue

            for obj in page['Contents']:
                key = obj['Key']
                # Filtrar solo imágenes
                if key.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_keys.append(key)

        return image_keys

    def _load_image_from_s3(self, key):
        """Carga una imagen desde S3"""
        response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
        image_data = response['Body'].read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        return image

    def __len__(self):
        return len(self.image_keys)

    def __getitem__(self, idx):
        img_key = self.image_keys[idx]
        image = self._load_image_from_s3(img_key)

        if self.transform:
            image = self.transform(image)

        # Extraer label del path (cat o fish)
        label = 0 if 'cat' in img_key.lower() else 1

        return image, label


def get_data_transforms(image_size=64):
    """
    Retorna las transformaciones para train y val
    """
    train_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    return train_transforms, val_transforms


def create_dataloaders(bucket_name, batch_size=32, image_size=64, num_workers=2):
    """
    Crea DataLoaders para train, val y test
    """
    train_transforms, val_transforms = get_data_transforms(image_size)

    # Datasets
    train_dataset = S3ImageDataset(
        bucket_name=bucket_name,
        prefix='train',
        transform=train_transforms
    )

    val_dataset = S3ImageDataset(
        bucket_name=bucket_name,
        prefix='val',
        transform=val_transforms
    )

    test_dataset = S3ImageDataset(
        bucket_name=bucket_name,
        prefix='test',
        transform=val_transforms
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader
