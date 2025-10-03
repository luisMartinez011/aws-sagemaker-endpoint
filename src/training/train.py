import os
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import boto3
from datetime import datetime
import logging

from model import create_model
from dataset import create_dataloaders

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Clase para entrenar el modelo
    """

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Crear modelo
        self.model = create_model(num_classes=args.num_classes).to(self.device)

        # Loss y optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )

        # Métricas
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []

    def train_epoch(self, train_loader, epoch):
        """Entrena una época"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Estadísticas
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if batch_idx % 10 == 0:
                logger.info(
                    f'Epoch: {epoch} [{batch_idx * len(inputs)}/{len(train_loader.dataset)} '
                    f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                    f'Loss: {loss.item():.4f}'
                )

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total

        logger.info(f'Train Epoch {epoch}: Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

        return epoch_loss, epoch_acc

    def validate(self, val_loader, epoch):
        """Valida el modelo"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total

        logger.info(f'Validation Epoch {epoch}: Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

        return epoch_loss, epoch_acc

    def train(self, train_loader, val_loader):
        """Loop principal de entrenamiento"""
        logger.info("="*50)
        logger.info("Starting Training")
        logger.info("="*50)

        for epoch in range(1, self.args.epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)

            # Validate
            val_loss, val_acc = self.validate(val_loader, epoch)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)

            # Scheduler
            self.scheduler.step(val_loss)

            # Guardar mejor modelo
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_model('best_model.pth')
                logger.info(f'✓ New best model saved! Validation Accuracy: {val_acc:.2f}%')

            logger.info("-"*50)

        logger.info("="*50)
        logger.info(f"Training Completed! Best Validation Accuracy: {self.best_val_acc:.2f}%")
        logger.info("="*50)

    def save_model(self, filename):
        """Guarda el modelo"""
        model_path = os.path.join(self.args.model_dir, filename)
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")

    def save_metrics(self):
        """Guarda las métricas de entrenamiento"""
        metrics = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'best_val_acc': self.best_val_acc,
            'hyperparameters': {
                'learning_rate': self.args.learning_rate,
                'batch_size': self.args.batch_size,
                'epochs': self.args.epochs,
                'image_size': self.args.image_size
            }
        }

        metrics_path = os.path.join(self.args.output_data_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")


def parse_args():
    """Parse argumentos de línea de comando"""
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--image-size', type=int, default=64)
    parser.add_argument('--num-classes', type=int, default=2)

    # SageMaker parameters
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))
    parser.add_argument('--bucket-name', type=str, default='objetos-random-tutoriales')

    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("="*50)
    logger.info("Training Configuration")
    logger.info("="*50)
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
    logger.info("="*50)

    # Crear directorios si no existen
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_data_dir, exist_ok=True)

    # Crear DataLoaders
    logger.info("Loading datasets from S3...")
    train_loader, val_loader, test_loader = create_dataloaders(
        bucket_name=args.bucket_name,
        batch_size=args.batch_size,
        image_size=args.image_size
    )

    # Entrenar
    trainer = ModelTrainer(args)
    trainer.train(train_loader, val_loader)

    # Guardar modelo final
    trainer.save_model('model_deteccion_imagenes.pth')

    # Guardar métricas
    trainer.save_metrics()

    logger.info("✓ Training pipeline completed successfully!")


if __name__ == "__main__":
    main()
