import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from config.config_loader import config
from model_deteccion_imagenes.code.inference import SimpleNet

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLflowModelTrainer:
    """Entrenamiento con tracking de MLflow"""

    def __init__(self, experiment_name="image-classification"):
        self.experiment_name = experiment_name

        # Configurar MLflow
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'))
        mlflow.set_experiment(experiment_name)

        logger.info(f"MLflow experiment: {experiment_name}")

    def train_model(self, epochs=10, lr=0.001, batch_size=32):
        """
        Entrena el modelo con tracking de MLflow

        Nota: Este es un ejemplo simulado. En producción,
        aquí irían tus datos reales y loop de entrenamiento.
        """

        with mlflow.start_run(run_name=f"training-{datetime.now().strftime('%Y%m%d-%H%M%S')}"):

            # Log de parámetros
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("learning_rate", lr)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("optimizer", "Adam")
            mlflow.log_param("model_architecture", "SimpleNet")

            # Log de configuración
            mlflow.log_param("image_size", config.IMAGE_SIZE)
            mlflow.log_param("num_classes", len(config.LABELS))

            # Inicializar modelo
            model = SimpleNet()
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()

            logger.info("Starting training simulation...")

            # Simulación de entrenamiento (reemplazar con datos reales)
            for epoch in range(epochs):
                # Simular métricas (en producción: calcular de verdad)
                train_loss = 0.5 * (0.9 ** epoch)  # Pérdida simulada que decrece
                train_acc = min(0.95, 0.5 + (0.05 * epoch))  # Accuracy simulada
                val_loss = 0.6 * (0.9 ** epoch)
                val_acc = min(0.93, 0.48 + (0.05 * epoch))

                # Log de métricas por época
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("train_accuracy", train_acc, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                mlflow.log_metric("val_accuracy", val_acc, step=epoch)

                logger.info(f"Epoch {epoch+1}/{epochs} - "
                          f"Loss: {train_loss:.4f} - "
                          f"Acc: {train_acc:.4f}")

            # Métricas finales
            final_metrics = {
                "final_train_accuracy": train_acc,
                "final_val_accuracy": val_acc,
                "final_train_loss": train_loss,
                "final_val_loss": val_loss
            }

            for metric_name, value in final_metrics.items():
                mlflow.log_metric(metric_name, value)

            # Guardar el modelo en MLflow
            logger.info("Logging model to MLflow...")
            mlflow.pytorch.log_model(
                model,
                "model",
                registered_model_name="image-classification-model"
            )

            # Guardar información adicional
            with open("model_info.txt", "w") as f:
                f.write(f"Model: SimpleNet\n")
                f.write(f"Training date: {datetime.now()}\n")
                f.write(f"Final accuracy: {train_acc:.4f}\n")

            mlflow.log_artifact("model_info.txt")
            os.remove("model_info.txt")

            # Tags para organización
            mlflow.set_tag("model_type", "CNN")
            mlflow.set_tag("framework", "PyTorch")
            mlflow.set_tag("task", "image_classification")
            mlflow.set_tag("status", "production_ready")

            logger.info("✅ Training completed and logged to MLflow")

            return model, final_metrics


def compare_models():
    """Compara modelos registrados en MLflow"""
    client = mlflow.tracking.MlflowClient()

    # Buscar todos los runs del experimento
    experiment = client.get_experiment_by_name("image-classification")

    if experiment:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.final_val_accuracy DESC"],
            max_results=5
        )

        logger.info("="*70)
        logger.info("TOP 5 MODELS BY VALIDATION ACCURACY")
        logger.info("="*70)

        for i, run in enumerate(runs, 1):
            logger.info(f"\n{i}. Run ID: {run.info.run_id}")
            logger.info(f"   Val Accuracy: {run.data.metrics.get('final_val_accuracy', 'N/A'):.4f}")
            logger.info(f"   Train Accuracy: {run.data.metrics.get('final_train_accuracy', 'N/A'):.4f}")
            logger.info(f"   Learning Rate: {run.data.params.get('learning_rate', 'N/A')}")
            logger.info(f"   Date: {run.info.start_time}")
    else:
        logger.warning("No experiment found")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train model with MLflow tracking')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--compare', action='store_true', help='Compare existing models')

    args = parser.parse_args()

    if args.compare:
        compare_models()
    else:
        trainer = MLflowModelTrainer()
        model, metrics = trainer.train_model(epochs=args.epochs, lr=args.lr)

        print("\n" + "="*70)
        print("TRAINING SUMMARY")
        print("="*70)
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
