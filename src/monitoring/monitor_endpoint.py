import boto3
import json
import logging
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from config.config_loader import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EndpointMonitor:
    def __init__(self, endpoint_name=None):
        self.config = config
        self.endpoint_name = endpoint_name or self.config.ENDPOINT_NAME
        self.cloudwatch = boto3.client('cloudwatch', region_name=self.config.AWS_REGION)
        self.sagemaker = boto3.client('sagemaker', region_name=self.config.AWS_REGION)

    def get_endpoint_status(self):
        """Obtiene el estado actual del endpoint"""
        logger.info(f"Checking status for endpoint: {self.endpoint_name}")

        try:
            response = self.sagemaker.describe_endpoint(
                EndpointName=self.endpoint_name
            )

            status = {
                'endpoint_name': response['EndpointName'],
                'status': response['EndpointStatus'],
                'creation_time': response['CreationTime'].isoformat(),
                'last_modified': response['LastModifiedTime'].isoformat()
            }

            logger.info(f"Endpoint status: {status['status']}")
            return status

        except Exception as e:
            logger.error(f"Error getting endpoint status: {str(e)}")
            raise

    def get_invocation_metrics(self, hours=1):
        """Obtiene métricas de invocaciones del endpoint"""
        logger.info(f"Getting metrics for last {hours} hour(s)")

        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)

        metrics = {}

        # Métricas a consultar
        metric_names = [
            'Invocations',
            'ModelLatency',
            'Invocation4XXErrors',
            'Invocation5XXErrors'
        ]

        for metric_name in metric_names:
            try:
                response = self.cloudwatch.get_metric_statistics(
                    Namespace='AWS/SageMaker',
                    MetricName=metric_name,
                    Dimensions=[
                        {
                            'Name': 'EndpointName',
                            'Value': self.endpoint_name
                        },
                        {
                            'Name': 'VariantName',
                            'Value': 'AllTraffic'
                        }
                    ],
                    StartTime=start_time,
                    EndTime=end_time,
                    Period=3600,
                    Statistics=['Sum', 'Average', 'Maximum']
                )

                metrics[metric_name] = response['Datapoints']
                logger.info(f"{metric_name}: {len(response['Datapoints'])} datapoints")

            except Exception as e:
                logger.error(f"Error getting {metric_name}: {str(e)}")
                metrics[metric_name] = []

        return metrics

    def check_endpoint_health(self):
        """Verifica la salud del endpoint"""
        logger.info("="*50)
        logger.info("ENDPOINT HEALTH CHECK")
        logger.info("="*50)

        health_report = {
            'timestamp': datetime.utcnow().isoformat(),
            'endpoint_name': self.endpoint_name,
            'status': None,
            'metrics': None,
            'issues': []
        }

        # 1. Verificar estado del endpoint
        try:
            status = self.get_endpoint_status()
            health_report['status'] = status

            if status['status'] != 'InService':
                health_report['issues'].append(
                    f"Endpoint not in service: {status['status']}"
                )
        except Exception as e:
            health_report['issues'].append(f"Cannot get endpoint status: {str(e)}")

        # 2. Verificar métricas
        try:
            metrics = self.get_invocation_metrics(hours=1)
            health_report['metrics'] = metrics

            # Verificar errores
            errors_4xx = sum([dp['Sum'] for dp in metrics.get('Invocation4XXErrors', [])])
            errors_5xx = sum([dp['Sum'] for dp in metrics.get('Invocation5XXErrors', [])])

            if errors_4xx > 0:
                health_report['issues'].append(f"Found {errors_4xx} 4XX errors")
            if errors_5xx > 0:
                health_report['issues'].append(f"Found {errors_5xx} 5XX errors")

            # Verificar latencia
            latencies = [dp['Average'] for dp in metrics.get('ModelLatency', []) if 'Average' in dp]
            if latencies:
                avg_latency = sum(latencies) / len(latencies)
                if avg_latency > 1000:  # > 1 segundo
                    health_report['issues'].append(
                        f"High latency detected: {avg_latency:.2f}ms"
                    )

        except Exception as e:
            health_report['issues'].append(f"Cannot get metrics: {str(e)}")

        # Resumen
        if not health_report['issues']:
            logger.info("✅ Endpoint is healthy!")
        else:
            logger.warning("⚠️  Issues detected:")
            for issue in health_report['issues']:
                logger.warning(f"  - {issue}")

        logger.info("="*50)

        return health_report

    def generate_report(self):
        """Genera un reporte completo del endpoint"""
        report = self.check_endpoint_health()

        # Guardar reporte
        report_filename = f"endpoint_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w') as f:
            json.dumps(report, indent=2, default=str)

        logger.info(f"Report saved to: {report_filename}")
        return report


if __name__ == "__main__":
    monitor = EndpointMonitor()
    report = monitor.generate_report()
    print(json.dumps(report, indent=2, default=str))
