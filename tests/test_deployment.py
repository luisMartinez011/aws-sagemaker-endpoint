import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.deployment.deploy_model import ModelDeployer


class TestModelDeployer:
    """Tests para el ModelDeployer"""

    @patch('src.deployment.deploy_model.boto3.client')
    def test_deployer_initialization(self, mock_boto_client):
        """Test que el deployer se inicializa correctamente"""
        deployer = ModelDeployer()

        assert deployer.config is not None
        assert deployer.timestamp is not None
        mock_boto_client.assert_called()

    @patch('src.deployment.deploy_model.tarfile.open')
    @patch('src.deployment.deploy_model.os.path.exists')
    def test_compress_model(self, mock_exists, mock_tarfile):
        """Test que compress_model crea el archivo tar correctamente"""
        mock_exists.return_value = True
        mock_tar = MagicMock()
        mock_tarfile.return_value.__enter__.return_value = mock_tar

        deployer = ModelDeployer()
        filename = deployer.compress_model()

        assert filename.endswith('.tar.gz')
        assert deployer.config.MODEL_NAME in filename
        mock_tar.add.assert_called()

    @patch('src.deployment.deploy_model.boto3.client')
    def test_save_model_metadata(self, mock_boto_client):
        """Test que los metadatos se guardan correctamente"""
        mock_s3 = Mock()
        mock_boto_client.return_value = mock_s3

        deployer = ModelDeployer()
        deployer.s3_client = mock_s3

        deployer._save_model_metadata('test-key', 's3://bucket/model')

        mock_s3.put_object.assert_called_once()
        call_args = mock_s3.put_object.call_args
        assert 'metadata.json' in call_args[1]['Key']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
