import pytest
import torch
import json
import sys
import os

# Agregar path para importar módulos
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model_deteccion_imagenes.code.inference import (
    SimpleNet, model_fn, input_fn, predict_fn, output_fn
)


class TestSimpleNet:
    """Tests para la arquitectura del modelo"""

    def test_model_initialization(self):
        """Test que el modelo se inicializa correctamente"""
        model = SimpleNet()
        assert model is not None
        assert hasattr(model, 'fc1')
        assert hasattr(model, 'fc2')
        assert hasattr(model, 'fc3')

    def test_model_forward_pass(self):
        """Test que el forward pass funciona correctamente"""
        model = SimpleNet()
        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 64, 64)

        output = model(input_tensor)

        assert output.shape == (batch_size, 2)
        assert not torch.isnan(output).any()


class TestInferenceFunctions:
    """Tests para las funciones de inferencia"""

    def test_input_fn_valid_json(self):
        """Test que input_fn procesa JSON válido correctamente"""
        sample_data = [[0.1] * 12288]
        request_body = json.dumps(sample_data)

        result = input_fn(request_body, 'application/json')

        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 12288)

    def test_input_fn_invalid_content_type(self):
        """Test que input_fn rechaza content types inválidos"""
        with pytest.raises(ValueError, match="Unsupported content type"):
            input_fn("data", 'text/plain')

    def test_predict_fn_shape(self):
        """Test que predict_fn retorna el shape correcto"""
        model = SimpleNet()
        model.eval()
        input_data = torch.randn(1, 3, 64, 64)

        predictions = predict_fn(input_data, model)

        assert predictions.shape == (1, 2)
        assert isinstance(predictions, torch.Tensor)

    def test_output_fn_json(self):
        """Test que output_fn formatea correctamente la salida"""
        predictions = torch.tensor([[0.3, 0.7]])

        result = output_fn(predictions, 'application/json')
        parsed_result = json.loads(result)

        assert 'predicted_class' in parsed_result
        assert parsed_result['predicted_class'] == 1
        assert 'scores' in parsed_result

    def test_output_fn_invalid_content_type(self):
        """Test que output_fn rechaza content types inválidos"""
        predictions = torch.tensor([[0.3, 0.7]])

        with pytest.raises(ValueError, match="Unsupported content type"):
            output_fn(predictions, 'text/plain')


class TestEndToEnd:
    """Tests end-to-end del pipeline de inferencia"""

    def test_full_inference_pipeline(self):
        """Test del pipeline completo de inferencia"""
        # Simular carga del modelo
        model = SimpleNet()
        model.eval()

        # Simular input
        sample_data = torch.randn(1, 3, 64, 64).numpy().tolist()
        request_body = json.dumps(sample_data)

        # Pipeline completo
        input_tensor = input_fn(request_body, 'application/json')
        predictions = predict_fn(input_tensor, model)
        output = output_fn(predictions, 'application/json')

        # Validaciones
        result = json.loads(output)
        assert 'predicted_class' in result
        assert result['predicted_class'] in [0, 1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
