import unittest
from unittest.mock import patch, MagicMock
from echolib.models.hf import HuggingFaceModel
from echolib.models.lm_studio import LMStudioModel
from echolib.common import globals_

class TestHuggingFaceModel(unittest.TestCase):
    def setUp(self):
        # Setup a HuggingFaceModel instance with mock data
        self.mock_config = {
            "api_url": "https://api-inference.huggingface.co/models",
            "headers": {
                "Authorization": "Bearer MOCK_API_KEY",
                "Content-Type": "application/json"
            },
            "model_huggingface_id": "mock/model-id",
            "default_parameters": {
                "max_length": -1,
                "max_new_tokens": 100,
                "temperature": 0.7,
                "use_cache": True,
                "wait_for_model": True
            }
        }
        self.model = HuggingFaceModel(api_url=self.mock_config["api_url"], headers=self.mock_config["headers"], config=self.mock_config)

    @patch('echolib.models.hf.requests.post')
    def test_generate_text_success(self, mock_post):
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = [b'{"generated_text": "Test response"}']
        mock_post.return_value = mock_response

        response = self.model.generate_text("Test prompt", {"max_new_tokens": 10})
        self.assertFalse(response["error"])
        self.assertEqual(response["generated_text"], "Test response")

    @patch('echolib.models.hf.requests.post')
    def test_generate_text_rate_limit(self, mock_post):
        # Mock rate limit response followed by success
        mock_response_429 = MagicMock()
        mock_response_429.status_code = 429
        mock_response_429.text = "Rate limit exceeded"

        mock_response_200 = MagicMock()
        mock_response_200.status_code = 200
        mock_response_200.iter_lines.return_value = [b'{"generated_text": "Rotated token response"}']

        mock_post.side_effect = [mock_response_429, mock_response_200]

        response = self.model.generate_text("Test prompt", {"max_new_tokens": 10})
        self.assertFalse(response["error"])
        self.assertEqual(response["generated_text"], "Rotated token response")

    @patch('echolib.models.hf.requests.post')
    def test_generate_text_failure(self, mock_post):
        # Mock failure response
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response

        with self.assertRaises(Exception):
            self.model.generate_text("Test prompt", {"max_new_tokens": 10})

class TestLMStudioModel(unittest.TestCase):
    def setUp(self):
        # Setup a LMStudioModel instance with mock data
        self.mock_config = {
            "api_url": "http://localhost:1234/v1",
            "instructions": "You are a helpful AI Assistant made by Alten.",
            "default_parameters": {
                "temperature": 0.7,
                "max_tokens": 100,
                "stream": False
            }
        }
        self.model = LMStudioModel(api_url=self.mock_config["api_url"], headers={}, config=self.mock_config)

    @patch('echolib.models.lm_studio.requests.post')
    def test_generate_text_success(self, mock_post):
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "Mocked response"}}]}
        mock_post.return_value = mock_response

        response = self.model.generate_text("Test prompt", {"temperature": 0.7})
        self.assertNotIn("error", response)
        self.assertIn("choices", response)
        self.assertEqual(response["choices"][0]["message"]["content"], "Mocked response")

    @patch('echolib.models.lm_studio.requests.post')
    def test_generate_text_failure(self, mock_post):
        # Mock failure response
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        mock_post.return_value = mock_response

        response = self.model.generate_text("Test prompt", {"temperature": 0.7})
        self.assertIn("error", response)
        self.assertEqual(response["error"], "400 Client Error: None for url: None")

if __name__ == '__main__':
    unittest.main()
