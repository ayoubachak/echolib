from abc import ABC, abstractmethod
from typing import Any, Dict
import json 

class BaseModel(ABC):
    """
    Abstract base class for models to interact with APIs and perform data processing.
    """
    
    def __init__(self, api_url: str, headers: Dict[str, str], config: Dict[str, Any]) -> None:
        self.api_url = api_url
        self.headers = headers
        self.config = config
    
    @staticmethod
    @abstractmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        Loads configuration from a specified path.
        """
        with open(config_path, 'r') as file:
            return json.load(file)

    @abstractmethod
    def generate_text(self, prompt: str, parameters: Dict[str, Any]) -> Any:
        """
        Generates text based on a prompt and parameters.
        This method needs to be implemented by the subclass.
        """
        pass

    @abstractmethod
    def predict(self, prompt: str, params: Dict[str, Any]) -> Any:
        """
        Processes a prompt and returns a prediction.
        This method needs to be implemented by the subclass.
        """
        pass
    
    @abstractmethod
    def inference(self) -> str:
        """
        Performs inference using the model.
        This method needs to be implemented by the subclass.
        """
        pass
    @abstractmethod
    def sys_inference(self, sys_prompt, user_prompt, seed:int | None =None) -> str:
        """
        Performs inference using the model with system prompt .
        This method needs to be implemented by the subclass.
        """
        pass
    
    @abstractmethod
    def update_token(self, new_token: str) -> None:
        """w
        Updates the API token used for authentication.
        This method needs to be implemented by the subclass.
        """
        pass

    
    @abstractmethod
    def calc_tokens(self, prompt: str) -> int:
        """
        Calculates the number of tokens in a prompt.
        This method needs to be implemented by the subclass.
        """
        pass

    def interactive_prompt(self) -> None:
        """
        Optional: Implement an interactive prompt for testing purposes.
        This method can be overridden by subclasses for specific interactive functionality.
        """
        print("This method can be overridden by subclasses.")

    @classmethod
    def setup_from_config(cls, config_path: str):
        """
        Sets up the model based on the specified configuration.
        This method must be implemented by subclasses.
        """
        pass
    
    def setup_from_dict(cls, config_json: Dict[str, Any] | str ):
        """
        Sets up the model based on the specified configuration.
        This method must be implemented by subclasses.
        """
        pass