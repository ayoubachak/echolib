from common.models import HFToken, ModelPreset, HFModel
from .logger import logger
import json


class Globals:
    def __init__(self):
        self.tokens : list[HFToken] = self.load_tokens()
        self.presets : list[ModelPreset] = self.load_presets()
        self.hf_models : list[HFModel] = self.load_hf_models()
        
    def load_tokens(self) -> list[HFToken] :
        try:
            with open('tokens.json', 'r') as file:
                tokens = json.load(file)
                assert len(tokens) > 0, "No tokens found, please make sure to add tokens to the tokens.json file."
                logger.info("Loaded {} tokens".format(len(tokens)))
                all_tokens = [HFToken(**token) for token in tokens]
                self.tokens = all_tokens
                return all_tokens
        except Exception as e:
            logger.error(f"Failed to load tokens: {e}")
            return []

    def load_presets(self) -> list[ModelPreset] :
        try:
            with open('presets.json', 'r') as file:
                presets = json.load(file)
                assert len(presets) > 0, "No presets found, please make sure to add presets to the presets.json file."
                logger.info("Loaded {} presets".format(len(presets)))
                all_presets = [ModelPreset(**preset) for preset in presets]
                self.presets = all_presets
                return all_presets
        except Exception as e:
            logger.error(f"Failed to load presets: {e}")
            return []
    
    def load_hf_models(self) -> list[HFModel]:
        try:
            with open('hf_models.json', 'r') as file:
                hf_models = json.load(file)
                assert len(hf_models) > 0, "No HuggingFace models found, please make sure to add models to the hf_models.json file."
                logger.info("Loaded {} HuggingFace models".format(len(hf_models)))
                all_hf_models = [HFModel(**model) for model in hf_models]
                self.hf_models = all_hf_models
                return all_hf_models
        except Exception as e:
            logger.error(f"Failed to load HuggingFace models: {e}")
            return []
    
    def get_preset_by_id(self, preset_id) -> ModelPreset:
        for preset in globals_.presets:
            if preset.id == preset_id:
                return preset
        return None
    
    def get_model_by_id(self, model_id) -> HFModel:
        for model in globals_.hf_models:
            if model.id == model_id:
                return model
        return None
    

globals_ = Globals()