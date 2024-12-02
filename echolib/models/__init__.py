from .hf import HuggingFaceModel
from .lm_studio import LMStudioModel
from .base import BaseModel
from echolib.common import globals_, ModelPreset, logger
import os

class AIModels:
    def __init__(self) -> None:
        self.models = {}
        self.load_hf_models()
        self.load_lm_studio_models()

    def load_hf_models(self) -> None:
        for hf_model in globals_.hf_models:
            if hf_model.type.upper() == "HUGGINGFACE":
                model = HuggingFaceModel.setup_from_dict(hf_model.kwargs)
                model.preset = globals_.get_preset_by_id(hf_model.preset)
                self.models[hf_model.name] = model
                logger.debug(f"Loaded HuggingFace model: {hf_model.name}")

    def load_lm_studio_models(self) -> None:
        # Example: Load LMStudio models if any
        # Assuming similar configuration loading
        lm_studio_config_path = os.path.join("echolib", "models", "configs", "lm_studio.config.json")
        if os.path.exists(lm_studio_config_path):
            lm_studio_model = LMStudioModel.setup_from_config(lm_studio_config_path)
            self.models["LM Studio"] = lm_studio_model
            logger.debug("Loaded LMStudio model: LM Studio")


ai_models = AIModels()
