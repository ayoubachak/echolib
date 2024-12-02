from .hf import HuggingFaceModel
from .lm_studio import LMStudioModel
from common import globals_
from .base import BaseModel


class AIModels:
    def __init__(self) -> None:
        self.models : dict[str, BaseModel]= {"LM Studio": LMStudioModel.load_config("models/configs/lm_studio.config.json")}
        for hf_model in globals_.hf_models:
            model : HuggingFaceModel = HuggingFaceModel.setup_from_dict(hf_model.kwargs)
            model.preset = globals_.get_preset_by_id(model.preset)
            self.models[hf_model.name] = model

    
ai_models = AIModels()
