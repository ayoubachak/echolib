class HFToken:
    def __init__(self, id: int, name: str, value: str) -> None:
        """
        Represents a token in the Hugging Face model.

        Args:
            id (int): The unique identifier for the token.
            name (str): The name of the token.
            value (str): The value of the token.
        """
        self.id = id
        self.name = name
        self.value = value

    def __repr__(self) -> str:
        return f"HFToken(id={self.id}, name='{self.name}', value='{self.value}')"

class ModelPreset:
    def __init__(
        self, 
        id: int = None, 
        name: str = None,
        input_prefix: str = None,
        input_suffix: str = None,
        antiprompt: str = None,
        pre_prompt: str = None,
        pre_prompt_prefix: str = None,
        pre_prompt_suffix: str = None,
    ) -> None:
        """
        Represents a preset configuration for a model.

        Args:
            id (int, optional): The unique identifier for the preset.
            name (str, optional): The name of the preset.
            input_prefix (str, optional): The prefix for input.
            input_suffix (str, optional): The suffix for input.
            antiprompt (str, optional): The antiprompt string.
            pre_prompt (str, optional): The pre-prompt string.
            pre_prompt_prefix (str, optional): The prefix for the pre-prompt.
            pre_prompt_suffix (str, optional): The suffix for the pre-prompt.
        """
        self.id = id
        self.name = name
        self.input_prefix = input_prefix
        self.input_suffix = input_suffix
        self.antiprompt = antiprompt
        self.pre_prompt = pre_prompt
        self.pre_prompt_prefix = pre_prompt_prefix
        self.pre_prompt_suffix = pre_prompt_suffix

    def __repr__(self) -> str:
        return (
            f"ModelPreset(id={self.id}, name='{self.name}', input_prefix='{self.input_prefix}', "
            f"input_suffix='{self.input_suffix}', antiprompt='{self.antiprompt}', "
            f"pre_prompt='{self.pre_prompt}', pre_prompt_prefix='{self.pre_prompt_prefix}', "
            f"pre_prompt_suffix='{self.pre_prompt_suffix}')"
        )
    
    

class HFModel:
    def __init__(
        self,
        id: int = None,
        name: str = None,
        type: str = None,
        kwargs: dict = None,
        preset: int = None
    ) -> None:
        """
        Represents a Hugging Face model configuration.

        Args:
            id (int, optional): The unique identifier for the model.
            name (str, optional): The name of the model.
            type (str, optional): The type of the model.
            kwargs (dict, optional): Additional keyword arguments for the model.
            preset (int, optional): The preset ID associated with the model.
        """
        self.id = id
        self.name = name
        self.type = type
        self.kwargs = kwargs if kwargs is not None else {}
        self.preset = preset

    def __repr__(self) -> str:
        return (
            f"HFModels(id={self.id}, name='{self.name}', type='{self.type}', "
            f"kwargs={self.kwargs}, preset={self.preset})"
        )
