from .inference_provider import InferenceProvider
from huggingface_hub import InferenceClient


from typing import override







from logging import getLogger
logger = getLogger()
 





#create annotation that takes args and kwargs and verifies that specific keys are passed when called




class HfClient(InferenceProvider):

    @classmethod
    def create_client(cls, model, api_key, *args, **kwargs):
        """
            passthrough to the hf class (with minimal restrictions)
        """

        return InferenceClient(*args, **({"model": model, "api_key": api_key, "timeout": 120 } | kwargs))

    
    
    def __init__(self, model, hf_token):
        assert isinstance(model, (str, InferenceClient))

        if isinstance(model, str):
            self.client = self.create_client(model=model, api_key=hf_token)
        else:
            self.client = model
            
        super().__init__()

    
    
    @override
    def call(self, prompt, *args, **kwargs):
        __params = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."}
            ],
            "max_tokens": 512,
            "temperature": 0.1
        } 
        __params["messages"].append({"role": "user", "content": prompt})
        __params = __params | kwargs


        response = self.client.chat.completions.create(
            *args, **__params
        )


        return response



    @override
    def process_generation(self, generation_result):
        
        for x in generation_result.choices:
            if x.finish_reason == "stop":
                return x.message.content
            
        raise ValueError("generations iznougud")