from cp.interfaces.chatbot import Chatbot, batchable
from cp.inference.models import OnDemandModel

import torch

from typing import override


import requests
from json import loads as json_load




"""
class APIGenerator(Generator):
    def __init__(self, url)
    pass
    
"""


class OllamaGenerator(Chatbot.Generator):
    __basecall = """{{
        "model": "{model}",
        "messages": [
            {{
                "role": "system", 
                "content": "{system_content}"
            }},
            {{
                "role": "user",
                "content": "{text_content}"
            }}
        ],
        "temperature": 0,
        "stream": false
    }}"""

    def __init__(self, url, modelname):
        super().__init__()
        self.url = url
        self.model = modelname

    @override
    def generate(self, **args):
        call = self.__basecall.format(
            {"model": self.model } | args
        )
        call = json_load(call)

        response = requests.post(url=self.url, json=call)
        response = response.json()


        return response







class OnDemandGenerator(OnDemandModel, Chatbot.Generator):
    
    def __init__(self, modelname):
        #super().__init__("meta-llama/Llama-3.3-70B-Instruct")
        super().__init__(modelname)


    @override
    @batchable(inherent=True)
    def inference(self, texts, *args, **kwargs):
        tokenized = self.tokenizer(texts, return_tensors="pt")

        with torch.no_grad():
            output_ids = self.model.generate(
                do_sample=False, 
                return_dict_in_generate=False,
                **tokenized
            )
        #output_text = self.tokenizer.decode(output_ids[0][tokenized["input_ids"].shape[-1] - output_ids.shape[-1]:], skip_special_tokens=True)
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)


        return output_text


    @override
    @batchable(inherent=True)
    def generate(self, text):
        return self(text)
            