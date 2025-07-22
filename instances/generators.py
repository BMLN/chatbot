from chatbot.interfaces.chatbot import Chatbot, batchable
from chatbot.inference.models import OnDemandModel
from chatbot.inference_providers.ollama import OllamaClient

import torch

from typing import override


import requests
from json import loads as json_load, dumps as json_dump




"""
class APIGenerator(Generator):
    def __init__(self, url)
    pass
    
"""


class OllamaGenerator(Chatbot.Generator, OllamaClient):
    
    @override
    def generate(self, **args):
        return OllamaClient.generate(self, **args)







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
            