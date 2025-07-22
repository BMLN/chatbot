from .inference_provider import InferenceProvider


import requests
from jinja2 import Template


from typing import override



class OllamaClient(InferenceProvider):
    
    def __init__(self, url, model):
        self.url = url + "/api/generate"
        self.model = model
        
        super().__init__()


    @override
    def call(self, prompt, system_prompt, *args, **kwargs):
        __basecall = Template("""{
            "model": "{{model}}",
            "messages": [
                {
                    "role": "system", 
                    "content": {{system_content | tojson }}
                },
                {
                    "role": "user",
                    "content": {{text_content | tojson }}
                }
            ],
            "temperature": 0,
            "keep_alive": 0,
            "stream": false
        }""")
        params = __basecall.render(*args, **{"model":self.model, "system_content": system_prompt, "text_content": prompt} | kwargs)

        
        response = requests.post(
            url=self.url, 
            headers={"Content-Type": "application/json"}, 
            data=params
        )


        return response
    

    @override
    def call(self, prompt, *args, **kwargs):
        __basecall = Template("""{
            "model": "{{model}}",
            "prompt": {{prompt | tojson}},
            "temperature": 0,
            "keep_alive": 0,
            "stream": false
        }""")
        params = __basecall.render(*args, **{"model":self.model, "prompt": prompt} | kwargs)

        response = requests.post(
            url=self.url, 
            headers={"Content-Type": "application/json"}, 
            data=params
        )


        return response
    

    @override
    def process_generation(self, generation_result):
        if generation_result.status_code != 200:
            raise Exception("<todo>")
        
        return generation_result.json().get("message", {}).get("content", None)


    @override
    def process_generation(self, generation_result):
        if generation_result.status_code != 200:
            raise Exception("<todo>")

        return generation_result.json().get("response", None)

