from interfaces import chatbot

from typing import override


import requests
from json import loads as json_load


"""
class APIGenerator(Generator):

    def __init__(self, url)


    pass
"""



class OllamaGenerator(chatbot.Chatbot.Generator):

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
    
    

    def __init__(self, url):
        self.url = url


    @override
    def generate(self, **args):
        call = self.__basecall.format(
            model="tinyllama",
            system_content="You are a helpful AI assistant.",
            text_content="Hello! How are you?"
        )
        call = json_load(call)

        response = requests.post(url=self.url, json=call)


        return response.json()