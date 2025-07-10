from chatbot.src.interfaces.chatbot import Chatbot
from chatbot.src.instances.knowledgebases import WeaviateKB

from typing import override


class WeaviateMatcher(Chatbot.Matcher):
    

    @classmethod
    @override
    def match(cls, vector, knowledgebase, **args):
        assert isinstance(knowledgebase, WeaviateKB)
        #get only text
        dist = -80
        prop = "text"

        result = knowledgebase.search(vector, **args)
        result = [ x.properties.get(prop) for x in result.objects if x.metadata.distance and x.metadata.distance > dist ]
        
        
        return result