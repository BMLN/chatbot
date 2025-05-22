from cp.interfaces.chatbot import Chatbot, batchable
from cp.inference.models import OnDemandModel

from sentence_transformers import SentenceTransformer
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoder, DPRContextEncoderTokenizer
import torch


from typing import override



class TFIDFVectorizer():
    pass


class TransformerVectorizer(Chatbot.Vectorizer):
    def __init__(self, model):
        assert isinstance(model, SentenceTransformer)

        super().__init__()
        self.model = model


    @override
    @batchable(inherent=True)
    def vectorize(self, text):
        return self.model.encode()
    

class DPRQEncoder(Chatbot.Vectorizer):
    def __init__(self, modelname):
        super().__init__()
        self.tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(modelname)
        self.model = DPRQuestionEncoder.from_pretrained(modelname)


    @override
    @batchable(inherent=True)
    def vectorize(self, text):
        input_ids = self.tokenizer(text, return_tensors="pt", padding="True")["input_ids"]
        embeddings = self.model(input_ids).pooler_output

        return embeddings.detach().numpy()
    
class DPRCEncoder(Chatbot.Vectorizer):
    def __init__(self, modelname):
        super().__init__()
        self.tokenizer = DPRContextEncoderTokenizer.from_pretrained(modelname)
        self.model = DPRContextEncoder.from_pretrained(modelname)

    @override
    @batchable(inherent=True)
    def vectorize(self, text):
        input_ids = self.tokenizer(text, return_tensors="pt", padding=True)["input_ids"]
        embeddings = self.model(input_ids).pooler_output

        return embeddings.detach().numpy()
    




class OnDemandDPREncoder(OnDemandModel, Chatbot.Vectorizer):
    
    def __init__(self):
        #super().__init__("facebook/dpr-question_encoder-multiset-base")
        super().__init__("facebook/dpr-question_encoder-single-nq-base")


    @override
    @batchable(inherent=True)
    def vectorize(self, text):
        return self.__call__(text)


    @override
    @batchable(inherent=True)
    def inference(self, text, *args, **kwargs):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        
        with torch.no_grad():
            outputs = self.model(**inputs)

            return outputs.pooler_output.squeeze().tolist()