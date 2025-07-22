from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModelForTokenClassification, AutoModelForSequenceClassification
from transformers import DPRContextEncoder, DPRQuestionEncoder
from transformers import AutoConfig, AutoTokenizer, BitsAndBytesConfig

import torch

from huggingface_hub import snapshot_download
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

from os import makedirs











from logging import getLogger
logger = getLogger()


from transformers import logging
logging.set_verbosity_error()
getLogger("accelerate.utils.modeling").setLevel(40) #logging.ERROR = 40




# old lazy idle version
# class OnDemandModel():
#     def __init__(self, modelname, idle_timeout=180):
#         self.modelname = modelname
#         self.idle_timeout = idle_timeout
#         self.lock = threading.Lock()

#         self._load()
        



#     def _load(self):
#         self.tokenizer, self.model = self.load(self.modelname)
#         self.last_use = time.time()


#     def _unload(self):
#         del self.model
#         self.model = None


#     def __call__(self, text):
#         with self.lock:
#             if self.model is None:
#                 self._load()
            
#             return self.inference(text)
        
    
#     def inference(self, *args):
#         #COMPLETELY DEFAULT CALL
#         return self.model(*args)


#     @classmethod
#     def load(cls, modelname, modelcache="/modelcache"):
#         makedirs(modelcache, exist_ok=True)
#         __dl =  snapshot_download(modelname)


#         with init_empty_weights():
#             model = AutoModel.from_config(AutoConfig.from_pretrained(__dl))

#         tokenizer = AutoTokenizer.from_pretrained(__dl)
#         model = load_checkpoint_and_dispatch(
#             model=model,
#             checkpoint=__dl,
#             device_map="sequential",
#             offload_folder=modelcache
#         )


#         return tokenizer, model


#new
MODEL_SUPPORT = {
    "CausalLM": AutoModelForCausalLM,
    "Seq2SeqLM": AutoModelForSeq2SeqLM,
    "TokenClassification": AutoModelForTokenClassification,
    "SequenceClassification": AutoModelForSequenceClassification,
    "DPRContextEncoder": DPRContextEncoder,
    "DPRQuestionEncoder": DPRQuestionEncoder
}

def get_model_class_from_cfg(cfg):
    model_name = str((cfg.architectures or [None])[0])
    
    try:
        for suffix, auto_class in MODEL_SUPPORT.items():
            if model_name.endswith(suffix):

                return auto_class
            
        raise ModuleNotFoundError

    except ModuleNotFoundError:
        logger.warning(f"no matching Model for {model_name}, using AutoModel")
        
        return AutoModel  





class OnDemandModel():
    def __init__(self, modelname):
        self.modelname = modelname
        self.tokenizer = None
        self.model = None
        

    def _load(self):
        self.tokenizer, self.model = self.load(self.modelname)


    def _unload(self):
        del self.tokenizer
        del self.model

        self.tokenizer = None
        self.model = None


    def __call__(self, *args, **kwargs):
        self._load()
        output = self.inference(*args, **kwargs)
        self._unload()
        
        return output
        

    def inference(self, *args, **kwargs):
        #COMPLETELY DEFAULT CALL
        assert self.tokenizer and self.model

        return self.model(*args, **kwargs)


    @classmethod
    def load(cls, modelname, modelcache="/modelcache"):
        makedirs(modelcache, exist_ok=True)

        try: 
            __dl = snapshot_download(modelname, local_files_only=True)
        except:
            __dl =  snapshot_download(modelname)

        config = AutoConfig.from_pretrained(__dl)

        with init_empty_weights():
            model = get_model_class_from_cfg(config)
            model = model.from_pretrained(__dl, config=config)

        tokenizer = AutoTokenizer.from_pretrained(__dl)
        model = load_checkpoint_and_dispatch(
            model=model,
            checkpoint=__dl,
            device_map="sequential",
            offload_folder=modelcache
        )


        return tokenizer, model