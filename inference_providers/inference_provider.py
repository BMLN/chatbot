from abc import ABC, abstractmethod

# from inspect import signature
# from functools import wraps




class InferenceProvider(ABC):
    
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def call(self, *args, **kwargs) -> str:
        """
            straight throughput to inference unit
        """
        pass

    @abstractmethod
    def process_generation(self, generation_result):
        pass


    def generate(self, prompt, *args, **kwargs):
        return self.process_generation(self.call(prompt, *args, **kwargs))




# def assert_kwargs(arg_keys: list[str]):
#     """
#         Decorator to check if the passed *args/**kwargs match specific arg_keys
#     """
#     assert isinstance(arg_keys, list)
    
    
#     def decorator(func):
#         @wraps(func)
#         def wrapper(*args, **kwargs):

#             if not arg_keys:
#                 return func(*args, **kwargs)
            

#             for i, x in enumerate(signature(func).parameters):
#                 print(x)
#                 if i < len(args):
#                     __param = x
#                 elif not kwargs:
#                     break
#                 else:
#                     pass
#                     #__param = list(kwargs.keys())[i - len(args)] #TODO: slow af
                
#                 if __param in arg_keys:
#                     arg_keys.remove(__param)

#             if arg_keys: 
#                 raise ValueError(f"Invalid parameter names: {arg_keys}")
            
            
#             return func(*args, **kwargs)

#         return wrapper
#     return decorator
