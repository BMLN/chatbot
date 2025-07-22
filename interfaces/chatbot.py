from abc import abstractmethod, ABC

from collections.abc import Iterable, Sized
from types import FunctionType, MethodType

from functools import wraps

from inspect import signature, isclass, ismethod, Parameter



# helper
def combine_args_kwargs(func, *args, **kwargs):
    sig = signature(func)
    params = list(sig.parameters.keys())

    combined = {}

    for name, value in zip(params, args):
        combined[name] = value

    combined.update(kwargs)


    return combined



#TODO ?
def is_class_function(obj, func):
    if not isinstance(obj, object):
        return False
    
    if not getattr(func, "__class__", function):
        return False
    
    if (func_class := getattr(func, "__self__", None)) == None:
        return False
    
    if not (func_class is obj.__class__ or isinstance(func_class, obj.__class__)):
        return False
    
    return True



def is_batch(x):

    if not isinstance(x, Sized) or isinstance(x, str):
        return False
    
    dim = None
    for _x in x:
        if _x is None or isinstance(_x, (str, int, float, bool)):
            _dim = 1
        elif isinstance(_x, Iterable):
            _dim = len(_x)
        else:
            return False
        
        if _dim != dim:
            if dim == None:
                dim = _dim
                continue
            
            return False

    return True





# decorators

# def batchable(inherent=False):
    
#     if callable(inherent):
#         return batchable()(inherent)
    
#     def decorator(func):
#         def wrapper(*args, **kwargs):
            
#             if inherent: #for single unit TODO
#                 return func(*args, **kwargs)
            
#             else:
#                 args = combine_args_kwargs(func, *args, **kwargs)

#                 item_keys = [ name for name, param in signature(func).parameters.items() if param.kind not in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD) ]
#                 items = { key: args[key] for key in item_keys[1:] if key in args } #temp_fix

#                 return list(func(**(args | dict(zip(items.keys(), x)))) for x in zip(*items.values()))
        
#         return wrapper
#     return decorator




# def batchable(inherent=False):

#     if callable(inherent):  
#         return batchable()(inherent)


#     def decorator(func):
#         @wraps(func)
#         def wrapper(*args, **kwargs):
#             if inherent:
#                 return func(*args, **kwargs)

            
#             arguments = signature(func).bind(*args, **kwargs)
#             arguments.apply_defaults()
#             arguments = arguments.arguments
#             cls = None
#             batch_keys = []
#             w = arguments[batch_keys[0]]
#             print(w, isinstance(func, (classmethod, staticmethod)), isinstance(type(func), (classmethod, staticmethod)))
#             exit()
            
#             if (isclass(arguments[batch_keys[0]]) or isclass(type(arguments[batch_keys[0]]))) and True:    
#                 batch_keys = batch_keys[1:]

#             first_arg = next((x for x in arguments.values() if x is not None), None)

#             if not (first_arg is not None and is_batch(first_arg)):
#                 return func(*args, **kwargs)
            
#             else:
#                 batch_size = len(first_arg)
#                 arguments = { key: value for key, value in arguments.items() if value is not None and is_batch(value) and len(value) == batch_size }

#                 batch_keys, batch_items = zip(*{ key: value for key, value in arguments.items() if (value is not None and is_batch(value) and len(value) == batch_size) }.items())

#                 return list(func(**(arguments | dict(zip(batch_keys, x)))) for x in zip(*batch_items))
                

#         return wrapper
#     return decorator

def batchable(inherent=False):

    if callable(inherent):  
        return batchable()(inherent)


    def decorator(func):
        assert len(signature(func).parameters), f"{func} can't be made batchable"


        @wraps(func)
        def wrapper(*args, **kwargs):
            if inherent:
                return func(*args, **kwargs)


            arguments = signature(func).bind(*args, **kwargs)
            arguments.apply_defaults()
            
            scalar_arguments = {}
            batch_arguments = {}
            batch_len = None
            
            for i, (key, value) in enumerate(arguments.arguments.items()):

                if i == 0:
                    if isclass(value):
                        __class = value
                    elif isclass(type(value)):
                        __class = type(value)
                    else:
                        __class = None

                    if __class and func.__qualname__.startswith(__class.__qualname__ + "."):
                        scalar_arguments[key] = value
                        continue
                        
                if not (value is not None): #have to is not None due to pandas
                    scalar_arguments[key] = value #also: None
                    
                elif not (is_batch(value) and (not batch_len or len(value) == batch_len)):
                    if not batch_len:   #didnt pass batches, can stop
                        break

                    scalar_arguments[key] = value

                else:
                    if not batch_len:
                        batch_len = len(value)
                            
                    batch_arguments[key] = value


            if not batch_len:
                return func(*args, **kwargs)
            
            else:
                return list(func(**(scalar_arguments | dict(zip(batch_arguments.keys(), x)))) for x in zip(*batch_arguments.values()))
                

        return wrapper
    return decorator




def batchify(kwarg):
    def decorator(func):
        while hasattr(func, "__wrapped__"): func = func.__wrapped__
        assert kwarg in signature(func).parameters.keys(), f"{kwarg} isn't a valid arguments for {str(func)}"
        
        #batcher
        @wraps(func)
        def wrapper(*args, **kwargs):
            params = signature(func).bind(*args, **kwargs)
            params.apply_defaults()
            
            if not (params.arguments[kwarg] is None) and not is_batch(params.arguments[kwarg]):
                params.arguments[kwarg] = [ params.arguments[kwarg] ]
                
            return func(*params.args, **params.kwargs)
        

        return wrapper
    return decorator





def inject_arg(arg_key, fill_with, only_if_none=False):
    def decorator(func):
        sig = signature(func)
        fill_args = [] if not isinstance(fill_with, FunctionType) else [ x for x in signature(fill_with).parameters if x in sig.parameters ]

        assert not isinstance(fill_with, FunctionType) or len(fill_args) == len(signature(fill_with).parameters), f"can't use {fill_with} to fill {func}-function"



        @wraps(func)
        def wrapper(*args, **kwargs):

            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            if not (only_if_none and bound.arguments.get(arg_key) is None):
                if isinstance(fill_with, FunctionType):
                    bound.arguments[arg_key] = fill_with(**{k: v for k, v in bound.arguments.items() if k in fill_args})
                else:
                    bound.arguments[arg_key] = fill_with

            return func(*bound.args, **bound.kwargs)

        
        return wrapper
    return decorator






# #TODO: assertions for config items
# #TODO: could add *arg passing as well
# def dec_injection(config={}):
#     config = { key.__name__: value for key, value in config.items() }


#     def class_decorator(cls):
#         for key, value in list(filter(lambda x: x[0] in config.keys(), cls.__dict__.items())):
#             if isinstance((__decorated := value), (classmethod, staticmethod)):
#                 __decorated = __decorated.__func__
                    
#             #adding decorators#
#             for decorator, args in config[key]:
#             #for decorator, args in config[key]:
#                 __decorated = decorator(**(args or {}))(__decorated)

#             if isinstance(value, staticmethod):
#                 __decorated = staticmethod(__decorated)
#             elif isinstance(value, classmethod):
#                 __decorated = classmethod(__decorated)

#             setattr(cls, value.__name__, __decorated)


#         return cls
#     return class_decorator


def dec_injection(config={}):
    config = { key.__name__: value for key, value in config.items() }
    
    def class_decorator(cls):
        funcs = list(filter(lambda x: x[0] in config.keys(), cls.__dict__.items()))
        
        for key, value in funcs:
            is_classmethod = isinstance(value, classmethod)
            is_staticmethod = isinstance(value, staticmethod)

            # Extract the actual function from descriptors
            if is_classmethod or is_staticmethod:
                func = value.__func__
            else:
                func = value

            # Apply decorators in correct order
            for decorator, kwargs in reversed(config[key]):
                kwargs = kwargs or {}
                func = decorator(**kwargs)(func)

            # Re-wrap in static/classmethod if needed
            if is_classmethod:
                func = classmethod(func)
            elif is_staticmethod:
                func = staticmethod(func)

            setattr(cls, key, func)
    return class_decorator











class Chatbot():
    """
       chatbot interface - intended for RAG usage, might be fine for other as well
    """
    
    def __init__(self, knowledgebase, vectorizer, matcher, instructor, generator):
        assert isinstance(knowledgebase, self.KnowledgeBase)
        assert isinstance(vectorizer, self.Vectorizer)
        assert isinstance(matcher, self.Matcher)
        assert isinstance(instructor, self.Instructor)
        assert isinstance(generator, self.Generator)
        
        self.knowledgebase = knowledgebase
        self.vectorizer = vectorizer
        self.matcher = matcher
        self.instructor = instructor
        self.generator = generator



    def load_context(self, data, keys, ids=None):
        self.knowledgebase.create(keys, data, ids)



    def respond(self, text, context=None, instructions=None):
        
        if instructions:
            pass

        else:
            if context is None and self.knowledgebase and self.vectorizer and self.matcher:
                context = self.matcher.match(self.vectorizer.vectorize(text), self.knowledgebase)

            if self.instructor:
                instructions = self.instructor.create_instructions(text, context)

            
        return self.generator.generate(**instructions)







    class KnowledgeBase(ABC):
        """
            general object to offer crud interface (underlying object doesn't matter - in principal atleast)
        """

        def __init_subclass__(cls):
            super().__init_subclass__()
            
            #could make assertions
            dec_injection(cls.CRUD_CFG)(cls)
            # inject_arg: {"arg_key": "id", "fill_with": KnowledgeBase.create_id, "only_if_none": True}


        @abstractmethod
        def create(self, id, data, **args):
            raise NotImplementedError
        
        @abstractmethod
        def retrieve(self, id, **args):
            raise NotImplementedError

        @abstractmethod
        def update(self, **args):
            raise NotImplementedError
        
        @abstractmethod
        def delete(self, **args):
            raise NotImplementedError
        
        @abstractmethod
        def search(self, **args):
            raise NotImplementedError

        @abstractmethod
        @batchable
        def create_id(self, data):
            raise NotImplementedError

        #TODO: use inspect instead of direct string name
        CRUD_CFG = { 
            create: [(batchify, {"kwarg": "id"}), (batchify, {"kwarg": "data"}), (inject_arg, {"arg_key": "id", "fill_with": create_id, "only_if_none": True}), (batchable, {})], 
            retrieve: [(batchify, {"kwarg": "id"}), (batchable, {})],
            update: [(batchify, {"kwarg": "id"}), (batchify, {"kwarg": "data"}), (batchable, {})],
            delete: [(batchify, {"kwarg": "id"}), (batchable, {})]
        }


    class Vectorizer(ABC):
        """
            defines the desired interaction to the data/knowledgebase
        """

        @abstractmethod
        def vectorize(self, text):
            raise NotImplementedError


    class Matcher(ABC):
        
        @classmethod
        @abstractmethod
        def match(cls, vector, knowledgebase, **args):
           raise NotImplementedError
           
            

    class Instructor(ABC):
        """
            module to formulate the input for the generation module in a form it can consume
        """

        @abstractmethod
        def create_instructions(self, text, context, **args):
            raise NotImplementedError



    class Generator(ABC):

        @abstractmethod
        def generate(self, **args):
            raise NotImplementedError
    