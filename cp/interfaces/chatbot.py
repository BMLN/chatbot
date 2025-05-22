from abc import abstractmethod, ABC
from inspect import signature, Parameter


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
        print("here?")
        return False
    
    if not (func_class is obj.__class__ or isinstance(func_class, obj.__class__)):
        return False
    
    return True


#TODO: assert
#TODO: self/cls not checked
def batchable(inherent=False):
    
    if callable(inherent):
        return batchable()(inherent)
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            
            if inherent: #for single unit TODO
                return func(*args, **kwargs)
            
            else:
                args = combine_args_kwargs(func, *args, **kwargs)

                item_keys = [ name for name, param in signature(func).parameters.items() if param.kind not in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD) ]
                items = { key: args[key] for key in item_keys[1:] if key in args } #temp_fix

                return list(func(**(args | dict(zip(items.keys(), x)))) for x in zip(*items.values()))
        
        return wrapper
    return decorator







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


    @batchable
    def respond(self, text, context=None, instructions=None):
        
        if instructions:
            pass
        else:
            if context is None and self.knowledgebase and self.vectorizer and self.matcher:
                context = self.matcher.match(self.vectorizer.vectorize(text), self.knowledgebase)
            
            if self.instructor:
                instructions = self.instructor.create_instructions(text, context)
            

        return self.generator.generate(instructions)







    class KnowledgeBase(ABC):
        """
            general object to offer crud interface (underlying object doesn't matter - in principal atleast)
        """

        @abstractmethod
        def create(self, **args):
            raise NotImplementedError
        
        @abstractmethod
        def retrieve(self, **args):
            raise NotImplementedError

        @abstractmethod
        def update(self, **args):
            raise NotImplementedError
        
        @abstractmethod
        def delete(self, **args):
            raise NotImplementedError
        
        @abstractmethod
        def search(self, x, **args):
            raise NotImplementedError



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
    