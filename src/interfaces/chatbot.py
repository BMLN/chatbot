
from abc import abstractmethod, ABC


class Chatbot():
    """
       chatbot interface - intended for RAG usage, might be fine for other as well
    """
    
    def __init__(self, knowledgebase, vectorizer, instructor, generator):
        assert isinstance(knowledgebase, self.KnowledgeBase)
        assert isinstance(vectorizer, self.Vectorizer)
        assert isinstance(instructor, self.Instructor)
        assert isinstance(generator, self.Generator)
        
        self.knowledgebase = knowledgebase
        self.vectorizer = vectorizer
        self.instructor = instructor
        self.generator = generator



    def respond(self, text, context=None, instructions=None):
        
        if instructions:
            pass
        else:
            if context is None and self.knowledgebase and self.vectorizer:
                context = self.vectorizer.match(text, self.knowledgebase)
            
            if self.instructor:
                instructions = self.instructor(text, context)
            

        return self.generator.generate(instructions)







    class KnowledgeBase(ABC):
        """
            general object to offer crud interface (underlying object doesn't matter - in principal atleast)
        """

        @abstractmethod
        def create(self, **args):
            raise NotImplementedError
        
        @abstractmethod
        def read(self, **args):
            raise NotImplementedError

        @abstractmethod
        def update(self, **args):
            raise NotImplementedError
        
        @abstractmethod
        def delete(self, **args):
            raise NotImplementedError
        


    class Vectorizer(ABC):
        """
            defines the desired interaction to the data/knowledgebase
        """

        @abstractmethod
        def vectorize(self, text):
            raise NotImplementedError

        @abstractmethod
        def search(self, vector, **args):
            raise NotImplementedError


        def match(self, text, knowledgebase, **args):
           
           embedding = self.vectorize(text)
           results = self.match(embedding, **args)


           return results
           
            

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
    