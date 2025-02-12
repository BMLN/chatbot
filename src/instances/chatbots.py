import definitions.chatbot as chatbot






class Bot(chatbot.Chatbot):
    def __init__(self, context_model):
        
        pass

    class LLMRetriever(chatbot.Chatbot.ContextRetriever):
        pass

    class ContextEnhancer(chatbot.Chatbot.ContextEnhancer):
        pass