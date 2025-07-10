from chatbot.src.interfaces.chatbot import Chatbot


class OllamaContextInstructor(Chatbot.Instructor):

    #text should be the question
    @classmethod
    def create_instructions(cls, text, context=None):
        __context = "\\n".join([f"Context {i+1}: {x}" for i, x in enumerate(context)]) if context else None

        return {
            "system_content": "You are a helpful AI assistant.",
            "text_content": f"Question: {text}\\n{"Context:" + __context if context else ""}Answer:"
        }