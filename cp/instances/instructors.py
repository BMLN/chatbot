from interfaces.chatbot import Chatbot


class OllamaInstructor(Chatbot.Instructor):

    #text should be the question
    def create_instructions(self, text, context):
        __context = "\n".join([f"Context {i+1}: {x}" for i, x in enumerate(context)])

        return {
            "system_content": "You are a helpful AI assistant.",
            "text_content": f"Question: {text}\nContext: {__context}\Answer:"
        }