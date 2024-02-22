# Memory and Chat Bots
import sys

from langchain.chains import ConversationChain
from langchain_openai import OpenAI

# Instantiate a OpenAI default model
llm = OpenAI(temperature=0)
conversation = ConversationChain(llm=llm, verbose=True)

print("Hi! I am your AI Chatbot. What can I do for you? At any tyme, you can type \"exit\" to end the chat.")

is_new_input = True
while is_new_input:
    human_input = input("You: ")

    if human_input == "exit":
        is_new_input = False
        sys.exit()

    ai_response = conversation.predict(input=human_input)
    print(f"AI: {ai_response}")
