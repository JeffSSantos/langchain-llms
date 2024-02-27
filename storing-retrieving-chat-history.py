# Storing and Retrieving Chat History

from langchain_openai import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.schema import messages_from_dict, messages_to_dict

# create a manual chat history, using add_user and add_ai from ChatMessageHistory
history = ChatMessageHistory()
history.add_user_message("Hello! Let's talk about giraffes")
history.add_ai_message("Hi! I'm down to talk about giraffes")

# use the chat history as a dictionary
dicts = messages_to_dict(history.messages)
new_messages = messages_from_dict(dicts)

# Instantiate an OpenAI default model
llm = OpenAI(temperature=0)

# from the dictionary, create a new chat history, put them into a buffer memory and initialize a new chat
history = ChatMessageHistory(messages=new_messages)
buffer = ConversationBufferMemory(chat_memory=history)
conversation = ConversationChain(llm=llm, memory=buffer, verbose=True)

# Ask the model about "they". Because the history, the model has context and knows how to respond for "What are 'they'"
print(conversation.predict(input="What are they?"))

# Print the conversation history to exemplify how chat history works
print(conversation.memory)
