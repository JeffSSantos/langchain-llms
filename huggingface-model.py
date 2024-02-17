# HuggingFace model

from langchain_community.llms import HuggingFaceHub

# Instantiate a HuggingFace Google flan-t5-base model
# repo_id="google/flan-t5-base": uses the Google flan-t5-base model
# model_kwargs: the model specification arguments dictionary
# temperature: define the model temperature
# max_length: limits the maximum characters of the model response
llm = HuggingFaceHub(repo_id="google/flan-t5-base",
                     model_kwargs={"temperature": 0, "max_length": 64})

# define a prompt
prompt = "What are good fitness tips?"

# order a prompt for the model
print(llm.invoke(prompt))
