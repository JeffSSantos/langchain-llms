# OpenAI Model

from langchain_openai import OpenAI

# Instantiate a OpenAi default model with a more creative temperature
llm = OpenAI(temperature=0.9)

# define a prompt
prompt = "Whats would a good company name be for a company that makes colorful socks?"

# order a prompt for the model
print(llm.invoke(prompt))

# request the model to generate multiple responses for the same prompt
# result = llm.generate([prompt]*5)
# for company_name in result.generations:
#    print(company_name[0].text)
