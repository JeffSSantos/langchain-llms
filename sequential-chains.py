# Simple Sequential Chains

from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI

# instantiate an OpenAI model with low temperature
llm = OpenAI(temperature=0)

# create a first prompt template and LLM chain that generates a company name
first_template = "What a good name for a company tha makes {product}?"
first_prompt = PromptTemplate.from_template(first_template)
first_chain = LLMChain(llm=llm, prompt=first_prompt)

# create the second prompt template and LLM chain that generates a company slogan for a given company name
second_template = "Write a catch phrase for the following company: {company_name}"
second_prompt = PromptTemplate.from_template(second_template)
second_chain = LLMChain(llm=llm, prompt=second_prompt)

# create an overall chain with the first and second chains where
# will use the response of the first prompt as input for the second prompt,
# meaning the company name generated in the first prompt will be used in the second prompt to generate the slogan.
overall_chain = SimpleSequentialChain(chains=[first_chain, second_chain], verbose=True)

# invoke the overall chain and gets the slogan for given company name
catchphrase = overall_chain.invoke({'input': "colorful socks"})
print(catchphrase)
