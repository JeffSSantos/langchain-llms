# Prompt templating and chaining

from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.chains import LLMChain

template = "You are a naming consultant for new companies. What a good name for a {company} that makes {product}?"

prompt = PromptTemplate.from_template(template)
llm = OpenAI(temperature=0.9)

chain = LLMChain(llm=llm, prompt=prompt)

print(chain.invoke({'company': "ABC Startup",
                    'product': "colorful socks"}))
