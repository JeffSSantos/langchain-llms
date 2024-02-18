# Agent Human as a Tool sample

from langchain.agents import load_tools, create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI

llm = OpenAI(temperature=0)
tools = load_tools(["human"])

template = '''Answer the following questions as best you can. You have access to the following tools: {tools}.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}
'''

prompt = PromptTemplate.from_template(template)

# Construct the ReAct agent
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke({'input': "What's my friend Andi's surname?"})
