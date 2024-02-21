# A plan and execute agent sample

from langchain_openai import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain import SerpAPIWrapper, LLMMathChain, WikipediaAPIWrapper
from langchain.agents.tools import Tool

llm = OpenAI(temperature=0)
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
search = SerpAPIWrapper()
wikipedia = WikipediaAPIWrapper()

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="Useful for when you need to answer questions about current events"
    ),
    Tool(
        name="Wikipedia",
        func=wikipedia.run,
        description="Useful for when you need to look up facts and statistics"
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="Useful for when you need to answer questions about math"
    )
]

prompt = ("Where are the next summer olympics going to be hosted? What is the population of that country raised to the "
          "0.43 power?")

model = ChatOpenAI(temperature=0)
planner = load_chat_planner(model)
executor = load_agent_executor(model, tools, verbose=True)
agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)

agent.invoke({'input': prompt})
