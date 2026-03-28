import os
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_openai import ChatOpenAI

# 1. Setup API Keys
os.environ["OPENAI_API_KEY"] = "your_openai_key"
os.environ["TAVILY_API_KEY"] = "your_tavily_key"

# 2. Define Tools
search = TavilySearchResults(k=3)
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
tools = [search, wikipedia]

# 3. Initialize LLM
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)

# 4. Pull Prompt Template (ReAct)
prompt = hub.pull("hwchase17/react")

# 5. Construct the Agent
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True, 
    handle_parsing_errors=True
)

def run_research(topic):
    instructions = f"""
    Research the topic: '{topic}'.
    Gather data from web search and Wikipedia.
    Once gathered, format the final response as a structured report with:
    - Cover Page (Topic, Date, Author: AI Agent)
    - Title
    - Introduction
    - Key Findings
    - Challenges
    - Future Scope
    - Conclusion
    """
    return agent_executor.invoke({"input": instructions})

# Execute
# result = run_research("Impact of AI in Healthcare")
# print(result['output'])
