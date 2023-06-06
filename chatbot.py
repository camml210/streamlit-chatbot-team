from typing import Callable
from langchain import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain import LLMChain
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish
from langchain.agents.agent_toolkits import NLAToolkit
from langchain.tools.plugin import AIPlugin
import re
import plugnplai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings

class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools_getter: Callable
    
    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Get the tools
        tools = self.tools_getter(kwargs["input"])
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in tools])
        return self.template.format(**kwargs)

class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

class ChatbotTeam:
    def __init__(self):
        # Get the plugins
        urls = plugnplai.get_plugins()
        AI_PLUGINS = []
        for url in urls:
            try:
                AI_PLUGINS.append(AIPlugin.from_url(url + "/.well-known/ai-plugin.json"))
            except:
                pass

        # LLM "tiiuae/falcon-7b-instruct"
        self.llm = HuggingFacePipeline.from_model_id(
            model_id="tiiuae/falcon-7b-instruct", 
            task="text-generation", 
            model_kwargs={
                "temperature":0.8, 
                "max_length":1024
                },
            trust_remote_code=True,
            )

        # Set up the vectorizer and toolkits
        vectorizer = TfidfVectorizer()
        docs = [
            Document(page_content=plugin.description_for_model, 
                     metadata={"plugin_name": plugin.name_for_model}
                    )
            for plugin in AI_PLUGINS
        ]
        vector_store = vectorizer.fit_transform([doc.page_content for doc in docs])
        toolkits_dict = {plugin.name_for_model: 
                         NLAToolkit.from_llm_and_ai_plugin(self.llm, plugin) 
                         for plugin in AI_PLUGINS}

        self.retriever = vector_store

        # Get the tools
        self.tools = self.get_tools("Generate 10 ideas for making money trading stocks or performing other online services or tasks.")

        # Set up the base template
        template = """Answer the following questions as best you can. You have access to the following tools:

        {tools}

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
        {agent_scratchpad}"""

        # Prompt template
        self.prompt = CustomPromptTemplate(
            template=template,
            tools_getter=self.get_tools,
            # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
            # This includes the `intermediate_steps` variable because that is needed
            input_variables=["input", "intermediate_steps"]
        )

        # Output parser
        self.output_parser = CustomOutputParser()

        # LLM chain consisting of the LLM and a prompt
        self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt)

        # Get the tool names
        tool_names = [x.metadata['plugin_name'] for x in docs]

        # Set up the agent
        self.agent = LLMSingleActionAgent(
            llm_chain=self.llm_chain, 
            output_parser=self.output_parser,
            stop=["\nObservation:"], 
            allowed_tools=tool_names
        )

        # Set up the agent executor
        self.agent_executor = AgentExecutor.from_agent_and_tools(agent=self.agent, tools=self.tools, verbose=True)

        # Initialize the cache for tools
        self.tool_cache = {}

    def get_tools(self, query):
        # Check if the tools for this query are in the cache
        if query in self.tool_cache:
            return self.tool_cache[query]

        # If not, compute the tools
        query_vector = vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.retriever).flatten()
        sorted_indices = similarities.argsort()[::-1]
        relevant_docs = [docs[idx] for idx in sorted_indices if similarities[idx] > 0]
        tool_kits = [toolkits_dict[doc.metadata["plugin_name"]] for doc in relevant_docs]
        tools = []
        for tk in tool_kits:
            tools.extend(tk.nla_tools)

        # Store the tools in the cache
        self.tool_cache[query] = tools

        return tools

    def run(self, input):
        # Run the agent executor with the given input
        return self.agent_executor.run(input)
