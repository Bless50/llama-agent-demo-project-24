from dotenv import load_dotenv
from llama_index.core.agent import ReActAgent
from llama_index.llms.groq import Groq
from llama_index.core.tools import FunctionTool
from llama_index.core import SimpleDirectoryReader
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.tools import QueryEngineTool
from llama_parse import LlamaParse

load_dotenv()
# Set the Groq API key in the LLM settings
Settings.llm = Groq(model="llama-3.1-70b-versatile")


#setting the embeding model
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)


# rag pipeline
documents = LlamaParse(result_type="markdown").load_data("./data")
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()


# Creating the tools
def multiply(a: float, b: float) -> float:
    """Multiply two numbers and return the product."""
    return a * b

# Creating the second tool
def add(a: float, b: float) -> float:
    """Add two numbers and return the sum."""
    return a + b

#creating the third tool
def substraction(a:float, b: float) -> float:
    """
    Returns: the substraction of two number 
    """
    return a - b

#division tool
def division(a: float, b: float) -> float:
    """divide two numbers and returns their result """
    if b == 0 :
        raise ValueError("Cannot divide by zero.")
    return a / b 

#string manupulation tools
#concatenation tool
def concatenate(str1:str, str2:str) -> str:
    """concatenate two strings"""
    return str1 + str2

#counting of vowels tool
def count_vowels(s:str) -> int:
    """return the number of vowels in a string"""
    vowels = "aieou"
    count = 0
    for char in s.lower():
        if char in vowels:
            count +=1

    return count

# Function to reverse a string
def reverse_string(s: str) -> str:
    """Reverse the input string."""
    return s[::-1]


budget_tool = QueryEngineTool.from_defaults(
    query_engine,
    name="canadian_budget_2023",
    description="A RAG engine with some basic facts about the 2023 Canadian federal budget.",
)

#turning the functions to tools to passed the agent
multiply_tool = FunctionTool.from_defaults(fn=multiply)
add_tool = FunctionTool.from_defaults(fn=add)
division_tool = FunctionTool.from_defaults(fn=division)
substraction_tool = FunctionTool.from_defaults(fn=substraction)
concatenate_tool = FunctionTool.from_defaults(fn=concatenate)
vowel_tool = FunctionTool.from_defaults(fn=count_vowels)
reverse_tool = FunctionTool.from_defaults(fn=reverse_string)


# Initialize the agent
agent = ReActAgent.from_tools(
    [
        multiply_tool,
        add_tool,
        division_tool,
        substraction_tool,
        concatenate_tool,
        vowel_tool,
        reverse_tool,
        budget_tool
    ], 
     llm=Settings.llm,
    verbose=True
)

#test the agents.
response = query_engine.query("What is the allocation for green technologies?")
print(response)