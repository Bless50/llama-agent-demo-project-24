from dotenv import load_dotenv

load_dotenv()
from llama_index.core.agent import ReActAgent
from llama_index.llms.groq import Groq
from llama_index.core.tools import FunctionTool
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex

# Load documents from the specified directory
documents = SimpleDirectoryReader("2023_budget.pdf").load_data()

# Create a Vector Store Index from the loaded documents
index = VectorStoreIndex.from_documents(documents)

# Create a query engine from the index
query_engine = index.as_query_engine()

# Test the query engine with a sample question
response = query_engine.query(
    "What was the total amount of the 2023 Canadian federal budget?"
)
print(response)

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
    return a -b

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


#create the tool class object
multiply_tool = FunctionTool.from_defaults(fn=multiply)
add_tool = FunctionTool.from_defaults(fn=add)
division_tool = FunctionTool.from_defaults(fn=division)
substraction_tool = FunctionTool.from_defaults(fn=substraction)
concatenate_tool = FunctionTool.from_defaults(fn=concatenate)
vowel_tool = FunctionTool.from_defaults(fn=count_vowels)
reverse_tool = FunctionTool.from_defaults(fn=reverse_string)


# Initializing the LLM
llm = Groq(model="llama3-70b-8192")

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
    ], 
    llm=llm, 
    verbose=True
)

# Ask a question
response = agent.chat("""
     what is 765 + 4 * 4444 / 45 -12345?
""")
print(response)