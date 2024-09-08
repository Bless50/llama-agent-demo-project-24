from dotenv import load_dotenv

load_dotenv()
from llama_index.core.agent import ReActAgent
from llama_index.llms.groq import Groq
from llama_index.core.tools import FunctionTool

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
    """_summary_

    Args:
        a (float): _description_
        b (float): _description_

    Returns: the substraction of two number
        float: _description_
    """
    return a -b

#creating the four tool
def division(a: float, b: float) -> float:
    """_summary_

    Args:
        a (float): _description_
        b (float): _description_

    Returns:
        float: _description_
    """
    if b == 0 :
        raise ValueError("Cannot divide by zero.")
    return a / b 

#create the tool class object
multiply_tool = FunctionTool.from_defaults(fn=multiply)
add_tool = FunctionTool.from_defaults(fn=add)
division_tool = FunctionTool.from_defaults(fn=division)
substraction_tool = FunctionTool.from_defaults(fn=substraction)

# Initializing the LLM
llm = Groq(model="llama3-70b-8192")

# Initialize the agent
agent = ReActAgent.from_tools(
    [multiply_tool, add_tool, division_tool, substraction_tool], 
    llm=llm, 
    verbose=True
)

# Ask a question
response = agent.chat("What (2/2) * 3456 + 2345 - (3456 / 123)? ")
print(response)