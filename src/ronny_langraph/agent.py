from langgraph.graph import add_messages
from langchain_core.messages import AnyMessage, AIMessage
from langgraph.prebuilt import ToolNode

from typing import Annotated, Any, MutableSequence, TypedDict
from langgraph.graph import StateGraph, END

from langchain_ollama import ChatOllama


class AgentState(TypedDict):
    messages: Annotated[MutableSequence[AnyMessage], add_messages]


# Define the config
class GraphConfig(TypedDict):
    pass


# Define the function that determines whether to continue or not
def should_continue(state: AgentState) -> str:
    messages = state["messages"]
    last_message = messages[-1]
    import rich.console

    rich.console.Console().print(messages)
    print(last_message)
    # If there are no tool calls, then we finish
    match last_message:
        case AIMessage(tool_calls=tool_calls) if tool_calls:
            return "end"
        case _:
            return "continue"


system_prompt = """Be a helpful assistant"""


# Define the function that calls the model
async def call_model(state: AgentState, config: GraphConfig) -> Any:
    messages = [{"role": "system", "content": system_prompt}, *state["messages"]]
    response = await model.ainvoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


def add(a: int, b: int) -> int:
    "adds 2 integers"
    return a + b


# Define the function to execute tools
tool_node = ToolNode([add])


model = ChatOllama(
    model="qwen2.5:7b",
).bind_tools([add])


# Define a new graph
workflow = StateGraph(AgentState, config_schema=GraphConfig)


# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.set_entry_point("agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
    # Finally we pass in a mapping.
    # The keys are strings, and the values are other nodes.
    # END is a special node marking that the graph should finish.
    # What will happen is we will call `should_continue`, and then the output of that
    # will be matched against the keys in this mapping.
    # Based on which one it matches, that node will then be called.
    {
        # If `tools`, then we call the tool node.
        "continue": "action",
        # Otherwise we finish.
        "end": END,
    },
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("action", "agent")

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
graph = workflow.compile()
