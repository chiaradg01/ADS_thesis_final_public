"""
Extends baseline 1 with second node that passes generated summary back to same LLM with zero-shot self-reflection prompt.
Asking it to reconsider whether the summary fairly represents all groups.

[generate] -> [reflect] -> END

Has same system prompts as Baseline 1 and same LLM is used in both nodes.
"""

# -----------------------------------------------------------------------------------------------------------------
### Imports ###
import os
from typing import TypedDict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI # for RC model
from langchain_google_genai import ChatGoogleGenerativeAI # for testing, free gemini
from langchain_ollama import ChatOllama # for Ollama model

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

from config import LLM_PROVIDER, LLM_MODEL_GEMINI, LLM_MODEL_OLLAMA, LLM_MODEL_OPENAI, LLM_MODEL_LOCAL_OLLAMA, TEMPERATURE, MAX_TOKENS
from config import OPENAI_BASE_URL
from prompts import SYSTEM_PROMPT, USER_PROMPT_1, SELF_REFLECTION_PROMPT

load_dotenv() # Load .env, containing API keys, before LLM is initialised
# -----------------------------------------------------------------------------------------------------------------
### State ###

class Baseline2State(TypedDict):
    input_data: str         # raw input data as single string
    user_prompt: str        # user question
    summary: str            # output of generate node (first turn), initial summary
    revised_summary: str    # output of reflect node (second turn), revised summary
    token_usage: dict       # stores dict of tokens per node


# -----------------------------------------------------------------------------------------------------------------
### LLM ###

# Checks which model is put in LLM_PROVIDER in config.py, uses that model

if LLM_PROVIDER == "gemini":
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL_GEMINI,
        temperature=TEMPERATURE,
        max_output_tokens=MAX_TOKENS,
    )
elif LLM_PROVIDER == "ollama":
    llm = ChatOllama(
        model=LLM_MODEL_OLLAMA,
        temperature=TEMPERATURE,
        num_predict=MAX_TOKENS,
    )
elif LLM_PROVIDER == "openai":
    llm = ChatOpenAI(
        model=LLM_MODEL_OPENAI,
        base_url=OPENAI_BASE_URL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
elif LLM_PROVIDER == "ollama-local":
    llm = ChatOpenAI(
        model=LLM_MODEL_LOCAL_OLLAMA,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        base_url="http://localhost:11434/v1",
        api_key = "ollama",
    )
else:
    raise ValueError(f"Unknown LLM_PROVIDER '{LLM_PROVIDER}' in config.py") # raises error if provider not matched

# -----------------------------------------------------------------------------------------------------------------
### Nodes ###

def generate(state: Baseline2State) -> dict:
    """
    Node 1: identical to Baseline1's generate node.
    Produces initial summary from input data.
    Receives current state, returns dict with updated fields.
    """
    system_prompt = SYSTEM_PROMPT.format(input_data=state["input_data"])

    messages = [
        SystemMessage(content=system_prompt),           # set model's role and grounding instructions
        HumanMessage(content=state["user_prompt"]),     # human turn: specific question to answer
    ]

    response = llm.invoke(messages)

    usage = response.response_metadata.get("token_usage", 
        response.response_metadata.get("usage_metadata", {}))

    return {
        "summary": response.content,
        "token_usage": {**state.get("token_usage", {}), "generate": usage}
    }


def reflect(state: Baseline2State) -> dict:
    """
    Node 2: self-reflection.
    Passes original summary back to LLM with reflection prompt.
    Conversation history is preserved so model has full context.
    Receives current state (including first summary) and returns updated fields.
    """
    system_prompt = SYSTEM_PROMPT.format(input_data=state["input_data"]) # reconstructs same system prompt used in generate node

    # Reconstruct conversion up to this point, then add reflection prompt as next human turn.
    messages = [
        SystemMessage(content=system_prompt),           # re-establish model's role and grounding instructions
        HumanMessage(content=state["user_prompt"]),     # original question
        AIMessage(content=state["summary"]),            # model's first response
        HumanMessage(content=SELF_REFLECTION_PROMPT),   # reflection instruction
    ]
    # Full conversation is passed so model has complete context for revision.

    response = llm.invoke(messages) # send full conversation, returns revised response

    usage = response.response_metadata.get("token_usage", 
        response.response_metadata.get("usage_metadata", {}))
    return {
        "revised_summary": response.content,
        "token_usage": {**state.get("token_usage", {}), "reflect": usage}
    }
# -----------------------------------------------------------------------------------------------------------------
### Graph ###

def build_graph():
    """
    Builds and compiles Baseline 2 graph.
    """
    builder = StateGraph(Baseline2State)

    builder.add_node("generate", generate)
    builder.add_node("reflect", reflect)

    builder.set_entry_point("generate")
    builder.add_edge("generate", "reflect")
    builder.add_edge("reflect", END)

    return builder.compile()


# -----------------------------------------------------------------------------------------------------------------
### Run function ###

def run_baseline2(input_data: str, user_prompt: str = USER_PROMPT_1) -> dict:
    """
    Run Baseline 2 on single input.

    Returns final state dict.
    Use result["revised_summary"] as output to evaluate -> post-reflection summary.
    result["summary"] gives the pre-reflection summary (use for ablation analysis)
    Wraps everything into single callable function.
    """
    graph = build_graph()

    initial_state: Baseline2State = {
        "input_data": input_data,
        "user_prompt": user_prompt,
        "summary": "",          # will be added by generate node
        "revised_summary": "",  # will be added by reflect node
        "token_usage": {}
    }

    # Run graph from entry point to END:
    result = graph.invoke(initial_state)

    return result