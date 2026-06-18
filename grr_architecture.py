"""
GRR multi-agent pipeline for bias=aware summarisation of humanitarian feedback.

Architecture overview:
Stage 1: Generators (3, run sequentially)
- Generator A: Red Cross Communications/Reporting Officer
- Generator B: Regional Coordinator
- Generator C: Protection and Inclusion Officer
Will all give a summary

Stage 2: Reviewer
- Bias-aware cross-summary analysis
- Gives back actionable feedback for the Rewriter to use

Stage 3: Rewriter
- Takes the actionable feedback from the Reviewer
- Outputs one final, revised summary based on that feedback

Each stage's output is preserved so that intermediate results can be inspected, logged and used for evaluation.
"""
# -----------------------------------------------------------------------------------------------------------------
### Imports ###
import os
from typing import TypedDict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI # for RC model
from langchain_google_genai import ChatGoogleGenerativeAI # for testing, free gemini
from langchain_ollama import ChatOllama # for Ollama model

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END, START

from config import LLM_PROVIDER, LLM_MODEL_GEMINI, LLM_MODEL_OLLAMA, LLM_MODEL_OPENAI, LLM_MODEL_LOCAL_OLLAMA, TEMPERATURE, MAX_TOKENS, OPENAI_BASE_URL
from prompts import (
    USER_PROMPT_1,                    # shared user prompt, same as baselines
    GENERATOR_A_SYSTEM_PROMPT,        # Communications/Reporting Officer persona
    GENERATOR_B_SYSTEM_PROMPT,        # Regional Coordinator persona
    GENERATOR_C_SYSTEM_PROMPT,        # Protection and Inclusion Officer persona
    REVIEWER_SYSTEM_PROMPT,           # Reviewer agent prompt
    REWRITER_SYSTEM_PROMPT,           # Rewriter agent prompt
    build_reviewer_user_message,      # assembles reviewer input from 3 summaries
    build_rewriter_user_message,      # assembles rewriter input from 3 summaries + feedback
)

load_dotenv() # load API keys from .env before LLM is initialised

# -----------------------------------------------------------------------------------------------------------------
### State ###

class GRRState(TypedDict):
    input_data: str         # raw input data as single string
    user_prompt: str        # user question
    summary_a: str          # output of first generate node (Communications Officer)
    summary_b: str          # output of second generate node (Regional Coordinator)
    summary_c: str          # output of third generate node (Protection and Inclusion Officer)
    reviewer_output: str     # output of review node (bias flag + rewriting instructions)
    revised_summary: str    # output of rewrite node (final, revised summary)
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

def generate_a(state: GRRState) -> dict:
    """
    Generator node A: Red Cross Communications/Reporting Officer persona.
    Focus on language accuracy, framing, hedging, non-affirmation.
    Receives same input_data and user prompts as baseline models.
    Persona is encoded in the system prompt.

    Returns dict with summary_a (raw text outputs from this generator)
    """

    system_prompt = GENERATOR_A_SYSTEM_PROMPT.format(input_data=state["input_data"])

    messages = [
        SystemMessage(content=system_prompt),           # set model's role and grounding instructions
        HumanMessage(content=state["user_prompt"]),     # human turn: specific question to answer
    ]

    response = llm.invoke(messages)

    print("DEBUG metadata:", response.response_metadata)

    usage = response.response_metadata.get("token_usage", 
        response.response_metadata.get("usage_metadata", {}))

    return {
        "summary_a": response.content,
        "token_usage": {**state.get("token_usage", {}), "generate_a": usage}
    }


def generate_b(state: GRRState) -> dict:
    """
    Generator node B: Regional Coordinator persona.
    Focus on operational/systemic coverage, omission of small groups, etc.
    Receives same input_data and user prompts as baseline models, same structure as generate_a but own persona system prompt.
    Persona is encoded in the system prompt.

    Returns dict with summary_b (raw text outputs from this generator)
    """

    system_prompt = GENERATOR_B_SYSTEM_PROMPT.format(input_data=state["input_data"])

    messages = [
        SystemMessage(content=system_prompt),           # set model's role and grounding instructions
        HumanMessage(content=state["user_prompt"]),     # human turn: specific question to answer
    ]

    response = llm.invoke(messages)
    print("DEBUG metadata:", response.response_metadata)
    
    usage = response.response_metadata.get("token_usage", 
        response.response_metadata.get("usage_metadata", {}))

    return {
        "summary_b": response.content,
        "token_usage": {**state.get("token_usage", {}), "generate_b": usage}
    }

def generate_c(state: GRRState) -> dict:
    """
    Generator node C: Protection and Inclusion Officer persona.
    Focus on equity, proportional group representation, underrepresentation of marginalised groups, etc.
    Receives same input_data and user prompts as baseline models, same structure as the other generate nodes but own persona system prompt.
    Persona is encoded in the system prompt.

    Returns dict with summary_c (raw text outputs from this generator)
    """

    # TODO: insert input data into generator system prompt

    system_prompt = GENERATOR_C_SYSTEM_PROMPT.format(input_data=state["input_data"])

    messages = [
        SystemMessage(content=system_prompt),           # set model's role and grounding instructions
        HumanMessage(content=state["user_prompt"]),     # human turn: specific question to answer
    ]

    response = llm.invoke(messages)
    print("DEBUG metadata:", response.response_metadata)
    usage = response.response_metadata.get("token_usage", 
        response.response_metadata.get("usage_metadata", {}))

    return {
        "summary_c": response.content,
        "token_usage": {**state.get("token_usage", {}), "generate_c": usage}
    }

def review(state: GRRState) -> dict:
    """
    Reviewer node: bias-away cross-summary analysis.
    Receives all 3 generator summaries (labeled A/B/C, not by persona name).
    Assesses summaries across 4 bias dimensions:
    1. Representation bias -> fair representation metric
    2. Ommission -> coverage score
    3. Factual accuracy -> alignment score
    4. Hedging & Non-affirmation

    Produces structured outputs with: BIAS FLAGS / POINTS OF CONVERGENCE / POINTS OF DIVERGENCE / REWRITING INSTRUCTIONS

    Output serves as actionable input to Rewriter and as audit trail for metric computation.

    Returs dict with reviewer_output.
    """
    # Reviewer works on the summaries, not raw feedback.

    user_message = build_reviewer_user_message(
        summary_a = state["summary_a"],
        summary_b = state["summary_b"],
        summary_c = state["summary_c"],
    )

    messages = [
        SystemMessage(content=REVIEWER_SYSTEM_PROMPT),  # set model's role and grounding instructions
        HumanMessage(content=user_message),     # the 3 summaries to review
    ]

    response = llm.invoke(messages)
    print("DEBUG metadata:", response.response_metadata)
    usage = response.response_metadata.get("token_usage", 
        response.response_metadata.get("usage_metadata", {}))

    return {
        "reviewer_output": response.content,
        "token_usage": {**state.get("token_usage", {}), "review": usage}
    }

def rewrite(state:GRRState) -> dict:
    """
    Rewriter node: produces a single, revised, bias-mitigated summary.

    Receives:
    - 3 original generator summaries (for grounding)
    - reviewer's structured, actionable feedback

    Constraints enforced by system prompts
    - Must follow each reviewer instruction precisely.
    - Must favour convergent content
    - Must NOT introduce claims not in original summaries (not new information).
    - No group's concerns should be stated in more hedged terms than others.

    Produces: REVISED SUMMARY / BIAS FLAGS ADDRESSED / FEEDBACK FORMS CONSULTED / RELEVANT IDS AND EXCERPTS

    Returns dict with revised_summary. Full rewriter output is stored in full for inspection.
    """

    user_message = build_rewriter_user_message(
        summary_a = state["summary_a"],
        summary_b = state["summary_b"],
        summary_c = state["summary_c"],
        reviewer_feedback=state["reviewer_output"], # structured feedback from review node
    )

    messages = [
        SystemMessage(content=REWRITER_SYSTEM_PROMPT),  # set model's role and grounding instructions
        HumanMessage(content=user_message),     # the 3 summaries to review + reviewer feedback
    ]

    response = llm.invoke(messages)
    print("DEBUG metadata:", response.response_metadata)
    usage = response.response_metadata.get("token_usage", 
        response.response_metadata.get("usage_metadata", {}))
    return {
        "revised_summary": response.content,
        "token_usage": {**state.get("token_usage", {}), "rewrite": usage}
    }

# No AIMessage needed, bc each agent has only one-turn conversation.
# It does not need its previous output as input (we explicitly do not want this).
# No node ever looks at its own prior input, and it shouldn't. Not being asked to reflect on smth it said itself (which is the case in Baseline 2)
# There is no conversational history.


# -----------------------------------------------------------------------------------------------------------------
### Graph ###

def build_graph():
    """
    Builds and compiles GRR graph.
    Node execution order: generate_a + generate_b + generate_c (in parallel) -> review -> rewrite -> END 
    Review node can only run once all three summaries are in state.
    """

    builder = StateGraph(GRRState)

    # Add all 5 agent nodes
    builder.add_node("generate_a", generate_a)  # Generator A: Communications Officer
    builder.add_node("generate_b", generate_b)  # Generator B: Regional Coordinator
    builder.add_node("generate_c", generate_c)  # Generator C: Protection & Inclusion Officer
    builder.add_node("review", review)          # Reviewer: cross-summary bias analysis
    builder.add_node("rewrite", rewrite)        # Rewriter: single revised summary

    builder.add_edge(START, "generate_a")
    builder.add_edge("generate_a", "generate_b") 
    builder.add_edge("generate_b", "generate_c") 
    builder.add_edge("generate_c", "review")    

    builder.add_edge("review", "rewrite")        # reviewer output ready, rewriter runs
    builder.add_edge("rewrite", END)             # pipeline complete

    return builder.compile()

# No reducer needed as generate a, b, and c write to a different state key (summary a, b and c).
# So there is no conflict and no reduced is needed.


# -----------------------------------------------------------------------------------------------------------------
### Run function ###

def run_grr(input_data: str, user_prompt: str = USER_PROMPT_1) -> dict:
    """
    Run full GRR pipeline.

    Parameters:
    - input_data: input data: # TODO: add how it looks like
    - user_prompt: task question, default to USER_PROMPT_1

    Returns: dict (final LangGraph state) with
    - "input_data"      : the original input (passed through)
    - "user_prompt"     : the question that was asked
    - "summary_a"       : raw Generator A output (Communications Officer)
    - "summary_b"       : raw Generator B output (Regional Coordinator)
    - "summary_c"       : raw Generator C output (Protection & Inclusion)
    - "reviewer_output" : full Reviewer output (bias flags + instructions)
    - "revised_summary" : full Rewriter output (use this as evaluated output, also contains BIAS FLAGS ADDRESSED audit trail)
    All intermediate fields are stored for ablation
    """
    graph = build_graph()

    initial_state: GRRState = {
        "input_data": input_data,
        "user_prompt": user_prompt,
        "summary_a": "",          # filled by generate_a node
        "summary_b": "",          # filled by generate_b node
        "summary_c": "",          # filled by generate_c node
        "reviewer_output": "",    # filled by review node
        "revised_summary": "",    # filled by rewrite node
        "token_usage": {},
    }

    # Run graph from entry point to END
    result = graph.invoke(initial_state)

    return result