"""
Ablation variants of the GRR pipeline.

All ablations reuse all nodes from grr_architecture.py where possible.
Only the graph structure and/or the rewriter prompt differ.
"""

# -----------------------------------------------------------------------------------------------------------------
### Imports ###
from typing import TypedDict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END, START

from config import (
    LLM_PROVIDER, LLM_MODEL_GEMINI, LLM_MODEL_OLLAMA, OPENAI_BASE_URL,
    LLM_MODEL_OPENAI, LLM_MODEL_LOCAL_OLLAMA, TEMPERATURE, MAX_TOKENS
)

from prompts import (
    USER_PROMPT_1,
    REWRITER_NO_REVIEWER_SYSTEM_PROMPT,
    REWRITER_SYSTEM_PROMPT,
    REVIEWER_SYSTEM_PROMPT,
    build_rewriter_no_reviewer_user_message,
    build_rewriter_user_message,
    build_reviewer_user_message,
    GENERATOR_A_SYSTEM_PROMPT,
    GENERATOR_B_SYSTEM_PROMPT,
    GENERATOR_C_SYSTEM_PROMPT,
    REVIEWER_NO_GENERATORS_SYSTEM_PROMPT,
    REWRITER_NO_GENERATORS_SYSTEM_PROMPT,
    build_reviewer_no_generators_user_message,
    build_rewriter_no_generators_user_message,
    build_reviewer_user_message_no_genA,
    build_rewriter_user_message_no_genA,
    build_reviewer_user_message_no_genB,
    build_rewriter_user_message_no_genB,
    build_reviewer_user_message_no_genC,
    build_rewriter_user_message_no_genC,
)

load_dotenv()

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
### Shared state ###
class GRRAblationState(TypedDict):
    input_data:       str   # raw input data
    user_prompt:      str   # user question
    summary_a:        str   # Generator A output
    summary_b:        str   # Generator B output
    summary_c:        str   # Generator C output
    reviewer_output:  str   # Reviewer output (empty string if no_reviewer condition)
    revised_summary:  str   # Rewriter output
    token_usage: dict       # stores dict of tokens per node


# -----------------------------------------------------------------------------------------------------------------
### Nodes ###
def generate_a(state: GRRAblationState) -> dict:
    """
    Same as in grr_architecture.py
    """
    system_prompt = GENERATOR_A_SYSTEM_PROMPT.format(input_data=state["input_data"])

    messages = [
        SystemMessage(content=system_prompt),          
        HumanMessage(content=state["user_prompt"]),    
    ]

    response = llm.invoke(messages)

    print("DEBUG metadata:", response.response_metadata)

    usage = response.response_metadata.get("token_usage", 
        response.response_metadata.get("usage_metadata", {}))

    return {
        "summary_a": response.content,
        "token_usage": {**state.get("token_usage", {}), "generate_a": usage}
    }


def generate_b(state: GRRAblationState) -> dict:
    """
    Same as in grr_architecture.py
    """
    system_prompt = GENERATOR_B_SYSTEM_PROMPT.format(input_data=state["input_data"])

    messages = [
        SystemMessage(content=system_prompt),           
        HumanMessage(content=state["user_prompt"]),     
    ]

    response = llm.invoke(messages)

    print("DEBUG metadata:", response.response_metadata)
    
    usage = response.response_metadata.get("token_usage", 
        response.response_metadata.get("usage_metadata", {}))

    return {
        "summary_b": response.content,
        "token_usage": {**state.get("token_usage", {}), "generate_b": usage}
    }

def generate_c(state: GRRAblationState) -> dict:
    """
   Same as in grr_architecture.py
    """
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

def review(state: GRRAblationState) -> dict:
    """
    Identical to review node in grr_architecture.py
    """
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

### grr_no_reviewer ###

def rewrite_no_reviewer(state:GRRAblationState) -> dict:
    """
    Rewriter node for grr_no_reviewer condition.
    Receives 3 summaires but no reviewer feedback.
    Uses REWRITER_NO_REVIEWER_SYSTEM_PROMPT instead of standard Rewriter prompt.
    """

    user_message = build_rewriter_no_reviewer_user_message(
        summary_a = state["summary_a"],
        summary_b = state["summary_b"],
        summary_c = state["summary_c"],
    )

    messages = [
        SystemMessage(content=REWRITER_NO_REVIEWER_SYSTEM_PROMPT),  
        HumanMessage(content=user_message),    
    ]

    response = llm.invoke(messages)

    print("DEBUG metadata:", response.response_metadata)
    usage = response.response_metadata.get("token_usage", 
        response.response_metadata.get("usage_metadata", {}))
    return {
        "revised_summary": response.content,
        "token_usage": {**state.get("token_usage", {}), "rewrite": usage}
    }

### grr_no_generators ###
def review_no_generators(state:GRRAblationState) -> dict:
    """
    Reviewer node for grr_no_generators condition.
    Receives raw input data directly instead of 3 generator summaries.
    Analyses the data for bias patterns and produces structured feedback.
    """
    user_message = build_reviewer_no_generators_user_message(
        input_data=state["input_data"],
        user_prompt=state["user_prompt"],
    )
    messages = [
        SystemMessage(content=REVIEWER_NO_GENERATORS_SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]
    response = llm.invoke(messages)

    print("DEBUG metadata:", response.response_metadata)
    usage = response.response_metadata.get("token_usage", 
        response.response_metadata.get("usage_metadata", {}))

    return {
        "reviewer_output": response.content,
        "token_usage": {**state.get("token_usage", {}), "review": usage}
    }


def rewrite_no_generators(state:GRRAblationState) -> dict:
    """
    Rewriter node for grr_no_generators condition.
    Receives raw input data and reviewer feedback (no generated summaries).
    Produces final summary based on source data guided by bias feedback from Reviewer.
    """
    user_message = build_rewriter_no_generators_user_message(
        input_data=state["input_data"],
        reviewer_feedback=state["reviewer_output"],
        user_prompt=state["user_prompt"],
    )
    messages = [
        SystemMessage(content=REWRITER_NO_GENERATORS_SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]
    response = llm.invoke(messages)
    
    print("DEBUG metadata:", response.response_metadata)

    usage = response.response_metadata.get("token_usage", 
        response.response_metadata.get("usage_metadata", {}))
    return {
        "revised_summary": response.content,
        "token_usage": {**state.get("token_usage", {}), "rewrite": usage}
    }

### grr_no_genA ###
def review_no_genA(state:GRRAblationState) -> dict:
    """
    Reviewer for grr_no_genA: receives summaries B and C only.
    Summary A slot is marked absent in the user message.
    """
    user_message = build_reviewer_user_message_no_genA(
        summary_b=state["summary_b"],
        summary_c=state["summary_c"],
    )
    messages = [
        SystemMessage(content=REVIEWER_SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]
    response = llm.invoke(messages)

    print("DEBUG metadata:", response.response_metadata)

    usage = response.response_metadata.get("token_usage", 
        response.response_metadata.get("usage_metadata", {}))

    return {
        "reviewer_output": response.content,
        "token_usage": {**state.get("token_usage", {}), "review": usage}
    }

def rewrite_no_genA(state:GRRAblationState) -> dict:
    """
    Rewriter for grr_no_genA: receives summaries B and C + reviewer feedback.
    """
    user_message = build_rewriter_user_message_no_genA(
        summary_b=state["summary_b"],
        summary_c=state["summary_c"],
        reviewer_feedback=state["reviewer_output"],
    )
    messages = [
        SystemMessage(content=REWRITER_SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]
    response = llm.invoke(messages)
    print("DEBUG metadata:", response.response_metadata)

    usage = response.response_metadata.get("token_usage", 
        response.response_metadata.get("usage_metadata", {}))
    
    return {
        "revised_summary": response.content,
        "token_usage": {**state.get("token_usage", {}), "rewrite": usage}
    }

### grr_no_genB ###
def review_no_genB(state:GRRAblationState) -> dict:
    """
    Reviewer for grr_no_genB: receives summaries A and C only.
    Summary B slot is marked absent in the user message.
    """
    user_message = build_reviewer_user_message_no_genB(
        summary_a=state["summary_a"],
        summary_c=state["summary_c"],
    )
    messages = [
        SystemMessage(content=REVIEWER_SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]
    response = llm.invoke(messages)
    print("DEBUG metadata:", response.response_metadata)
    usage = response.response_metadata.get("token_usage", 
        response.response_metadata.get("usage_metadata", {}))

    return {
        "reviewer_output": response.content,
        "token_usage": {**state.get("token_usage", {}), "review": usage}
    }

def rewrite_no_genB(state:GRRAblationState) -> dict:
    """
    Rewriter for grr_no_genB: receives summaries A and C + reviewer feedback.
    """
    user_message = build_rewriter_user_message_no_genB(
        summary_a=state["summary_a"],
        summary_c=state["summary_c"],
        reviewer_feedback=state["reviewer_output"],
    )
    messages = [
        SystemMessage(content=REWRITER_SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]
    response = llm.invoke(messages)
    print("DEBUG metadata:", response.response_metadata)
    usage = response.response_metadata.get("token_usage", 
        response.response_metadata.get("usage_metadata", {}))
    return {
        "revised_summary": response.content,
        "token_usage": {**state.get("token_usage", {}), "rewrite": usage}
    }


### grr_no_genC ###
def review_no_genC(state:GRRAblationState) -> dict:
    """
    Reviewer for grr_no_genC: receives summaries A and B only.
    Summary C slot is marked absent in the user message.
    """
    user_message = build_reviewer_user_message_no_genC(
        summary_a=state["summary_a"],
        summary_b=state["summary_b"],
    )
    messages = [
        SystemMessage(content=REVIEWER_SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]
    response = llm.invoke(messages)
    print("DEBUG metadata:", response.response_metadata)
    usage = response.response_metadata.get("token_usage", 
        response.response_metadata.get("usage_metadata", {}))

    return {
        "reviewer_output": response.content,
        "token_usage": {**state.get("token_usage", {}), "review": usage}
    }

def rewrite_no_genC(state:GRRAblationState) -> dict:
    """
    Rewriter for grr_no_genC: receives summaries A and B + reviewer feedback.
    """
    user_message = build_rewriter_user_message_no_genC(
        summary_a=state["summary_a"],
        summary_b=state["summary_b"],
        reviewer_feedback=state["reviewer_output"],
    )
    messages = [
        SystemMessage(content=REWRITER_SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]
    response = llm.invoke(messages)
    print("DEBUG metadata:", response.response_metadata)
    usage = response.response_metadata.get("token_usage", 
        response.response_metadata.get("usage_metadata", {}))
    return {
        "revised_summary": response.content,
        "token_usage": {**state.get("token_usage", {}), "rewrite": usage}
    }


# -----------------------------------------------------------------------------------------------------------------
### Graph: grr_no_reviewer ###
def build_graph_no_reviewer():
    """
    GRR without Reviewer step.
    Generators fan out from START, all feed directly into Rewriter, then END.
    """
    builder = StateGraph(GRRAblationState)

    builder.add_node("generate_a", generate_a)
    builder.add_node("generate_b", generate_b)
    builder.add_node("generate_c", generate_c)
    builder.add_node("rewrite", rewrite_no_reviewer)

    builder.add_edge(START, "generate_a")
   
    builder.add_edge("generate_a", "generate_b")
    builder.add_edge("generate_b", "generate_c")
    builder.add_edge("generate_c", "rewrite")
    builder.add_edge("rewrite", END)

    return builder.compile()


# -----------------------------------------------------------------------------------------------------------------
### Graph: grr_no_generators ###
def build_graph_no_generators():
    """
    GRR without generator layer.
    Reviewer analyses raw input data directly, Rewriter produces final summary.
    Pipeline: START -> review -> rewrite -> END.
    No fan-out/fan-in needed since there are no parallel generators.
    """
    builder = StateGraph(GRRAblationState)

    builder.add_node("review", review_no_generators)
    builder.add_node("rewrite", rewrite_no_generators)

    builder.set_entry_point("review")
    builder.add_edge("review", "rewrite")
    builder.add_edge("rewrite", END)

    return builder.compile()

# -----------------------------------------------------------------------------------------------------------------
### Graph: grr_no_genA ###
def build_graph_no_genA():
    """
    GRR without Generator A (Communications Officer).
    Pipeline: START -> generate_b + generate_c (parallel) -> review -> rewrite -> END
    """
    builder = StateGraph(GRRAblationState)

    builder.add_node("generate_b", generate_b)
    builder.add_node("generate_c", generate_c)
    builder.add_node("review", review_no_genA)
    builder.add_node("rewrite", rewrite_no_genA)

    builder.add_edge(START, "generate_b")
    builder.add_edge("generate_b", "generate_c")

    builder.add_edge("generate_c", "review")

    builder.add_edge("review", "rewrite")
    builder.add_edge("rewrite", END)

    return builder.compile()

# -----------------------------------------------------------------------------------------------------------------
### Graph: grr_no_genB ###
def build_graph_no_genB():
    """
    GRR without Generator B (Regional Coordinator).
    Pipeline: START -> generate_a + generate_c (parallel) -> review -> rewrite -> END
    """
    builder = StateGraph(GRRAblationState)

    builder.add_node("generate_a", generate_a)
    builder.add_node("generate_c", generate_c)
    builder.add_node("review", review_no_genB)
    builder.add_node("rewrite", rewrite_no_genB)

    builder.add_edge(START, "generate_a")
    builder.add_edge("generate_a", "generate_c")

    builder.add_edge("generate_c", "review")

    builder.add_edge("review", "rewrite")
    builder.add_edge("rewrite", END)

    return builder.compile()

# -----------------------------------------------------------------------------------------------------------------
### Graph: grr_no_genC ###
def build_graph_no_genC():
    """
    GRR without Generator C (Protection & Inclusion).
    Pipeline: START -> generate_a + generate_b (parallel) -> review -> rewrite -> END
    """
    builder = StateGraph(GRRAblationState)

    builder.add_node("generate_a", generate_a)
    builder.add_node("generate_b", generate_b)
    builder.add_node("review", review_no_genC)
    builder.add_node("rewrite", rewrite_no_genC)

    builder.add_edge(START, "generate_a")
    builder.add_edge("generate_a", "generate_b")

    builder.add_edge("generate_b", "review")

    builder.add_edge("review", "rewrite")
    builder.add_edge("rewrite", END)

    return builder.compile()

# -----------------------------------------------------------------------------------------------------------------
### Run functions ###
def run_grr_no_reviewer(input_data: str, user_prompt: str = USER_PROMPT_1) -> dict:
    """
    Run GRR ablation - no reviewer
    evaluated_output key: revised_summary (rewriter output, produced without reviewer guidance)
    """
    graph = build_graph_no_reviewer()

    initial_state: GRRAblationState = {
        "input_data":      input_data,
        "user_prompt":     user_prompt,
        "summary_a":       "",
        "summary_b":       "",
        "summary_c":       "",
        "reviewer_output": "",   # never filled in this condition
        "revised_summary": "",
        "token_usage": {},
    }

    return graph.invoke(initial_state)


def run_grr_no_generators(input_data: str, user_prompt: str = USER_PROMPT_1) -> dict:
    """
    Run GRR ablation - no generators.
    Reviewer analyses raw input data directly, Rewriter writer final summary based on that.
    user_prompt is passed to both Reviewer and Rewriter since there are no Generator agents to receive and act on the task questions.
    evaluated_output key: "revised_summary".
    """
    graph = build_graph_no_generators()

    initial_state: GRRAblationState = {
        "input_data":      input_data,
        "user_prompt":     user_prompt,
        "summary_a":       "", # not used in this condition
        "summary_b":       "", # not used in this condition
        "summary_c":       "", # not used in this condition
        "reviewer_output": "",
        "revised_summary": "",  
        "token_usage": {},
    }

    return graph.invoke(initial_state)

def run_grr_no_genA(input_data: str, user_prompt: str = USER_PROMPT_1) -> dict:
    """
    Run GRR ablation without Generator A (Communications/Reporting Officer).
    Generators B (Regional Coordinator) and C (Protection & Inclusion) run in parallel.
    evaluated_output key: revised_summary.
    """
    graph = build_graph_no_genA()

    initial_state: GRRAblationState = {
        "input_data":      input_data,
        "user_prompt":     user_prompt,
        "summary_a":       "",   # not produced in this condition
        "summary_b":       "",
        "summary_c":       "",
        "reviewer_output": "",
        "revised_summary": "",
        "token_usage": {},
    }

    return graph.invoke(initial_state)

def run_grr_no_genB(input_data: str, user_prompt: str = USER_PROMPT_1) -> dict:
    """
    Run GRR ablation without Generators B (Regional Coordinator).
    Generators A (Communications/Reporting Officer) and C (Protection & Inclusion) run in parallel.
    evaluated_output key: revised_summary.
    """
    graph = build_graph_no_genB()

    initial_state: GRRAblationState = {
        "input_data":      input_data,
        "user_prompt":     user_prompt,
        "summary_a":       "",   
        "summary_b":       "",      # not produced in this condition
        "summary_c":       "",
        "reviewer_output": "",
        "revised_summary": "",
        "token_usage": {},
    }

    return graph.invoke(initial_state)

def run_grr_no_genC(input_data: str, user_prompt: str = USER_PROMPT_1) -> dict:
    """
    Run GRR ablation without C (Protection & Inclusion).
    Generators A (Communications/Reporting Officer) and Generator B (Regional Coordinator) run in parallel.
    evaluated_output key: revised_summary.
    """
    graph = build_graph_no_genC()

    initial_state: GRRAblationState = {
        "input_data":      input_data,
        "user_prompt":     user_prompt,
        "summary_a":       "",   
        "summary_b":       "",
        "summary_c":       "",      # not produced in this condition
        "reviewer_output": "",
        "revised_summary": "",
        "token_usage": {},
    }

    return graph.invoke(initial_state)