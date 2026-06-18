"""
Runs baseline pipelines and GRR across dataset.
Tracks metadata and ablation conditions.
Results are saved to results/results.jsonl: one JSON object per run.
"""
# -----------------------------------------------------------------------------------------------------------------
### Imports ###
import pandas as pd
import json
import os
from datetime import datetime
from dotenv import load_dotenv
import time

from config import LLM_PROVIDER, LLM_MODEL_OPENAI, LLM_MODEL_LOCAL_OLLAMA, LLM_MODEL_GEMINI, LLM_MODEL_OLLAMA
from config import OPENAI_BASE_URL

from prompts import USER_PROMPT_1, USER_PROMPT_2
from baseline1_plain import run_baseline1
from baseline2_reflect import run_baseline2
## add GRR/MBG imports when that is set up ##
from grr_architecture import run_grr
from grr_ablations import run_grr_no_reviewer, run_grr_no_generators
from grr_ablations import run_grr_no_genA, run_grr_no_genB, run_grr_no_genC

load_dotenv() # loads API keys from .env file before code runs


# -----------------------------------------------------------------------------------------------------------------
### Config ###

DATA_PATH = "Thesis_data_final.xlsx"

BATCH_SIZE = 100

RESULTS_DIR = "results"
RESULTS_FILE = os.path.join(RESULTS_DIR, "results.jsonl")

# Pipelines to run -> comment out the one(s) you want to run (here now set to baselines)
PIPELINES = {
    "baseline1":        run_baseline1,
    "baseline2":        run_baseline2,
    # "grr_full":         run_grr,
    # "grr_no_reviewer":  run_grr_no_reviewer,
    # "grr_no_generators":  run_grr_no_generators,
    # "grr_no_genA": run_grr_no_genA,
    # "grr_no_genB": run_grr_no_genB,
    # "grr_no_genC": run_grr_no_genC,
}

# How many times to run each pipeline on each input.
N_RUNS = 3 

# User prompts to evaluate
USER_PROMPTS = {
    "prompt1": USER_PROMPT_1,
    "prompt2": USER_PROMPT_2,
}

# -----------------------------------------------------------------------------------------------------------------
### Data loading ###


def load_dataset(path: str) -> list[dict]:
    """
    Loads input data, randomly shuffles it, splits it into batches of 100.
    Each row becomes one feedback entry. 

    Returns a list with one dict, containing
    - "batch_id": label for this batch (used in result logging)
    - "input_data": all rows formatting as a single string, one row per line, with all column values included (not only just feedback text column)
    """

    df = pd.read_excel(path)

    # Shuffle dataset randomly, uses random_state=42 so we get exact same random batches each run (reproducibility)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Format every row as a single line: "Col1: val | Col2: val | ..."
    def format_row(row):
        # return " | ".join(f"{col}: {val}" for col, val in row.items())
        return " | ".join(
            f"{col}: {str(val).replace(chr(10), ' ').replace(chr(13), ' ')}"
            for col, val in row.items()
    )
    
    batches = []
    
    # Iterate through data in chunks of the batch size
    for i in range(0, len(df), BATCH_SIZE):
        chunk = df.iloc[i : i + BATCH_SIZE]

        # Format only rows in this chunk
        formatted_rows = chunk.apply(format_row, axis=1) # one string per row
        input_data = "\n".join(formatted_rows) # join all rows into one block

        # Create clean batch id
        batch_number = (i // BATCH_SIZE) + 1
        batch_id = f"batch_{str(batch_number).zfill(2)}"

        batches.append({
            "batch_id": batch_id,
            "input_data": input_data,
        })

    return batches

# -----------------------------------------------------------------------------------------------------------------
### Result loading/saving ###

def load_results() -> list[dict]:
    """
    Reads all results from JSONL file into list of dicts.
    """
    if not os.path.exists(RESULTS_FILE):
        return [] # empty if no results saved yet
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()] # parse each non-empty line as JSON object and return list

def save_result(result: dict):
    """
    Appends one result dict to JSONL file.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULTS_FILE, "a", encoding="utf-8") as f: # to not overwrite existing results
        f.write(json.dumps(result) + "\n") # serialise result dict to JSON string, write as new line

# -----------------------------------------------------------------------------------------------------------------
### Main running function ###

def run_all():
    # Check if results already exist for this model
    if os.path.exists(RESULTS_FILE):
        existing = load_results()
        existing_models = set(r.get("model", "unknown") for r in existing)
        print(f"WARNING: {RESULTS_FILE} already contains {len(existing)} results "
              f"from model(s): {existing_models}")
        confirm = input("Continue and append? (y/n): ")
        if confirm.strip().lower() != "y":
            print("Aborted.")
            return

    dataset = load_dataset(path = DATA_PATH) # CHANGE TO TEST ON SUBSETS

    run_counter = 0

    for pipeline_name, pipeline_fn in PIPELINES.items():
        for prompt_name, user_prompt in USER_PROMPTS.items():
            for run_index in range(1, N_RUNS + 1):
                for batch in dataset:
                    run_counter += 1
                    print( # print progress to terminal
                        f"[{run_counter}] pipeline={pipeline_name} | "
                        f"batch={batch['batch_id']} | "
                        f"prompt={prompt_name} | "
                        f"run={run_index}/{N_RUNS}"
                    )

                    try: # wrap pipeline call so single failure does not stop entire evaluation
                        t_start = time.time()
                        output = pipeline_fn( # call pipeline helper function with current batch + prompt
                            input_data=batch["input_data"],
                            user_prompt=user_prompt,
                        )
                        t_elapsed = round(time.time() - t_start, 4)

                        # Calculate total tokens from the token_usage dict in the output state
                        token_data = output.get("token_usage", {})

                        prompt_tokens = 0
                        completion_tokens = 0

                        for usage in token_data.values():
                            if isinstance(usage, dict):
                                meta = usage.get("token_usage", usage) # Handle if it's nested or flat
                                prompt_tokens += meta.get("prompt_tokens") or 0
                                completion_tokens += meta.get("completion_tokens") or 0


                        total_tokens = prompt_tokens + completion_tokens

                        print(f"✓ Done in {t_elapsed}s | Input: {prompt_tokens} | Output: {completion_tokens} | Total: {total_tokens}")

                        # For Baseline 1, the evaluated output is "summary".
                        # For Baseline 2, it is "revised_summary".
                        # For GRR it will be "revised_summary" too.

                        evaluated_output = (
                            output.get("summary") if pipeline_name == "baseline1"
                            else output.get("revised_summary") if pipeline_name in ("baseline2", "grr_full", "grr_no_reviewer", "grr_no_generators",
                                                                                    "grr_no_genA", "grr_no_genB", "grr_no_genC",)
                            else None
                        )

                        result = {
                            "run_id": run_counter,
                            "pipeline": pipeline_name,
                            "batch_id": batch["batch_id"],
                            "input_data": batch["input_data"],
                            "prompt_name": prompt_name,
                            "user_prompt": user_prompt,
                            "run_index": run_index,
                            "timestamp": datetime.now().isoformat(),
                            "output": output,                       # full state dict (all intermediate fields)
                            "evaluated_output": evaluated_output,   # specific field used for metrics 
                            "error": None,
                            "model": LLM_MODEL_OPENAI if LLM_PROVIDER == "openai" else
                                    LLM_MODEL_LOCAL_OLLAMA if LLM_PROVIDER == "ollama-local" else
                                    LLM_MODEL_GEMINI if LLM_PROVIDER == "gemini" else
                                    LLM_MODEL_OLLAMA if LLM_PROVIDER == "ollama" else
                                    "unknown",
                            "runtime_generation_seconds": t_elapsed,
                            "token_usage": {
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": completion_tokens,
                                "total_tokens": total_tokens
                            },
                            "n_requests": {
                                "baseline1": 1,
                                "baseline2": 2,
                                "grr_full": 5, # 3 generators, 1 reviewer, 1 rewriter
                                "grr_no_reviewer": 4, # 3 generators + rewriter
                                "grr_no_generators": 2, # reviewer + rewriter
                                "grr_no_genA": 4, # 2 generators + reviewer + rewriter
                                "grr_no_genB": 4, # 2 generators + reviewer + rewriter
                                "grr_no_genC": 4, # 2 generators + reviewer + rewriter
                            }.get(pipeline_name, None),
                        }
                    
                    except Exception as e: # catch errors
                        result = {
                            "run_id": run_counter,
                            "pipeline": pipeline_name,
                            "batch_id": batch["batch_id"],
                            "input_data": batch["input_data"],
                            "prompt_name": prompt_name,
                            "user_prompt": user_prompt,
                            "run_index": run_index,
                            "timestamp": datetime.now().isoformat(),
                            "output": None, # no output due to error
                            "error": str(e), # store error message string
                        }
                        print(f"   ERROR: {e}")
                    
                    save_result(result) # append result to results file
    print(f"\nDone. {run_counter} runs saved to {RESULTS_FILE}")   


# -----------------------------------------------------------------------------------------------------------------
### Loading back results for analysis ###


def results_for_pipeline(pipeline_name: str) -> list[dict]:
    """
    Convenience filter: get all results for specific pipeline.
    """
    return [r for r in load_results() if r["pipeline"] == pipeline_name] # only returns results where pipeline label matches

# -----------------------------------------------------------------------------------------------------------------
### Entry point ###
if __name__ == "__main__":
    run_all()