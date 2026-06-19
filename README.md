# Bias Mitigation in Large Language Models using a Multi-Agent Generate-Review-Rewrite (GRR) Architecture

This repository contains the code used in my Applied Data Science master's thesis *"Bias Mitigation in Large Language Models using a Multi-Agent Generator-Reviewer Architecture"* (Utrecht University, in cooperation with the Netherlands Red Cross). \
It implements and evaluates a Generate-Review-Rewrite (GRR) multi-agent pipeline for bias-aware summarisation of humanitarian community feedback, benchmarked against a zero-shot and a self-reflecting single-agent baseline, across GPT-4.1 and Llama 3.1 8B (adjusted version with a 32k context window).

## Repository Structure
```
.
├── config.py                  # NOT INCLUDED - see below
├── .env                       # NOT INCLUDED - see "API Keys" below
├── prompts.py                 # All prompt strings and prompt-building 
├── baseline1_plain.py         # Baseline 1: zero-shot single-agent pipeline
├── baseline2_reflect.py       # Baseline 2: self-reflecting single-agent 
├── grr_architecture.py        # Full GRR pipeline
├── grr_ablations.py           # GRR ablation variants
├── run_evaluation.py          # Manages generation runs across all conditions
├── compute_metrics.py         # Post-hoc metric computation (alignment, coverage, EC, hedging)
├── analysis.ipynb             # Aggregation, statistical testing and plotting
└── EDA_wrangling_code_thesis.ipynb  # EDA, data wrangling
```

## File Descriptions
### `prompts.py`
Contains all prompt strings and prompt-builder functions used across all architectures, so that prompt design is kept separate from pipeline logic. Contains:
- The shared humanitarian system prompt (`SYSTEM_PROMPT`) and the two user task prompts (`USER_PROMPT_1`, `USER_PROMPT_2`), used by every pipeline.
- The Baseline 2 self-reflection prompt (`SELF_REFLECTION_PROMPT`).
- The three GRR Generator persona prompts (`GENERATOR_A/B/C_SYSTEM_PROMPT`): Communications/Reporting Officer (A), Regional Coordinator (B), and Protection & Inclusion Officer (C).
- The Reviewer and Rewriter system prompts (`REVIEWER_SYSTEM_PROMPT`, `REWRITER_SYSTEM_PROMPT`) and the corresponding `build_*_user_message()` functions that assemble their multi-part input messages.
- Equivalent prompts and message builders for every ablation condition (no_reviewer, no_generators, no_genA/B/C).

This file has no dependencies on the other project files and is imported by every pipeline script.

### `baseline1_plain.py`
Implements **Baseline 1**: a single-node LangGraph pipeline (`generate → END`) that calls the LLM once with the shared system prompt and a user prompt, with no bias-mitigation intervention. Defines `Baseline1State`, builds the graph in `build_graph()`, and wraps everything together into a final run function: `run_baseline1(input_data, user_prompt)`.

### `baseline2_reflect.py`
Implements **Baseline 2**: a two-node pipeline (`generate → reflect → END`) that extends Baseline 1 with a second conversational turn. The model's own first summary is passed back to it (full conversation history preserved) along with `SELF_REFLECTION_PROMPT`, asking it to assess and revise the summary across the four bias dimensions. Defines `Baseline2State` and `run_baseline2(input_data, user_prompt)`, which returns both `summary` (pre-reflection) and `revised_summary` (post-reflection, used later for evaluation).

### `grr_archicture.py`
Implements the **full GRR pipeline**: three Generator nodes (`generate_a`, `generate_b`, `generate_c`, run sequentially for local hardware/VRAM reasons) feed into a `review` node, which feeds into a `rewrite` node. Defines `GRRState` and `run_grr(input_data, user_prompt)`, which returns outputs for each node (`summary_a/b/c`, `reviewer_output`, `revised_summary`). `revised_summary` is used later for evaluation.

### `grr_ablations.py`
Implements all **ablation conditions**, reusing nodes from `grr_architecture.py` where possible and sharing the `GRRAblationState` TypedDict:
- `grr_no_reviewer`: generators feed directly into the Rewriter, skipping the Reviewer step.
- `grr_no_generators`: Reviewer analyses the raw input data directly, Rewriter produces the final summary from that.
- `grr_no_genA` / `grr_no_genB` / `grr_no_genC`: each removes one Generator persona, keeping the other two plus Reviewer and Rewriter.

Each condition has its own `build_graph_*()` and `run_grr_*()` function, all following the same input/output structure as `run_grr()`.

### `run_evaluation.py`
The script that actually runs the generation part. It:
- Loads and formats the dataset (`load_dataset()`), shuffling with `random_state=42` and splitting into batches of 100 rows for reproducibility (36 batches for this thesis).
- Defines the `PIPELINES` dict (mapping pipeline name → run function from the files above → (un)comment which pipelines to run here) and the `USER_PROMPTS` dict.
- Loops over _pipeline x prompt x run-index x batch_, calls the relevant pipeline function, extracts token usage from the LangChain response metadata, and writes one JSON line per run to `results/results.jsonl` via `save_result()`.
- Provides `results_for_pipeline()` as a convenience filter for downstream loading.

This file should be run directly (`python run_evaluation.py`) to generate raw pipeline outputs.

### `compute_metrics.py`
Reads `results/results.jsonl` and computes all evaluation metrics post-hoc, writing an enriched `results/results_with_metrics.jsonl` (one line appended per processed result, so the script is resumable/inspectable mid-run). Computes:
- **Hedging score**: spaCy-lemmatised matching against a fixed hedging lexicon (`compute_hedging_score()`).
- **Alignment & coverage scores**: via DeepEval's `SummarizationMetric`, using a fixed Azure-hosted judge model (GPT-5.4-mini) (`compute_deepeval_metrics()`).
- **Equal Coverage (EC)**: the fairness metric from Li et al. (2024), adapted to a CPU/GPU-light NLI cross-encoder (`cross-encoder/nli-MiniLM2-L6-H768`) instead of the original RoBERTa-large-MNLI, computed per demographic group column defined in `GROUP_COLUMNS` (`compute_equal_coverage()`, `compute_entailment_matrix()`).
- Supporting parsing utilities (`extract_summary_text()`, `parse_input_data()`, `get_group_label()`, `get_document_texts()`) that reverse-engineer the structured pipeline outputs and the formatted input-data string back into usable text/group labels.

Run this after `run_evaluation.py` has produced `results/results.jsonl` (`python compute_metrics.py`).

### `analysis.ipynb`
Reads `results/results_with_metrics.jsonl`, performs the batch-level aggregation, normality checks, Welch's t-tests / Mann-Whitney U tests with Benjamini-Hochberg correction, and produces all figures and tables used in the thesis.

### `EDA_wrangling_code_thesis.py`
Reads the input data and produces EDA plots used in the thesis.

## Why `config.py` is not included
`config.py` is the file where the LLM provider, model names, and API base URLs are set, and is imported by every pipeline script (`LLM_PROVIDER`, `LLM_MODEL_*`, `OPENAI_BASE_URL`, `TEMPERATURE`, `MAX_TOKENS`, etc.). It is excluded from this repository because it contains the Netherlands Red Cross's internal Azure OpenAI deployment names and resource endpoint, which are not meant to be public.

To reproduce the code, you should create your own `config.py` in the project root with the same structure, for example:
```python
LLM_PROVIDER = "openai"   # or "gemini" / "ollama" / "ollama-local"

LLM_MODEL_GEMINI = "gemini-2.5-flash" # used for testing
LLM_MODEL_OLLAMA = "llama3.1" # original local llama model
LLM_MODEL_OPENAI = "your-openai-deployment-name"
DEEPEVAL_JUDGE_MODEL = "your-judge-deployment-name"
LLM_MODEL_LOCAL_OLLAMA = "llama3.1-32k"   # see Appendix D of the thesis how to create this model

OPENAI_BASE_URL = "https://your-endpoint.openai.azure.com/openai/v1/"
OPENAI_BASE_URL_JUDGE = "https://your-endpoint.openai.azure.com/"
AZURE_API_VERSION = "2024-12-01-preview"

LOCAL_NLI_MODEL_PATH_CONFIG = "models/nli-MiniLM2-L6-H768" # For EC
NLI_MODEL_CONFIG = "cross-encoder/nli-MiniLM2-L6-H768" # For EC

TEMPERATURE = 0.0
MAX_TOKENS = 1024
```

## API keys: `.env`
All scripts call `load_dotenv()` (from `python-dotenv`) at import time to load API credentials (e.g. `OPENAI_API_KEY`) from a local `.env` file before instantiating the LLM. \
This file is excluded for the obvious reason that it contains personal API keys, and committing it would expose them publicly. You thus need to create your own `.env` in the project root, containing e.g.:

```
OPENAI_API_KEY=your-key-here
```

(plus any other provider keys required depending on which `LLM_PROVIDER` you select/include in `config.py`, e.g. a Google API key for `gemini`).


## How to use the code
### 1. Install dependencies
The project relies on `langchain`, `langgraph`, `langchain-openai`, `langchain-google-genai`, `langchain-ollama`, `python-dotenv`, `pandas`, `openpyxl`, `deepeval`, `sentence-transformers`, `spacy` (with the `en_core_web_sm` model), `numpy`, `scipy`, and `torch`. Install these in your environment before running anything. \
This repository also includes a `requirements.txt` file with all of these included.

### 2. Create `config.py` and `.env`
As described above. `LLM_PROVIDER` in `config.py` controls which model every pipeline script uses, so this must be set correctly before running evaluations.

### 3. Provide the dataset
The dataset itself (Ukrainian Red Cross community feedback, see thesis Chapter 3) is **not included** in this repository, as it was provided under a data-sharing agreement with the Netherlands Red Cross and is not meant to be public. `run_evaluation.py` expects an Excel file at the path set in `DATA_PATH`, with columns matching those described in the thesis (`APPLICATION NO.:`, `Feedback`, `Sector_list`, `Region`, `Sex`, `Age group`, `Invalidity`, `IDP_Status`, `Relative_Deceased_Disappeared_War`, `Lonely_Elderly`).

### 4. Run the pipelines (generation)
```bash
python run_evaluation.py
```
Edit the `PIPELINES` dict at the top of `run_evaluation.py` to select which architectures to run (`baseline1`, `baseline2`, `grr_full`, `grr_no_reviewer`, `grr_no_generators`, `grr_no_genA/B/C`), and `N_RUNS` for the number of repetitions per batch/prompt. This writes results to `results/results.jsonl`.

### 5. Compute metrics
```bash
python compute_metrics.py
```
Reads `results/results.jsonl` and writes `results/results_with_metrics.jsonl`, with alignment, coverage, Equal Coverage and hedging scores added per result. The NLI cross-encoder model is downloaded once and cached locally under `models/` on the first run.

### 6. Run the analysis
Open `analysis.ipynb` and run it against `results/results_with_metrics.jsonl` to reproduce the statistical tests and figures.

**Folder structure used for this thesis:**
```
FINAL_RESULTS_FOLDER/ \
    results_baseline_gpt41/results_with_metrics_baselines_gpt41.jsonl
    results_baseline_llama32k/results_with_metrics_baselines_llama32k.jsonl
    results_grr_full_gpt41/results_with_metrics_grr_full_gpt41.jsonl
    results_grr_full_llama32k/results_with_metrics_grr_full_llama32k.jsonl
    results_grr_no_genA_llama32k/results_with_metrics_grr_no_genA_llama32k
    ...
```
- Each subfolder contains one results_with_metrics_[pipeline]_[model].jsonl file. 
- Baselines are in the same run per model. Pipeline field inside each row distinguished them

### Dependency graph
```
config.py ──┐
            ├──> baseline1_plain.py ───┐
prompts.py ─┤                          │
            ├──> baseline2_reflect.py ─┤
            ├──> grr_architecture.py ──┼──> run_evaluation.py ──> results.jsonl
            └──> grr_ablations.py ─────┘                              │
                                                                      v
                                compute_metrics.py ──> results_with_metrics.jsonl
                                                                      │
                                                                      v
                                                                analysis.ipynb
```

## Citation
For now, this is an unpublished Master's thesis. \
In the future, if you use this code or anything from the written thesis, please cite it: \
`de Groot, C. (2026). Bias Mitigation in Large Language Models using a Multi-Agent Generator-Reviewer Architecture. Master's thesis, Utrecht University.`
