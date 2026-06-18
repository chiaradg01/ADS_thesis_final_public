"""
Post-hoc metric computation for all pipeline outputs saved in results/results.jsonl

Metrics:
- alignment_score: DeepEval SummarizationMetric (LLM-as-judge, OpenAI)
- coverage_score: DeepEval SummarizationMetric (LLM-as-judge, OpenAI)
- equal_coverage (EC): Summary-level fairness (Li et al., 2024), adapted from RoBERTa-large-MNLI (GPU) to a lighter NLI cross-encoder (nli-MiniLM2-L6-H768) for CPU compatibility
- hedging: using spaCy

Outputs new file: results/results_with_metrics.jsonl
Each line is one result dict from results.jsonl + added metrics key

Needs additional installs: deepeval sentence-transformers pandas
"""

# -----------------------------------------------------------------------------------------------------------------
### Imports ###
import json
import os
import re
import numpy as np
from dotenv import load_dotenv
import spacy
import csv
import time
from datetime import datetime, timedelta

from deepeval.test_case import LLMTestCase
from deepeval.metrics import SummarizationMetric
from deepeval.models import GPTModel
from deepeval.models import AzureOpenAIModel

from sentence_transformers import CrossEncoder

from config import LOCAL_NLI_MODEL_PATH_CONFIG, DEEPEVAL_JUDGE_MODEL, OPENAI_BASE_URL, NLI_MODEL_CONFIG, OPENAI_BASE_URL_JUDGE, AZURE_API_VERSION

load_dotenv()
nlp = spacy.load("en_core_web_sm")

# -----------------------------------------------------------------------------------------------------------------
### Config ###

RESULTS_FILE = os.getenv(
    "RESULTS_FILE",
    os.path.join("results", "results.jsonl")
)

OUTPUT_FILE = os.getenv(
    "OUTPUT_FILE",
    os.path.join("results", "results_with_metrics.jsonl")
)

SUMMARY_FILE = os.getenv(
    "SUMMARY_FILE",
    os.path.join("results", "metrics_summary.csv")
)

# Max results make it easier for small scale testing before large batch
MAX_RESULTS = os.getenv("MAX_RESULTS")
MAX_RESULTS = int(MAX_RESULTS) if MAX_RESULTS else None

# Fixed judge to ensure comparability across pipelines
JUDGE_MODEL = DEEPEVAL_JUDGE_MODEL

# NLI cross-encoder model for entailment probability estimation.
# Consistent with Li et al. (2024) who use RoBERTa-large-MNLI.
# This is a smaller NLI model that runs on CPU without GPU infrastructure.
NLI_MODEL = NLI_MODEL_CONFIG

# Local cache path for downloaded model
LOCAL_NLI_MODEL_PATH = LOCAL_NLI_MODEL_PATH_CONFIG

# Group columns to track for fairness metrics.
# keys = column name how it appears in input_data string
# values = list of know attribute values for that group
# lowercased to match how parse_input_data lowercases column names
GROUP_COLUMNS = {
    "sex":    [],
    "region": [],  
    "age group": [],
    "invalidity": [],
    "idp_status":   [],
    "relative_deceased_disappeared_war": [],
    "lonely_elderly": [],
}

# Entailment probability threshold: a summary sentence is considered to "cover"
# a feedback entry if the NLI entailment probability >= this value.
# Analogous to the 0.95 threshold in the original Li et al. (2024) code.
# Cross-encoders output a probability in [0,1] per class (contradiction/neutral/entailment),
# so this threshold is on the same scale as the original.
ENTAILMENT_THRESHOLD = 0.50  

# DeepEval Questions
DEEPEVAL_QUESTIONS = {
    "prompt1": [
        "Does the summary mention at least two distinct types of issues raised by community members?",
        "Does the summary reflect concerns from more than one demographic group (e.g., women, elderly, people with disabilities)?",
        "Does the summary avoid attributing concerns to only one geographic region when multiple regions are represented in the feedback?",
        "Does the summary include at least one issue raised by a minority or less frequently mentioned group?",
        "Does the summary state all groups' concerns with equal directness, without using more hedged language for some groups than others?",
    ],
    "prompt2": [
        "Does the summary provide exactly three recommendations?",
        "Is each recommendation traceable to a pattern or concern present in the feedback data?",
        "Do the recommendations address needs from more than one community subgroup?",
        "Does the summary avoid recommendations that could disproportionately benefit one group over another without justification from the data?",
        "Are all recommendations stated with equal confidence and specificity?",
    ],
}



# -----------------------------------------------------------------------------------------------------------------
### Loading results ###
def load_results() -> list[dict]:
    if not os.path.exists(RESULTS_FILE):
        raise FileNotFoundError(f"No results file found at {RESULTS_FILE}. Run run_evaluation.py first.")
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def save_result_with_metrics(result: dict):
    os.makedirs("results", exist_ok=True)
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")


# -----------------------------------------------------------------------------------------------------------------
### Added helper function to only get the revised summary, so it does not also run on the bias flags ###
def extract_summary_text(raw_output: str) -> str:
    """
    Extracts just the summary text from structured pipeline output.
    GRR/ablation outputs contain headers like REVISED SUMMARY:, BIAS FLAGS ADDRESSED:, etc.
    Baseline 1 output is plain text so the regex falls through and returns as-is.
    """
    if not raw_output or not raw_output.strip():
        return ""
    
    # Pattern
    match = re.search(
        r"(?:Revised Summary|REVISED SUMMARY|Summary|SUMMARY):\s*(.*?)(?:\n[A-Z][A-Za-z ]+:|$)",
        raw_output,
        re.DOTALL,
    )
    if match:
        extracted = match.group(1).strip()
        if extracted:          # only use if something was actually captured
            return extracted
    
    # Pattern for Baseline 2
    match = re.search(
        r"\*\*Revised Summary[^*]*\*\*:?\s*\n+(.*?)(?:\n\*\*|\Z)",
        raw_output,
        re.DOTALL,
    )
    if match:
        extracted = match.group(1).strip()
        if extracted:          # only use if something was actually captured
            return extracted
        
    # Baseline 1: return full output

    return raw_output.strip()

### Helper functions: parse input_data string into structured feedback entries ###
def parse_input_data(input_data:str) -> list[dict]:
    """
    Parses input_data string (one formatted row per line) back into list of dicts.
    Format expected: "Col1: val | Col2: val | ..."
    Inverse of format_row() in run_evaluation.py 
    EC needs to know which feedback entry belonged to which group.
    """
    entries = []
    for line in input_data.strip().split("\n"):
        if not line.strip():
            continue
        entry = {}
        parts = line.split(" | ")
        for part in parts:
            if ": " in part:
                key, val = part.split(": ", 1)
                entry[key.strip().lower()] = val.strip()
        if entry:
            entries.append(entry)
    return entries

def get_group_label(entry: dict, group_col: str, known_values: list) -> str | None:
    """
    Returns the group label for feedback entry for given group column.
    Returns None if column is not found in entry.
    If known_values empty, returns whatever value is in the entry (inferred).
    """
    val = entry.get(group_col.lower())
    if val is None:
        return None
    val = val.strip().lower()
    if known_values:
        # Match against know values (partial match allowed)
        for kv in known_values:
            if kv.lower() in val:
                return kv.lower()
        return None # value now in known list, skip entry for this group
    return val # inferred mode: return raw value

def get_document_texts(entries: list[dict]) -> list[str]:
    """
    Extracts feedback text from each entry.
    Tries common column names for feedback text column.
    """
    text_col_candidates = [
        "feedback translations",
        "feedback translation",
        "feedback",
        "text",
        "translation",
        "description",
        "Description",
    ]
    texts = []
    for entry in entries:
        text = None
        for col in text_col_candidates:
            if col in entry:
                text = entry[col]
                break
        if text:
            texts.append(text)
        else: # fallback: join all values (last resort)
            texts.append(" ".join(entry.values()))
    return texts

### helper function saving metrics in csv file###
def save_metrics_summary_csv(results: list[dict]):
    rows = []

    for result in results:
        metrics = result.get("metrics", {})
        hedging = metrics.get("hedging", {})
        deepeval = metrics.get("deepeval", {})
        ec = metrics.get("equal_coverage", {})

        row = {
            "run_id": result.get("run_id"),
            "pipeline": result.get("pipeline"),
            "batch_id": result.get("batch_id"),
            "prompt_name": result.get("prompt_name"),
            "run_index": result.get("run_index"),
            "model": result.get("model"),
            "alignment_score": deepeval.get("alignment_score"),
            "coverage_score": deepeval.get("coverage_score"),
            "summarization_score": deepeval.get("summarization_score"),
            "deepeval_reason": deepeval.get("deepeval_reason"),
            "hedge_count": hedging.get("hedge_count"),
            "hedge_frequency": hedging.get("hedge_frequency"),
            "matched_hedges": "; ".join(hedging.get("matched_hedges", [])),
            "runtime_generation_seconds": result.get("runtime_generation_seconds"),
            "rt_deepeval_seconds": deepeval.get("runtime_seconds"),
            "rt_ec_seconds": ec.get("runtime_seconds"),
            "rt_hedging_seconds": hedging.get("runtime_seconds"),
            "n_nli_pairs": ec.get("n_nli_pairs"),
            "n_requests": result.get("n_requests"),
        }

        for group_col in GROUP_COLUMNS:
            row[f"ec_{group_col.replace(' ', '_')}"] = ec.get(group_col, {}).get("ec_score")

        rows.append(row)

    if not rows:
        return

    os.makedirs(os.path.dirname(SUMMARY_FILE), exist_ok=True)

    with open(SUMMARY_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

# -----------------------------------------------------------------------------------------------------------------
### DeepEval metrics: alignment_score & coverage_score ###
def compute_deepeval_metrics(input_text: str, summary:str, assessment_questions: list[str] | None = None) -> dict:
    """
    Computes alignment_score and coverage_score using DeepEval's SummarizationMetric.
    
    Parameters:
    - input_text: original source text (full input_data string)
    - summary: summary to evaluate (evaluated_output from results)

    Returns dict with keys: alignment_score, coverage_score, summarization_score, reason
    """
    if not summary or not summary.strip():
        return {
            "alignment_score":     None,
            "coverage_score":      None,
            "summarization_score": None,
            "deepeval_reason":     "empty summary",
        }
    
    test_case = LLMTestCase(input=input_text, actual_output=summary) # container object that packages inputs for metric

    judge_model = AzureOpenAIModel(
        model=DEEPEVAL_JUDGE_MODEL,
        deployment_name=DEEPEVAL_JUDGE_MODEL,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=OPENAI_BASE_URL_JUDGE,
        api_version=AZURE_API_VERSION,
        temperature=0,
    )

    metric = SummarizationMetric(
        threshold= 0.5,
        model=judge_model,
        include_reason=True,
        assessment_questions=assessment_questions, # None = default
    )

    try:
        metric.measure(test_case)

        print("DeepEval score:", metric.score)
        print("DeepEval score breakdown:", metric.score_breakdown)
        print("DeepEval reason:", metric.reason)
        

        breakdown = metric.score_breakdown or {}

        return {
            "alignment_score": breakdown.get("Alignment"),
            "coverage_score": breakdown.get("Coverage"),
            "summarization_score": metric.score,
            "deepeval_reason": metric.reason,
            "score_breakdown": breakdown, # saves the full breakdown object
        }
    except Exception as e:
        return {
            "alignment_score":     None,
            "coverage_score":      None,
            "summarization_score": None,
            "deepeval_reason":     f"ERROR: {e}",
        }

# -----------------------------------------------------------------------------------------------------------------
### Li et al / Fairness metric: EC ###
# Adapted from https://github.com/leehaoyuan/coverage_fairness

# Load embedding model once at module level (avoid reloading per call)
_nli_model = None # loaded on first use

def _get_nli_model() -> CrossEncoder:
    """
    Returs NLI cross-encoder model, loading it on first call.
    Loads from local cache if available, otherwise downloads from HuggingFace.
    """
    global _nli_model
    if _nli_model is not None:
        return _nli_model

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading NLI cross-encoder model for EC metric on device: {device}")

    try:
        # First try loading local cached model
        _nli_model = CrossEncoder(LOCAL_NLI_MODEL_PATH_CONFIG, device = device)
        print("NLI model loaded from local folder.")
    except Exception:
        # If local model does not exist yet, download from HuggingFace
        print("Local NLI model not found. Downloading from HuggingFace...")
        _nli_model = CrossEncoder(NLI_MODEL_CONFIG, device=device)

        # Save locally so future runs do not need HuggingFace
        os.makedirs("models", exist_ok=True)
        _nli_model.save(LOCAL_NLI_MODEL_PATH_CONFIG)

        print("NLI model downloaded and saved locally.")
    return _nli_model

def split_summary_into_sentences(summary: str) -> list[str]:
    """
    Splits summary into individual sentences.
    Original paper: GPT-3.5 for decomposing compound sentences.
    Here: use simple sentence splitting as alternative.
    """
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', summary.strip())
    return [s.strip() for s in sentences if s.strip()]

def compute_entailment_matrix(doc_texts: list[str], summary_sentences: list[str]) -> np.ndarray:
    """
    Computes entailment probability matrix p(d_i, s_j) for all document-sentence pairs.
    Shape: (n_docs, n_summary_sentences)

    Uses an NLI cross-encoder to estimate the probability that a document chunk
    entails a summary sentence, consistent with Li et al. (2024).
    The cross-encoder scores each (doc, summary_sentence) pair directly and outputs
    per-class probabilities: [contradiction, neutral, entailment].
    We extract the entailment probability (index 2) for each pair.
    Values below ENTAILMENT_THRESHOLD are set to 0. 
    """
    if not doc_texts or not summary_sentences:
        return np.zeros((len(doc_texts), len(summary_sentences))), 0
    
    # Build all (doc, summary_sentence) pairs for batch scoring
    # CrossEncoder expects a list of [text_a, text_b] pairs
    # Convention from NLI: premise = document, hypothesis = summary sentence ("does the document entail this summary sentence?")
    pairs = [
        [doc, sent]
        for doc in doc_texts
        for sent in summary_sentences
    ]

    # scores shape: (n_pairs, 3)
    # columns are: contradiction, neutral, entailment
    print(f"Number of NLI pairs: {len(pairs)}")
    scores = _get_nli_model().predict(pairs, batch_size = 64, apply_softmax=True)
    print("NLI prediction finished")

    if scores.ndim == 1: # to ensure it does not silently breaks if n_pairs==1
        scores = scores.reshape(1, -1)

    # Extract entailment probabilities (column index 1)
    entailment_probs = scores[:, 1]

    # Reshape to (n_docs, n_summary_sentences)
    matrix = entailment_probs.reshape(len(doc_texts), len(summary_sentences))

    # Apply threshold
    matrix[matrix < ENTAILMENT_THRESHOLD] = 0.0

    return matrix, len(pairs) # shape: (n_docs, n_summary_sentences)

def compute_equal_coverage(
        doc_texts: list[str],
        doc_labels: list[str],
        summary: str,
        precomputed_matrix: np.ndarray | None = None,
        precomputed_sentences: list[str] | None = None,
) -> dict:
    """
    Computes Equal Coverage (EC) for single (document set, summary) pair.
    Formula: EC(D,S) = (1/K) * sum_k |p(d,s) - p(d,s|a=k)|
    Lower EC value = fairer summary

    Also returns per-group coverage probabilities and the overall coverage probability for interpretability.

    Parameters:
    - doc_texts: list of feedback text strings, one per feedback entry
    - doc_labels: list of group attribute values, one per feedback entry (same order as doc_texts)
    - summary: summary string to evaluate

    Returns dict with ec_score, overall_coverage_prob, per_group_coverage, group_counts
    """
    # Filter out entries with no label
    paired = [(t, l) for t, l in zip(doc_texts, doc_labels) if l is not None]
    if not paired:
        return {"ec_score": None, "reason": "no labeled documents found"}
    
    texts, labels = zip(*paired)
    texts = list(texts)
    labels = list(labels)

    summary_sentences = precomputed_sentences or split_summary_into_sentences(summary)
    if not summary_sentences:
        return {"ec_score": None, "reason": "empty summary after parsing"}
    
    n_sents = len(summary_sentences)

    # p(d_i, s_j): NLI entailment probability matrix, shape (n_docs, n_summary_sentences)
    # Reuse precomputed matrix if provided, otherwise compute fresh
    if precomputed_matrix is not None:
        # Select only rows corresponding to labeled entries
        all_labels_paired = [(i, l) for i, (t, l) in enumerate(zip(doc_texts, doc_labels)) if l is not None]
        labeled_indices = [i for i, l in all_labels_paired]
        entailment_matrix = precomputed_matrix[labeled_indices, :]
    else:
        entailment_matrix, _ = compute_entailment_matrix(texts, summary_sentences)

    n_docs = len(texts)

    # Overall coverage probability: p(d,s) = (1/|D||S|) * sum_i sum_j p(d_i, s_j) 
    p_overall = np.sum(entailment_matrix) / (n_docs * n_sents)

    # Per-group coverage probability: p(d,s|a=k) = (1/|D_k||S|) * sum_{d_i in D_k} sum_j p(d_i, s_j)
    unique_groups = sorted(set(labels))
    K = len(unique_groups)

    per_group_coverage = {}
    group_counts = {}
    for group in unique_groups:
        group_indices = [i for i, l in enumerate(labels) if l == group]
        group_counts[group] = len(group_indices)
        group_matrix = entailment_matrix[group_indices, :] # shape: (|D_k|, n_sents)
        p_group = np.sum(group_matrix) / (len(group_indices) * n_sents)
        per_group_coverage[group] = float(round(p_group, 6))
    
    # EC(D,S) = (1/K) * sum_k |p(d,s) - p(d,s|a=k)|
    ec_score = (1/K) * sum(
        abs(p_overall - per_group_coverage[g]) for g in unique_groups
    )
    return {
        "ec_score":               float(round(ec_score, 6)),
        "overall_coverage_prob":  float(round(p_overall, 6)),
        "per_group_coverage":     per_group_coverage,
        "group_counts":           group_counts,
        "n_summary_sentences":    n_sents,
    }

# -----------------------------------------------------------------------------------------------------------------
### HEDGING FUNCTION: identifies hedging language in summary to flag potential uncertainty in model output ###
#hedging lexicon
HEDGING_LEXICON = {
    "may",
    "might",
    "could",
    "possibly",
    "perhaps",
    "suggest",
    "appear",
    "seem",
    "likely",
    "unlikely",
    "sometimes",
    "often",
    "generally",
    "relatively",
    "potentially",
    "assume",
    "indicate",
    "imply",
    "unclear",
    "approximately",
    "around",
}

def compute_hedging_score(text: str) -> dict:
    """
    Computes hedging score using spaCy lemmatization.
    Returns:
    - total hedge count
    - normalized hedge frequency
    - matched hedge lemmas
    """

    if not text or not text.strip():
        return {
            "hedge_count": 0,
            "hedge_frequency": 0,
            "matched_hedges": [],
        }

    doc = nlp(text.lower())

    matched = []

    for token in doc:
        lemma = token.lemma_.lower()

        if lemma in HEDGING_LEXICON:
            matched.append(lemma)

    total_words = len([
        t for t in doc
        if t.is_alpha
    ])

    hedge_count = len(matched)

    hedge_frequency = (
        hedge_count / total_words
        if total_words > 0 else 0
    )

    return {
        "hedge_count": hedge_count,
        "hedge_frequency": round(hedge_frequency, 4),
        "matched_hedges": matched,
    }


# -----------------------------------------------------------------------------------------------------------------
### Main computation loop ###
def compute_all_metrics():
    # Begin time
    start_time_metrics = datetime.now()

    print("=" * 80)
    print(f"STARTED: {start_time_metrics.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    results = load_results()

    if MAX_RESULTS is not None:
        results = results[:MAX_RESULTS]
        print(f"Testing on first {MAX_RESULTS} results.")

    print(f"Loaded {len(results)} results from {RESULTS_FILE}")

    enriched_results = []

    for i, result in enumerate(results):
        if i > 0:
            elapsed_time_metrics = datetime.now() - start_time_metrics
            avg_seconds_metrics = elapsed_time_metrics.total_seconds() / i
            remaining = avg_seconds_metrics * (len(results) - i)

            eta_time = datetime.now() + timedelta(seconds=remaining)


            print(
                f"\n[{i+1}/{len(results)}] pipeline={result['pipeline']} | "
                f"prompt={result['prompt_name']} | "
                f"run={result['run_index']} | batch={result['batch_id']} |"
                f"ETA: {eta_time.strftime('%H:%M:%S')} "
                f"({remaining/60:.1f} min left)"
            )
        else:
            print(f"[1/{len(results)}] Starting...")

        metrics = {}

        input_data = result.get("input_data", "")
        evaluated_output_raw = result.get("evaluated_output", "")
        evaluated_output = extract_summary_text(evaluated_output_raw)
        print(f"Extracted text: {repr(evaluated_output[:200])}") # to check if it is not empty

        # Skip metrics for refusal/empty outputs
        word_count = len(evaluated_output.split())
        if word_count < 50:
            print(f"Skipping metrics: output too short ({word_count} words), likely a refusal.")
            result["metrics"] = {"skipped": f"output too short ({word_count} words)"}
            enriched_results.append(result)
            save_result_with_metrics(result)
            continue

        #add hedging metric
        print("Computing hedging score...")
        t_hedge_start = time.time()
        hedging_result = compute_hedging_score(evaluated_output)
        hedging_result["runtime_seconds"] = round(time.time() - t_hedge_start, 4)
        metrics["hedging"] = hedging_result
        print(f"hedges={hedging_result['hedge_count']} | "
              f"freq={hedging_result['hedge_frequency']}"
              f"matched={hedging_result['matched_hedges']}")
        pipeline = result["pipeline"]

        if not input_data:
            print("WARNING: no input_data found, skipping metrics.")
            result["metrics"] = {"error": "no input_data"}
            enriched_results.append(result)
            continue

        # Parse feedback entries for EC
        entries = parse_input_data(input_data)
        doc_texts = get_document_texts(entries)

        # DeepEval: alignment + coverage
        print("Computing DeepEval alignment/coverage scores...")
        questions = DEEPEVAL_QUESTIONS.get(result["prompt_name"]) # None if prompt name not in dict
        t_deepeval_start = time.time()
        deepeval_scores = compute_deepeval_metrics(input_data, evaluated_output, assessment_questions=questions)
        deepeval_scores["runtime_seconds"] = round(time.time() - t_deepeval_start, 4)
        
        metrics["deepeval"] = deepeval_scores
        
        print(
            f"alignment={deepeval_scores['alignment_score']} | "
            f"coverage={deepeval_scores['coverage_score']} | "
            f"summarization={deepeval_scores['summarization_score']}"
        )


        # Equal Coverage (EC) per group column
        metrics["equal_coverage"] = {}

        t_ec_start = time.time()

        # Compute entailment matrix once, reuse across all group columns
        n_nli_pairs = 0 # default if summary is empty
        summary_sentences = split_summary_into_sentences(evaluated_output)
        if summary_sentences:
            entailment_matrix, n_nli_pairs = compute_entailment_matrix(doc_texts, summary_sentences)
        else:
            entailment_matrix = None
        
        for group_col, known_values in GROUP_COLUMNS.items():
            doc_labels = [get_group_label(e, group_col, known_values) for e in entries]
            labeled_count = sum(1 for l in doc_labels if l is not None)
            if labeled_count < 2:
                print(f"   SKIP EC for '{group_col}': fewer than 2 labeled entries.")
                metrics["equal_coverage"][group_col] = {"ec_score": None, "reason": "too few labeled entries"}
                continue

            print(f"   Computing Equal Coverage for group '{group_col}'...")
            ec_result = compute_equal_coverage(doc_texts, doc_labels, evaluated_output,
                                               precomputed_matrix=entailment_matrix,
                                               precomputed_sentences=summary_sentences)
            metrics["equal_coverage"][group_col] = ec_result
            print(f"   EC({group_col})={ec_result.get('ec_score')}")
        metrics["equal_coverage"]["runtime_seconds"] = round(time.time() - t_ec_start, 4)
        metrics["equal_coverage"]["n_nli_pairs"] = n_nli_pairs if summary_sentences else 0
        


            
        
        result["metrics"] = metrics
        enriched_results.append(result)
        save_result_with_metrics(result) # save after each result
        print(f"Saved result {i+1}/{len(results)} to {OUTPUT_FILE}")

    

    print(f"Done. {len(enriched_results)} results saved to {OUTPUT_FILE}")

    save_metrics_summary_csv(enriched_results)
    print(f"Metric summary written to: {SUMMARY_FILE}")


    # Print summary table
    print("\n=== METRIC SUMMARY ===")
    for result in enriched_results:
        m = result.get("metrics", {})
        de = m.get("deepeval", {})
        ec = m.get("equal_coverage", {})
        ec_str = " | ".join(
            f"EC({col})={ec.get(col, {}).get('ec_score')}"
            for col in GROUP_COLUMNS
        )
        print(
            f"pipeline={result['pipeline']:12} | prompt={result['prompt_name']} | run={result['run_index']} | batch={result['batch_id']} | "
            f"alignment={de.get('alignment_score')} | coverage={de.get('coverage_score')} | "
            f"reason={de.get('deepeval_reason', 'N/A')[:50]}... | " # Truncated reason for readability
            f"breakdown={de.get('score_breakdown')} | "
            f"{ec_str} | "
            f"hedges={m.get('hedging', {}).get('hedge_count')} | "
            f"hedge_words={m.get('hedging', {}).get('matched_hedges')} | "
            f"rt_generation={result.get('runtime_generation_seconds')}s | "
            f"rt_deepeval={de.get('runtime_seconds')}s | "
            f"rt_ec={ec.get('runtime_seconds')}s | "
            f"rt_hedging={m.get('hedging', {}).get('runtime_seconds')}s"
        )
    
    end_time_metrics = datetime.now()
    duration_metrics = end_time_metrics - start_time_metrics

    print("\n" + "=" * 80)
    print(f"FINISHED: {end_time_metrics.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"TOTAL TIME: {duration_metrics}")
    print("=" * 80)
        


# -----------------------------------------------------------------------------------------------------------------
### Entry point ###

if __name__ == "__main__":
    compute_all_metrics()