# Voyage Charter Party Parser

A multimodal LLM pipeline designed to deterministically extract legal clauses from Voyage Charter Party documents. Built for the Marcura Staff AI Engineer Challenge.

## Overview
Parsing legal PDFs with standard OCR or text-extraction tools often fails due to complex layouts, margin notes, and manual redactions. This solution implements a **Cyclic State Machine** using LangGraph and Gemini 2.5 Pro (Multimodal). Rather than passing the entire document context at once, the graph processes the document page-by-page. This ensures deterministic state management, prevents context-window degradation, and enables precise handling of cross-page clause fragmentation.

## Key Technical Highlights

To achieve production-grade reliability, this pipeline implements several advanced preprocessing and state-management techniques:

* **Character-Level Strikethrough Detection:** Standard PDF parsers group text into spans, meaning a single struck-through character can cause an entire sentence to be dropped. This pipeline unpacks text spans and uses character-level coordinate math with strict vertical/horizontal tolerances to surgically remove *only* legally voided text.
* **Multimodal Hallucination Prevention:** Multimodal LLMs are incredibly adept at reading struck-through text visually, which often causes them to hallucinate voided text back into the payload. To prevent this, the Python preprocessor physically draws white redaction boxes over struck-through text on the image payload before passing it to Gemini, perfectly synchronizing the text and visual contexts.
* **Defense in Depth (Reconciliation):** Relying purely on an LLM to flag multi-page clause continuations is risky due to non-determinism. The pipeline utilizes a hybrid Reconciler Node: it listens to the LLM's boolean continuation flag, but falls back on deterministic Python logic (e.g., merging fragments that appear first on a page with no ID/Title).
* **Dynamic Layout State Tracking:** The required target range (Pages 6–39) contains distinct structural shifts. Standard Shellvoy templates use far-left margin titles, whereas custom Rider clauses abandon this layout. A state tracker (`is_rider`) actively watches for transition headers and dynamically toggles layout-specific parsing rules for subsequent pages.

## Architecture

The LangGraph state machine consists of the following nodes:
1. **Preprocess Node**: Uses `PyMuPDF` to evaluate coordinates, strip line numbers, handle strikethroughs, inject `[MARGIN_TITLE]` hints, and generate a visually redacted base64 image of the page.
2. **Extract Node**: A multimodal LLM call (cleaned text + redacted image) that extracts top-level clauses into a strict Pydantic schema.
3. **Conditional Router**: Evaluates the state to either loop back to `Preprocess`, or break to the `Reconcile` node (upon completion or if a critical error is flagged).
4. **Reconcile Node**: A deterministic Python function that merges fragmented clauses across page breaks and formats the final JSON.

## Data Schema & Business Logic
The pipeline's data contracts (`schemas.py` & `prompts.py`) are strictly tailored to the business constraints of legal document extraction:
* **Exact Document IDs**: The `id` field maps strictly to the printed number to preserve legal referencing. Because struck-through clauses are physically removed during preprocessing, extracted IDs correctly jump (e.g., from 5 to 7).
* **Top-Level Extraction**: Extracts only top-level parent clauses. Nested sub-paragraphs (e.g., (i), (a)) are seamlessly preserved within the parent's `text` string to maintain context.
* **Missing Titles**: If a clause contains a printed ID but no explicit text title, the `title` field correctly returns `null`.
* **Hybrid Error Handling**: 
  * *Graceful Degradation:* If the LLM repeatedly fails to parse a specific page into the schema, the pipeline logs the failure and skips to the next page to salvage the rest of the document.
  * *Fail-Fast:* If a critical system or API error occurs, the state machine halts immediately to prevent silent data loss, reconciles what it has, and injects a `SYSTEM_ERROR` clause into the JSON to definitively alert downstream systems.

## Setup & Installation

### Prerequisites
* [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
* Google Gemini API Key (`GOOGLE_API_KEY`)
* (Optional) LangSmith API Key for execution tracing.

### Installation
1. Clone the repository and navigate to the directory:
```bash
git clone https://github.com/rafaello9472/marcura-test.git
cd marcura-test
```

2. Create and activate the Conda environment:
```bash
conda create -n marcura-env python=3.12 -y
conda activate marcura-env
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables. Create a `.env` file in the root directory:
```
export GOOGLE_API_KEY="your_google_api_key_here"

# Optional: LangSmith tracing
export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_API_KEY="your_langsmith_key_here"
export LANGCHAIN_PROJECT="marcura_parser"
```

## Usage
For convenience, the target PDF is already located in the `data/` directory. You can run the main orchestration script directly:

```
python src/main.py
```

### Output
The script logs page-by-page execution state to stdout. Upon completion, it outputs an `extracted_clauses.json` file containing the reconciled array of extracted clauses.

## Performance & Trade-offs

**Execution Time:** Processing the full document range (Pages 6–39) takes approximately **15 minutes**. 

For this technical challenge, the architectural focus was strictly on **accuracy, determinism, and edge-case handling** (e.g., character-level coordinate math and multimodal hallucination prevention) rather than raw speed. 

In a true production environment, if lower latency is a strict business requirement, this pipeline could be optimized significantly. Potential production optimizations include:
* **Parallelization:** Processing multiple pages concurrently (map-reduce) rather than sequentially.
* **Model Routing:** Using a smaller, faster model (e.g., Gemini 2.5 Flash) for standard pages, and routing to the heavier Pro model only for complex pages containing redactions or layout shifts.
* **Token Optimization:** Cropping the image payloads to only the bounding boxes of the text rather than sending the full 300dpi page.