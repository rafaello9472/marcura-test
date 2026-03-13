import json
import os
import logging
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
from graph import parser_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def process_document(pdf_path: str, start_page: int = 6, end_page: int = 39, output_file: str = "output/extracted_clauses.json") -> None:
    """Orchestrates the document extraction pipeline via LangGraph.

    Initializes the state machine, streams node execution events to update a progress bar,
    handles error contingencies, and writes the final structured output to a JSON file.

    Args:
        pdf_path (str): Path to the target PDF document.
        start_page (int, optional): The page number to begin extraction (1-indexed). Defaults to 6.
        end_page (int, optional): The page number to end extraction (inclusive, 1-indexed). Defaults to 39.
        output_file (str, optional): The file path where the resulting JSON will be saved. Defaults to "output/extracted_clauses.json".
    """
    logger.info(f"Initializing Graph for {pdf_path} (Pages {start_page}-{end_page})")
    
    initial_state = {
        "pdf_path": pdf_path,
        "current_page": start_page,
        "end_page": end_page,
        "is_rider": False,
        "all_extractions": []
    }
    
    total_pages = end_page - start_page + 1
    final_state = initial_state.copy()
    
    with logging_redirect_tqdm():
            with tqdm(total=total_pages, desc="Extracting Pages", unit="page") as pbar:
                for event in parser_app.stream(initial_state):
                    
                    if "extract" in event:
                        pbar.update(1)
                    
                    for node_name, state_update in event.items():
                        final_state.update(state_update)

    final_clauses = final_state.get("final_clauses", [])
    
    if final_state.get("error"):
        logger.error(f"PIPELINE HALTED DUE TO ERROR: {final_state['error']}")
        logger.warning("The output JSON will contain partial data and an error flag.")
        
        final_clauses.insert(0, {
            "id": "SYSTEM_ERROR",
            "title": "INCOMPLETE EXTRACTION",
            "text": f"The parser encountered a critical error and halted early. Partial data follows. Error details: {final_state['error']}"
        })
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_clauses, f, indent=4, ensure_ascii=False)
        
    logger.info(f"Finished! Output saved to {output_file}")

if __name__ == "__main__":
    if not os.environ.get("GOOGLE_API_KEY"):
        logger.error("GOOGLE_API_KEY is missing from environment variables!")
    else:
        # As per requirements, process pages 6-39
        process_document("data/voyage-charter-example.pdf", start_page=6, end_page=39)