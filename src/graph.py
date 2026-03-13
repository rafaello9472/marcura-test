import base64
import logging
import operator
import time
import fitz

from typing import List, Annotated, Literal
from typing_extensions import TypedDict, NotRequired

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

from schemas import PageExtraction
from prompts import EXTRACTOR_SYSTEM_PROMPT
from utils import extract_clean_text_from_pdf, _is_struck_through

logger = logging.getLogger(__name__)

class DocumentState(TypedDict):
    """Represents the global state of the document extraction pipeline across pages."""
    pdf_path: str
    current_page: int
    end_page: int
    is_rider: bool
    page_text: NotRequired[str]
    page_image_base64: NotRequired[str]
    all_extractions: Annotated[List[PageExtraction], operator.add]
    final_clauses: NotRequired[List[dict]]
    error: NotRequired[str]


def get_page_image_base64(pdf_path: str, page_num: int) -> str:
    """Generates a base64 encoded image of a PDF page with struck-through text visually redacted.

    This prevents multimodal LLMs from reading physically voided text that remains visible
    underneath drawn strike-through lines.

    Args:
        pdf_path (str): Path to the PDF document.
        page_num (int): The page number to convert (1-indexed).

    Returns:
        str: A base64 encoded JPEG string of the redacted page image.
    """
    doc = fitz.open(pdf_path)
    page = doc[page_num - 1]
    
    horizontal_lines = []
    for d in page.get_drawings():
        for item in d.get("items", []):
            if item[0] == "l":
                p1, p2 = item[1], item[2]
                if abs(p1.y - p2.y) < 5 and abs(p1.x - p2.x) > 1:
                    rect = fitz.Rect(min(p1.x, p2.x), min(p1.y, p2.y), max(p1.x, p2.x), max(p1.y, p2.y))
                    horizontal_lines.append(rect)
            elif item[0] == "re":
                rect = item[1]
                if (rect.y1 - rect.y0) < 5 and (rect.x1 - rect.x0) > 1:
                    horizontal_lines.append(rect)
                    
    page_dict = page.get_text("dict")
    for block in page_dict.get("blocks", []):
        if "lines" not in block:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                text = span["text"]
                x0, y0, x1, y1 = span["bbox"]
                char_width = (x1 - x0) / max(len(text), 1)
                current_x = x0
                
                for char in text:
                    char_x1 = current_x + char_width
                    if char.strip():
                        char_rect = fitz.Rect(current_x - 1, y0, char_x1 + 1, y1)
                        if _is_struck_through(char_rect, horizontal_lines):
                            # Redact with white to hide from Vision LLM
                            page.add_redact_annot(char_rect, fill=(1, 1, 1))
                    current_x = char_x1
                    
    page.apply_redactions()
    
    pix = page.get_pixmap(dpi=300)
    img_bytes = pix.tobytes("jpeg")
    return base64.b64encode(img_bytes).decode("utf-8")


def preprocess_node(state: DocumentState) -> DocumentState:
    """Graph node responsible for extracting text and images from the current PDF page.

    Args:
        state (DocumentState): The current execution state.

    Returns:
        DocumentState: Updates the state with `page_text`, `page_image_base64`, and 
        the updated layout flag `is_rider`.
    """
    if state.get("error"):
        return {}
        
    page_num = state["current_page"]
    pdf_path = state["pdf_path"]
    current_is_rider = state.get("is_rider", False) 
    
    logger.info(f"Preprocessing Page {page_num}...")
    try:
        page_text, updated_is_rider = extract_clean_text_from_pdf(
            pdf_path, 
            start_page=page_num, 
            end_page=page_num, 
            is_rider=current_is_rider
        )
        page_image_b64 = get_page_image_base64(pdf_path, page_num)
        
        return {
            "page_text": page_text, 
            "page_image_base64": page_image_b64,
            "is_rider": updated_is_rider 
        }
    except Exception as e:
        error_msg = f"Preprocessing failed: {str(e)}"
        logger.error(f"Page {page_num} - {error_msg}")
        return {"error": error_msg}


# --- Initialize LLM ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.0, max_retries=3)
structured_llm = llm.with_structured_output(PageExtraction)

def extract_node(state: DocumentState) -> DocumentState:
    """Graph node responsible for invoking the multimodal LLM to parse clauses.

    Args:
        state (DocumentState): The current execution state.

    Raises:
        OutputParserException: If the LLM repeatedly fails to return the required Pydantic schema.

    Returns:
        DocumentState: Updates the state with extracted clauses and increments the page counter.
    """
    page_num = state["current_page"]
    header = f"--- PAGE {page_num} ---"
    content_only = state.get("page_text", "").replace(header, "").strip()
    
    if not content_only:
        logger.info(f"Page {page_num} is completely redacted or empty. Skipping LLM call.")
        return {
            "all_extractions": [PageExtraction(clauses=[])], 
            "current_page": page_num + 1
        }
    
    logger.info(f"Extracting clauses from Page {page_num}...")
    
    messages = [
        SystemMessage(content=EXTRACTOR_SYSTEM_PROMPT),
        HumanMessage(content=[
            {"type": "text", "text": f"Here is the cleaned text for Page {page_num}:\n\n{state['page_text']}"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{state['page_image_base64']}"}}
        ])
    ]
    
    max_retries = 3 
    
    try:
        for attempt in range(max_retries):
            try:
                result: PageExtraction = structured_llm.invoke(messages)
                return {"all_extractions": [result], "current_page": page_num + 1}
            except OutputParserException as e:
                logger.warning(f"Page {page_num} - Attempt {attempt + 1}/{max_retries} failed with OutputParserException. Retrying...")
                if attempt < max_retries - 1:
                    time.sleep(2) 
                else:
                    raise e

    except OutputParserException as e:
        logger.error(f"Page {page_num} - Could not parse LLM output after {max_retries} attempts: {e}. Skipping page.")
        failed_page_extraction = PageExtraction(clauses=[])
        return {"all_extractions": [failed_page_extraction], "current_page": page_num + 1}
    except Exception as e:
        error_msg = f"LLM Extraction failed with a critical error: {str(e)}"
        logger.error(f"Page {page_num} - {error_msg}")
        return {"error": error_msg}


def reconcile_node(state: DocumentState) -> DocumentState:
    """Graph node responsible for merging fragmented clauses across page breaks.

    Uses a hybrid approach (LLM signals + deterministic indexing) to identify and concatenate 
    paragraphs that span multiple pages into a single logical clause.

    Args:
        state (DocumentState): The current execution state.

    Returns:
        DocumentState: Updates the state with a unified list of final dictionaries.
    """
    logger.info("Reconciling fragmented clauses across pages...")
    final_clauses = []
    current_clause = None
    
    for page_ext in state.get("all_extractions", []):
        if not page_ext or not page_ext.clauses:
            continue
            
        for index, clause in enumerate(page_ext.clauses):
            llm_flag = clause.is_continued_from_previous_page
            is_first_on_page = (index == 0)
            deterministic_flag = is_first_on_page and not clause.id and not clause.title
            
            should_merge = current_clause is not None and (llm_flag or deterministic_flag)
            
            if should_merge:
                current_clause["text"] += " " + clause.text.strip()
            else:
                if current_clause:
                    final_clauses.append(current_clause)
                
                current_clause = {
                    "id": clause.id,
                    "title": clause.title,
                    "text": clause.text.strip()
                }
                
    if current_clause:
        final_clauses.append(current_clause)
        
    logger.info(f"Reconciliation complete. Total clauses: {len(final_clauses)}")
    return {"final_clauses": final_clauses}


def should_continue(state: DocumentState) -> Literal["preprocess", "reconcile"]:
    """Conditional edge router determining the next step in the state machine.

    Args:
        state (DocumentState): The current execution state.

    Returns:
        Literal["preprocess", "reconcile"]: The name of the next node to execute.
    """
    if state.get("error"):
        logger.warning("Error state detected. Routing directly to reconcile node.")
        return "reconcile"
        
    if state["current_page"] > state["end_page"]:
        return "reconcile"
    return "preprocess"

# --- Build and Compile the Graph ---
workflow = StateGraph(DocumentState)
workflow.add_node("preprocess", preprocess_node)
workflow.add_node("extract", extract_node)
workflow.add_node("reconcile", reconcile_node)
workflow.set_entry_point("preprocess")
workflow.add_edge("preprocess", "extract")
workflow.add_conditional_edges(
    "extract",
    should_continue,
    {
        "preprocess": "preprocess",
        "reconcile": "reconcile"
    }
)
workflow.add_edge("reconcile", END)
parser_app = workflow.compile()