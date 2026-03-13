import fitz  
import logging
import re

logger = logging.getLogger(__name__)

# --- Heuristic Constants ---
RIGHT_MARGIN_THRESHOLD = 0.85 # Percentage of page width to consider as right margin
LEFT_MARGIN_THRESHOLD = 0.15  # Percentage of page width to consider as left margin

def _is_right_margin_line_number(text: str, x0: float, page_width: float) -> bool:
    """Determines if a text element is a line number situated in the right margin.

    Args:
        text (str): The text element to evaluate.
        x0 (float): The x-coordinate of the text's bounding box left edge.
        page_width (float): The total width of the PDF page.

    Returns:
        bool: True if the text is a line number in the right margin, False otherwise.
    """
    text = text.strip()
    if not text.isdigit() or len(text) > 4:
        return False
    return x0 > (page_width * RIGHT_MARGIN_THRESHOLD)

def _is_left_margin_title(text: str, x0: float, page_width: float, is_rider: bool) -> bool:
    """Identifies if a text element is placed in the far left margin, serving as a title.
    
    This dynamically respects the layout state. If the document has entered the Rider
    clauses section, standard left margin layouts are ignored.

    Args:
        text (str): The text element to evaluate.
        x0 (float): The x-coordinate of the text's bounding box left edge.
        page_width (float): The total width of the PDF page.
        is_rider (bool): State flag indicating if the document is in the Rider clauses section.

    Returns:
        bool: True if the text is a left margin title, False otherwise.
    """
    text = text.strip()
    
    # If we have entered the Rider clauses, the standard left margin layout disappears
    if is_rider:
        return False
        
    in_left_margin = x0 < (page_width * LEFT_MARGIN_THRESHOLD)
    is_short = len(text.split()) <= 5
    
    return in_left_margin and is_short

def _is_struck_through(rect: fitz.Rect, horizontal_lines: list) -> bool:
    """Checks if a text bounding box intersects with any drawn horizontal strike-through lines.

    Args:
        rect (fitz.Rect): The bounding box of the text character.
        horizontal_lines (list[fitz.Rect]): A list of bounding boxes representing drawn horizontal lines.

    Returns:
        bool: True if the text intersects with a strike-through line, False otherwise.
    """
    text_height = rect.y1 - rect.y0
    # Expand strike zone to the middle 80% to catch lines drawn slightly high/low
    strike_zone_top = rect.y0 + (text_height * 0.1) 
    strike_zone_bottom = rect.y1 - (text_height * 0.1)
    
    for line_rect in horizontal_lines:
        if line_rect.x0 <= rect.x1 and line_rect.x1 >= rect.x0:
            if strike_zone_top <= line_rect.y0 <= strike_zone_bottom or strike_zone_top <= line_rect.y1 <= strike_zone_bottom:
                return True
    return False

def extract_clean_text_from_pdf(pdf_path: str, start_page: int = 6, end_page: int = 39, is_rider: bool = False) -> tuple[str, bool]:
    """Extracts clean, non-redacted text from a specific page range in a PDF.

    This function utilizes character-level bounding box intersection to surgically remove
    struck-through text. It also detects layout changes (e.g., entering the Rider section)
    and strips out line numbers from the margins.

    Args:
        pdf_path (str): The file path to the PDF document.
        start_page (int, optional): The starting page number (1-indexed). Defaults to 6.
        end_page (int, optional): The ending page number (inclusive, 1-indexed). Defaults to 39.
        is_rider (bool, optional): Initial state indicating if parsing starts in the Rider section. Defaults to False.

    Returns:
        tuple[str, bool]: A tuple containing the combined cleaned text string for the requested pages 
        and the updated `is_rider` boolean state flag.
    """
    clean_pages = []
    
    with fitz.open(pdf_path) as doc:
        start_idx = max(0, start_page - 1)
        end_idx = min(end_page, len(doc))
        
        for page_index in range(start_idx, end_idx):
            page = doc[page_index]
            page_width = page.rect.width
            
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
                    
            page_dict = page.get_text("dict", sort=True)
            page_text = ""
            
            for block in page_dict.get("blocks", []):
                if "lines" not in block:
                    continue
                
                for line in block["lines"]:
                    line_text = ""
                    
                    for span in line["spans"]:
                        text = span["text"]
                        x0, y0, x1, y1 = span["bbox"]
                        
                        if _is_right_margin_line_number(text, x0, page_width):
                            continue
                            
                        char_width = (x1 - x0) / max(len(text), 1)
                        clean_chars = []
                        current_x = x0
                        
                        for char in text:
                            char_x1 = current_x + char_width
                            
                            if char.strip():
                                char_rect = fitz.Rect(current_x - 1, y0, char_x1 + 1, y1)
                                if not _is_struck_through(char_rect, horizontal_lines):
                                    clean_chars.append(char)
                            else:
                                clean_chars.append(char)
                                
                            current_x = char_x1
                            
                        text = "".join(clean_chars)
                        text = re.sub(r' {2,}', ' ', text)
                        
                        if not text.strip():
                            continue
                            
                        if not is_rider and "SHELL ADDITIONAL CLAUSES" in text.upper():
                            logger.info(f"\n{'='*50}\nSTRUCTURE CHANGE DETECTED on Page {page_index + 1}: Entering 'SHELL ADDITIONAL CLAUSES' section. Margin titles disabled.\n{'='*50}")
                            is_rider = True
                        
                        is_bold = "Bold" in span["font"]
                        if is_bold:
                            text = f"**{text.strip()}** "
                        
                        if _is_left_margin_title(text, x0, page_width, is_rider):
                            text = f"[MARGIN_TITLE] {text.strip()} [/MARGIN_TITLE] "
                            
                        line_text += text
                    
                    if line_text.strip():
                        page_text += line_text.strip() + "\n"
                
                page_text += "\n"
                
            clean_pages.append(f"--- PAGE {page_index + 1} ---\n{page_text.strip()}")

    return "\n\n".join(clean_pages), is_rider