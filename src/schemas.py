from pydantic import BaseModel, Field
from typing import List, Optional

class PageClause(BaseModel):
    """Pydantic model representing a single legal clause or fragment extracted from a page."""
    
    id: Optional[str] = Field(
        default=None,
        description="The clause identifier (e.g., '1', '2', '3'). Leave as null if this text is a continuation from the previous page and the ID is not visible."
    )
    title: Optional[str] = Field(
        default=None,
        description="The clause title/heading. Use the [MARGIN_TITLE] tags or bold text as hints. Leave as null if this is a continuation and no title is visible."
    )
    text: str = Field(
        description="The text content of the clause found on this specific page. Do not invent text that is not on the page."
    )
    is_continued_from_previous_page: bool = Field(
        default=False,
        description="Set to True if this text belongs to the clause from the previous page. This is usually the case for the first text on a page if it lacks a formal clause ID and title, even if it begins a new sentence."
    )

class PageExtraction(BaseModel):
    """Pydantic model representing all extracted clauses from a single PDF page."""
    
    clauses: List[PageClause] = Field(
        description="List of all clauses and clause fragments extracted from the current page, strictly in the order they appear."
    )