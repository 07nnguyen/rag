from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class QueryReq(BaseModel):
    query: str = Field(..., min_length=2, max_length=4000)
    top_k: int = 6
filters: Optional[Dict[str, Any]] = None
session_id: Optional[str] = None

class Source(BaseModel):
    text: str
    path: Optional[str] = None
    score: float
class QueryResp(BaseModel):
    answer: str
    sources: List[Source]
