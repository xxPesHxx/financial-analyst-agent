from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional

class AskReq(BaseModel):
    query: str

class ToolCallSchema(BaseModel):
    tool: str
    args: Dict[str, Any]

class AskResponse(BaseModel):
    answer: str
    tool_calls: List[ToolCallSchema] = []
