from pydantic import BaseModel
from typing import List

class Board(BaseModel):
    board: List[List[str]]