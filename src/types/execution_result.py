from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class ExecutionResult:
    """Result of a pattern execution"""
    success: bool = False
    duration: float = 0.0
    adaptations: List[Dict] = None
    context: Dict = None
    error: Optional[str] = None

    def __init__(self):
        self.success = False
        self.duration = 0.0
        self.adaptations = []
        self.context = {}
        self.error = None
