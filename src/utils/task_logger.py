import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

class TaskLogger:
    def __init__(self, task_id: str, task_description: str):
        """Initialize TaskLogger with task ID and description"""
        self.task_id = task_id
        self.log_data = {
            "task_id": task_id,
            "timestamp": datetime.now().isoformat(),
            "task_description": task_description,
            "execution_flow": {
                "pattern_matching": {
                    "found_patterns": [],
                    "selected_pattern": None,
                    "reason": None
                },
                "pattern_execution": {
                    "abstract_steps": [],
                    "concrete_steps": [],
                    "current_step": {
                        "abstract": None,
                        "concrete": None
                    }
                },
                "learning": {
                    "new_pattern_created": False,
                    "pattern_type": None,
                    "abstracted_steps": []
                }
            },
            "result": {
                "success": False,
                "execution_time": None,
                "pattern_reusability_score": None
            }
        }
        self._start_time = datetime.now()
        self._ensure_log_directory()

    def _ensure_log_directory(self):
        """Create log directories if they don't exist"""
        date_str = datetime.now().strftime('%Y-%m-%d')
        self.log_dir = Path('logs/tasks') / date_str
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_pattern_match(self, patterns: List[Dict], selected_pattern: Optional[Dict] = None, reason: Optional[str] = None):
        """Log pattern matching results"""
        self.log_data["execution_flow"]["pattern_matching"]["found_patterns"] = [
            {
                "pattern_id": p.get("id"),
                "similarity_score": p.get("similarity_score"),
                "why_matched": self._extract_match_reasons(p)
            }
            for p in patterns
        ]
        
        if selected_pattern:
            self.log_data["execution_flow"]["pattern_matching"]["selected_pattern"] = selected_pattern.get("id")
            self.log_data["execution_flow"]["pattern_matching"]["reason"] = reason

    def log_execution_step(self, step_data: Dict):
        """Log execution of a single step with abstract and concrete steps"""
        # Track both abstract and concrete steps
        abstract_step = step_data.get("abstract_step")
        concrete_step = step_data.get("concrete_step")
        
        if abstract_step:
            self.log_data["execution_flow"]["pattern_execution"]["abstract_steps"].append(abstract_step)
        if concrete_step:
            self.log_data["execution_flow"]["pattern_execution"]["concrete_steps"].append(concrete_step)
            
        # Update current step tracking
        self.log_data["execution_flow"]["pattern_execution"]["current_step"].update({
            "abstract": abstract_step,
            "concrete": concrete_step
        })

        # Store the regular step data as well
        self.log_data["execution_flow"]["pattern_execution"].setdefault("steps", []).append({
            "step": step_data.get("type"),
            "input": step_data.get("input"),
            "result": step_data.get("result"),
            "time_taken": step_data.get("duration", "0s")
        })

    def format_steps_message(self) -> str:
        """Format steps for magnetic groupchat message with concise numbering"""
        abstract_steps = self.log_data["execution_flow"]["pattern_execution"]["abstract_steps"]
        concrete_steps = self.log_data["execution_flow"]["pattern_execution"]["concrete_steps"]
        
        # Format abstract steps with concise numbering
        abstract_formatted = ["1) Abstract steps:"]
        for i, step in enumerate(abstract_steps, 1):
            abstract_formatted.append(f"{i}) {step}")
        
        # Format concrete steps with concise numbering
        concrete_formatted = ["\n1) Concrete implementations:"]
        for i, step in enumerate(concrete_steps, 1):
            concrete_formatted.append(f"{i}) {step}")
            
        return "\n".join(abstract_formatted + concrete_formatted)

    def log_pattern_learning(self, pattern_data: Dict):
        """Log pattern learning information"""
        self.log_data["execution_flow"]["learning"].update({
            "new_pattern_created": True,
            "pattern_type": pattern_data.get("type"),
            "abstracted_steps": [
                {
                    "step": step.get("type"),
                    "requirements": step.get("requirements", []),
                    "validation_type": step.get("validation_type")
                }
                for step in pattern_data.get("steps", [])
            ]
        })

    def set_result(self, success: bool, pattern_reusability_score: float = None):
        """Set final execution result"""
        execution_time = (datetime.now() - self._start_time).total_seconds()
        self.log_data["result"].update({
            "success": success,
            "execution_time": f"{execution_time:.2f}s",
            "pattern_reusability_score": pattern_reusability_score
        })

    def _extract_match_reasons(self, pattern: Dict) -> List[str]:
        """Extract reasons why a pattern was matched"""
        reasons = []
        if pattern.get("operations"):
            reasons.extend(f"operation: {op}" for op in pattern["operations"])
        if pattern.get("input_type"):
            reasons.append(f"input_type: {pattern['input_type']}")
        if pattern.get("output_type"):
            reasons.append(f"output_type: {pattern['output_type']}")
        return reasons

    def save(self):
        """Save log to file"""
        log_file = self.log_dir / f"task_{self.task_id}.json"
        latest_file = Path('logs/tasks/latest.json')
        
        # Save task-specific log
        with open(log_file, 'w') as f:
            json.dump(self.log_data, f, indent=2)
            
        # Update latest.json
        latest_file.parent.mkdir(parents=True, exist_ok=True)
        with open(latest_file, 'w') as f:
            json.dump(self.log_data, f, indent=2)

        return str(log_file)
