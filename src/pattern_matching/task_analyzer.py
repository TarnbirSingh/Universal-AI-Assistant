from dataclasses import dataclass
from typing import List, Dict, Optional
from collections import Counter
import logging
import re

logger = logging.getLogger(__name__)

@dataclass
class TaskAnalysis:
    task_description: str
    actions: List[str]
    requirements: Dict[str, any]
    complexity: float
    constraints: List[str]
    context: Dict[str, any]

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dictionary"""
        return {
            "task_description": self.task_description,
            "actions": list(self.actions),
            "requirements": {k: str(v) if not isinstance(v, (bool, int, float, list, dict)) else v
                           for k, v in self.requirements.items()},
            "complexity": float(self.complexity),
            "constraints": list(self.constraints),
            "context": {k: str(v) if not isinstance(v, (bool, int, float, list, dict)) else v
                       for k, v in self.context.items()}
        }

class ActionExtractor:
    """Extracts actions from task descriptions with improved pattern recognition"""
    
    def __init__(self):
        self.core_actions = {
            'navigate': [
                'go to', 'open', 'access', 'navigate', 'visit', 'load',
                'launch', 'start', 'bring up', 'show me'
            ],
            'find': [
                'find', 'search', 'get', 'retrieve', 'look for', 'locate',
                'fetch'
            ],
            'show': [
                'show', 'display', 'give', 'tell', 'provide', 'present',
                'output', 'return', 'reveal'
            ],
            'create': [
                'create', 'add', 'make', 'generate', 'new', 'start',
                'initialize', 'setup'
            ],
            'update': [
                'update', 'modify', 'change', 'edit', 'revise', 'alter',
                'adjust'
            ],
            'delete': [
                'delete', 'remove', 'clear', 'erase', 'eliminate',
                'dispose'
            ]
        }
        
        # Additional action patterns that combine verbs with business contexts
        self.action_patterns = {
            'navigate': [
                r'(go|navigate|open|access).*(s/?4|sap|system|page|screen|view)',
                r'(show|display|bring).*(screen|page|view)',
                r'(start|launch|run).*(application|app|system)'
            ],
            'find': [
                r'(find|search|get|look).*(customer|product|order|document)',
                r'(show|display|list).*(details|information|data)',
                r'(retrieve|fetch).*(record|entry|item)'
            ],
            'show': [
                r'(show|give|tell).*(information|details|facts|data)',
                r'(display|present).*(results|status|summary)',
                r'(output|return).*(values|records|items)'
            ]
        }
    
    def extract_actions(self, task_description: str) -> List[str]:
        """Extract actions from task description using improved pattern matching"""
        task_lower = task_description.lower()
        actions = set()
        
        # Check for direct action words
        for action_type, indicators in self.core_actions.items():
            if any(indicator in task_lower for indicator in indicators):
                actions.add(action_type)
        
        # Check for complex patterns
        for action_type, patterns in self.action_patterns.items():
            if any(re.search(pattern, task_lower) for pattern in patterns):
                actions.add(action_type)
        
        return list(actions)

class RequirementAnalyzer:
    """Analyzes task requirements with improved system understanding"""
    
    def __init__(self):
        self.requirement_patterns = {
            'web_access': [
                r'(s/?4|sap|browser|url|web|http|cloud)',
                r'(login|authenticate|access).*(system|portal)',
                r'(open|visit|navigate).*(page|site)'
            ],
            'file_system': [
                r'(file|folder|directory|path)',
                r'(save|load|store|read).*(file|data)',
                r'(export|import).*(file|data)'
            ],
            'database': [
                r'(record|entry|table|database)',
                r'(store|retrieve|query).*(data|information)',
                r'(sql|db|database)'
            ],
            'authentication': [
                r'(login|authenticate|credential)',
                r'(user|password|token|access)',
                r'(permission|authorize|rights)'
            ]
        }
    
    async def analyze(self, task_description: str) -> Dict[str, any]:
        """Analyze requirements from task description"""
        requirements = {}
        task_lower = task_description.lower()
        
        # Check each requirement type
        for req_type, patterns in self.requirement_patterns.items():
            if any(re.search(pattern, task_lower) for pattern in patterns):
                requirements[req_type] = True
        
        return requirements

class TaskAnalyzer:
    """Main task analysis system with improved pattern recognition"""
    
    def __init__(self):
        self.action_extractor = ActionExtractor()
        self.requirement_analyzer = RequirementAnalyzer()
        
    async def analyze_task(self, task_description: str) -> TaskAnalysis:
        """Analyze a task and extract its characteristics"""
        try:
            # Extract core actions
            actions = self.action_extractor.extract_actions(task_description)
            logger.info(f"Extracted actions: {actions}")
            
            # Analyze requirements
            requirements = await self.requirement_analyzer.analyze(task_description)
            logger.info(f"Analyzed requirements: {requirements}")
            
            # Calculate complexity (improved heuristic)
            complexity = self._calculate_complexity(task_description, actions, requirements)
            
            # Extract constraints
            constraints = self._extract_constraints(task_description)
            
            # Build context
            context = self._build_context(task_description, actions, requirements)
            
            return TaskAnalysis(
                task_description=task_description,
                actions=actions,
                requirements=requirements,
                complexity=complexity,
                constraints=constraints,
                context=context
            )
            
        except Exception as e:
            logger.error(f"Error analyzing task: {str(e)}")
            raise
    
    def _calculate_complexity(
        self,
        task_description: str,
        actions: List[str],
        requirements: Dict[str, any]
    ) -> float:
        """Calculate task complexity score with improved metrics"""
        complexity_score = 0.0
        
        # Action complexity (reduced weight for basic actions)
        complexity_score += len(actions) * 0.15
        
        # Requirement complexity (reduced base weight)
        complexity_score += len(requirements) * 0.15
        
        # System interaction complexity (reduced weights)
        if 'web_access' in requirements:
            complexity_score += 0.1
        if 'database' in requirements:
            complexity_score += 0.15
        if 'authentication' in requirements:
            complexity_score += 0.1
        
        # Length/detail complexity (reduced impact)
        words = task_description.split()
        complexity_score += min(len(words) * 0.005, 0.1)
        
        return min(complexity_score, 1.0)
    
    def _extract_constraints(self, task_description: str) -> List[str]:
        """Extract constraints with improved pattern recognition"""
        constraints = []
        task_lower = task_description.lower()
        
        # Time constraints
        time_patterns = [
            r'(before|after|within|during)',
            r'(today|tomorrow|now)',
            r'(\d+\s*(minute|hour|day))'
        ]
        if any(re.search(pattern, task_lower) for pattern in time_patterns):
            constraints.append("time_constrained")
        
        # Resource constraints
        resource_patterns = [
            r'(using|with|without)',
            r'(only|must|should|need)',
            r'(limit|maximum|minimum)'
        ]
        if any(re.search(pattern, task_lower) for pattern in resource_patterns):
            constraints.append("resource_constrained")
        
        # Data constraints
        data_patterns = [
            r'(valid|invalid|correct)',
            r'(format|type|structure)',
            r'(required|optional|mandatory)'
        ]
        if any(re.search(pattern, task_lower) for pattern in data_patterns):
            constraints.append("data_constrained")
        
        return constraints
    
    def _build_context(
        self,
        task_description: str,
        actions: List[str],
        requirements: Dict[str, any]
    ) -> Dict[str, any]:
        """Build task context with improved system understanding"""
        return {
            "description": task_description,
            "primary_action": actions[0] if actions else None,
            "secondary_actions": actions[1:] if len(actions) > 1 else [],
            "system_requirements": requirements,
            "domain": self._infer_domain(task_description, actions, requirements),
            "complexity_factors": self._get_complexity_factors(actions, requirements)
        }
    
    def _infer_domain(
        self,
        task_description: str,
        actions: List[str],
        requirements: Dict[str, any]
    ) -> str:
        """Infer the task domain with improved recognition"""
        task_lower = task_description.lower()
        
        # Check for S/4HANA specific indicators
        if any(term in task_lower for term in ['s/4', 'sap', 'hana']):
            return "s4_hana"
        
        # Check for web operations
        if 'web_access' in requirements:
            return "web"
            
        # Check for data operations
        if any(term in task_lower for term in ['data', 'record', 'entry', 'database']):
            return "data"
            
        # Check for file operations
        if 'file_system' in requirements:
            return "filesystem"
            
        # Check for communication
        if any(term in task_lower for term in ['send', 'message', 'communicate']):
            return "communication"
            
        return "general"
    
    def _get_complexity_factors(
        self,
        actions: List[str],
        requirements: Dict[str, any]
    ) -> List[str]:
        """Identify factors contributing to task complexity"""
        factors = []
        
        if len(actions) > 2:
            factors.append("multiple_actions")
            
        if 'authentication' in requirements:
            factors.append("requires_auth")
            
        if 'database' in requirements and 'web_access' in requirements:
            factors.append("multi_system")
            
        return factors
