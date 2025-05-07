import logging
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Union, TYPE_CHECKING

from autogen_core.memory import ListMemory

# Use environment variable directly
OPENAI_API_MODEL = os.getenv("OPENAI_API_MODEL", "gpt-4") # Default to gpt-4 if not set

if TYPE_CHECKING:
    from src.agents.coordinator_agent import CoordinatorAgent

from src.pattern_matching.task_analyzer import TaskAnalyzer
from src.pattern_matching.pattern_learner import PatternLearner
from src.kb.pattern_store import PatternStore
from src.kb.knowledge_base import KnowledgeBase
from src.agents.abstraction_agent import AbstractionAgent
from src.utils.task_logger import TaskLogger
from src.utils.id_utils import generate_unique_id
from src.types.execution_result import ExecutionResult

logger = logging.getLogger(__name__)

class HybridKnowledgeBase:
    def __init__(self, coordinator=None):
        """Initialize knowledge base components"""
        # Get Neo4j settings
        neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD", "neo4j")
        
        # Initialize knowledge components
        self.pattern_store = PatternStore(neo4j_uri, neo4j_user, neo4j_password)
        self.knowledge_base = KnowledgeBase()
        self.memory = ListMemory()
        
        # Initialize abstraction agent
        self.abstraction_agent = AbstractionAgent(memory=self.memory)
        
        # Initialize task analysis components
        self.task_analyzer = TaskAnalyzer()
        self.coordinator = coordinator
        self.pattern_learner = PatternLearner(neo4j_client=self.pattern_store)

    async def process_task(self, task_description: str) -> Dict[str, Union[str, Dict]]:
        """Process task using hybrid approach of pattern matching and planning"""
        logger.info(f"Processing task: {task_description}")
        # Ensure valid text for task ID generation
        task_id_text = f"task_{task_description[:32] if task_description else 'unknown_task'}"
        task_logger = TaskLogger(generate_unique_id(task_id_text), task_description)
        try:
            # First check Neo4j connection
            await self.pattern_store.initialize()  # Setup Neo4j schema
            # Test connection
            await self.pattern_store.get_pattern_by_id("test")
        except Exception as e:
            logger.error(f"Neo4j connection error: {str(e)}")
            raise RuntimeError("Could not connect to Neo4j database. Please ensure it's running.")
            
        try:
            # Extract problem structure
            problem_structure = self._extract_problem_structure(task_description)
            logger.info(f"Identified problem structure: {problem_structure}")
            
            # Use coordinator and pattern learner to find similar tasks
            task_type = self._extract_task_type(task_description)
            
            # Use TaskAnalyzer to understand the task
            task_analysis = await self.task_analyzer.analyze_task(task_description)
            logger.info(f"Task analysis complete: {task_analysis}")
            
            # Find similar tasks and their solutions
            matched_tasks = await self.pattern_store.find_similar_tasks(task_description)
            logger.info(f"Found {len(matched_tasks)} potential task matches")
            
            if matched_tasks:
                # Have pattern learner analyze the matches
                analysis_results = []
                for task in matched_tasks:
                    # Create default execution result for analysis
                    execution_result = ExecutionResult()
                    
                    # Create execution result with default values
                    execution_result = ExecutionResult(
                        success=True,  # Default to True since we haven't executed yet
                        duration=0.0,  # No execution time yet
                        adaptations=[],  # No adaptations yet
                        context={"task_type": task_type}
                    )
                    
                    analysis = await self.pattern_learner._analyze_execution(
                        task=task_analysis,
                        pattern=task,
                        execution_result=execution_result
                    )
                    
                    similarity_score = analysis.get("pattern_applicability", 0)
                    if similarity_score > 0.7:  # Minimum threshold
                        analysis_results.append((task, similarity_score))
                        
                # Sort by similarity
                analysis_results.sort(key=lambda x: x[1], reverse=True)
                task_similarities = analysis_results

                if matched_tasks:
                    best_match = matched_tasks[0]  # Tasks are already sorted by similarity
                    score = best_match["similarity_score"]
                    logger.info(f"Found similar task with score {score:.2f}")
                    
                    # Solutions are already included in the matched task
                    if best_match["concrete_solution"] and best_match["abstract_solution"]:
                            # Create new task with both solution types
                            logger.info("Creating new task with reused solutions")
                            task = {
                                "name": f"task_{task_type}_{generate_unique_id(task_description)[:8]}",
                                "description": task_description,
                                "type": task_type,
                            }
                            
                            task_id = await self.pattern_store.store_task_with_solutions(
                                task=task,
                                concrete_steps=best_match["concrete_solution"]["steps"],
                                abstract_steps=best_match["abstract_solution"]["steps"]
                            )
                            
                            # Log the match
                            task_logger.log_pattern_match(
                                matched_tasks,
                                best_match["task"],
                                f"Found similar task with score {score:.2f}"
                            )
                            
                            # Store execution result
                            metadata = {
                                "task_id": task_id,
                                "similar_task_id": best_match["task"]["id"],
                                "similarity_score": score,
                                "problem_structure": problem_structure
                            }
                            await self._store_task_result(
                                task_description=task_description,
                                task_id=task_id,
                                success=True,
                                metadata=metadata,
                                task_logger=task_logger,
                                task_type=task_type
                            )
                            
                            return {
                                "type": "task_similarity_match",
                                "task_id": task_id,
                                "similar_task": best_match["task"],
                                "concrete_solution": best_match["concrete_solution"],
                                "abstract_solution": best_match["abstract_solution"],
                                "similarity_score": score
                            }
            
            # No suitable pattern found, create new abstract pattern from task execution
            logger.info("Creating new pattern from task execution")
            task_type = self._extract_task_type(task_description)
            logger.info(f"Extracted task type: {task_type}")

            # Track how agents solve this task
            execution_steps = []
            
            # Log start of solution process
            task_logger.log_execution_step({
                "type": "start_solution",
                "input": task_description,
                "result": "starting task solution"
            })

            # Initialize step order counter
            step_order = 0

            # Validate required inputs
            required_inputs = self._validate_required_inputs(task_description, problem_structure)
            execution_steps.append({
                "type": "validate_input",
                "description": "Validate required inputs",
                "required_inputs": required_inputs,
                "validation_type": "input_validation",
                "validation_description": "Check if all required information is provided",
                "order": step_order
            })
            step_order += 1

            if not all(required_inputs.values()):
                # Log missing inputs
                task_logger.log_execution_step({
                    "type": "validation_failed",
                    "input": required_inputs,
                    "result": "missing required inputs"
                })
                missing = [k for k, v in required_inputs.items() if not v]
                raise ValueError(f"Missing required inputs: {', '.join(missing)}")

            # Record task execution steps
            service_calls = []
            if task_type == "navigation":
                service_calls.append({
                    "type": "service_call",
                    "service": "route_finding",
                    "description": "Find route between locations",
                    "parameters": ["start_location", "end_location"],
                    "order": step_order
                })
                step_order += 1

            for call in service_calls:
                execution_steps.append(call)

            # Create abstract pattern using AbstractionAgent
            # Create abstract pattern using AbstractionAgent
            pattern = {
                "steps": execution_steps,
                "name": f"Pattern for {task_description}",
                "placeholders": {},  # Initialize empty placeholders
                "created_at": datetime.now().isoformat(),
                "metadata": {
                    "task_type": task_type,
                    "domain_terms": list(self._extract_domain_terms(task_description))  # Convert set to list
                }
            }
            try:
                abstract_pattern = await self.abstraction_agent.abstract_pattern(pattern)
                if not abstract_pattern.get("steps"):
                    logger.warning("Abstract pattern has no steps, falling back to concrete steps")
                    abstract_pattern["steps"] = execution_steps
            except Exception as e:
                logger.error(f"Error abstracting pattern: {str(e)}")
                # Fallback to using concrete steps if abstraction fails
                abstract_pattern = {
                    **pattern,
                    "steps": execution_steps
                }
            
            # Store task with both solutions using store_task_with_solutions
            task = {
                "name": f"task_{task_type}_{generate_unique_id(task_description)[:8]}",
                "description": task_description,
                "type": task_type,
            }

            task_id = await self.pattern_store.store_task_with_solutions(
                task=task,
                concrete_steps=execution_steps,
                abstract_steps=abstract_pattern["steps"]
            )
            logger.info(f"Created new task with both solution types, ID: {task_id}")

            # Log pattern creation and solution metrics
            if task_logger:
                task_logger.log_pattern_learning({
                    "type": "new_task",
                    "id": task_id,
                    "task_type": task_type,
                    "has_abstract": True,
                    "has_concrete": True
                })

                await self._store_task_result(
                    task_description=task_description,
                    task_id=task_id,
                    success=True,
                    metadata={
                        "task_id": task_id,
                        "problem_structure": problem_structure,
                        "execution_steps": execution_steps
                    },
                    task_logger=task_logger,
                    task_type=task_type
                )

            task_logger.set_result(True, 1.0)  # Task completed successfully
            log_file = task_logger.save()
            logger.info(f"Task execution log saved to: {log_file}")

            return {
                "type": "new_task",
                "task_id": task_id,
                "concrete_steps": execution_steps,
                "abstract_steps": abstract_pattern["steps"],
                "log_file": log_file
            }
            
        except Exception as e:
            logger.error(f"Task processing failed: {str(e)}")
            # Log error and save
            task_logger.set_result(False)
            log_file = task_logger.save()
            logger.info(f"Task execution log saved to: {log_file}")
            raise RuntimeError(f"Task processing failed: {str(e)}")
            
    async def _store_task_result(self, task_description: str, task_id: str, success: bool, metadata: Dict = None, task_logger: Optional[TaskLogger] = None, task_type: str = None):
        """Store task execution result and update solution metrics"""
        try:
            logger.info(f"Updating metrics for task: {task_id}")
            
            # Log execution step
            if task_logger:
                task_logger.log_execution_step({
                    "type": "store_solution",
                    "input": task_id,
                    "result": "success" if success else "failure",
                    "metadata": metadata
                })
            
            # Update solution metrics
            await self.pattern_store.update_solution_metrics(
                task_id=task_id,
                success=success,
                execution_time=1.0,  # Default time if not provided
                used_solution_type='concrete'  # We use concrete solutions for execution
            )
            logger.info(f"Updated metrics for concrete solution of task: {task_id}")
            
            # Log execution storage
            if task_logger:
                task_logger.log_execution_step({
                    "type": "store_execution",
                    "input": task_description,
                    "result": f"updated metrics for task: {task_id}",
                    "metadata": {
                        "task_id": task_id,
                        "success": success
                    }
                })
            
        except Exception as e:
            logger.error(f"Error storing task result: {str(e)}")
            if task_logger:
                task_logger.log_execution_step({
                    "type": "store_error",
                    "input": task_id,
                    "result": "error",
                    "error": str(e)
                })
            
    def _extract_problem_structure(self, task_description: str) -> Dict:
        """Extract key features of the problem from task description"""
        structure = {
            "input_type": "unknown",
            "output_type": "unknown",
            "operations": [],
            "constraints": [],
            "context": {}
        }
        
        # Extract input/output types
        if any(term in task_description.lower() for term in ["time", "minutes", "hours"]):
            structure["input_type"] = "time"
            structure["output_type"] = "duration"
            structure["operations"].append("time_calculation")
            structure["operations"].append("mathematical_operation")
            
        if any(term in task_description.lower() for term in ["image", "picture", "photo"]):
            structure["input_type"] = "image"
            
        if any(term in task_description.lower() for term in ["route", "path", "direction"]):
            structure["output_type"] = "route"
            structure["operations"].append("route_finding")
            
        # Extract context
        time_matches = self._extract_time_values(task_description)
        if time_matches:
            structure["context"]["times"] = time_matches
            
        return structure
        
    def _extract_time_values(self, text: str) -> List[str]:
        """Extract time values from text"""
        import re
        time_pattern = r'\d{1,2}:\d{2}'
        return re.findall(time_pattern, text)

    def _validate_required_inputs(self, task_description: str, problem_structure: Dict) -> Dict[str, bool]:
        """Validate all required inputs are present"""
        task_lower = task_description.lower()
        
        # Required inputs based on task type and structure
        required = {}
        
        # Navigation tasks require start and end locations
        if problem_structure.get("output_type") == "route":
            required["start_location"] = any(word in task_lower for word in ["from", "at", "starting"])
            required["end_location"] = any(word in task_lower for word in ["to", "destination"])
        
        # Image tasks require image source
        if problem_structure.get("input_type") == "image":
            required["image_source"] = any(word in task_lower for word in ["image", "picture", "photo"])
        
        # If no specific requirements identified, task is assumed to have all required inputs
        if not required:
            required["task_description"] = True
            
        return required
        
    def _extract_domain_terms(self, text: str) -> List[str]:
        """Extract domain-specific terms from text"""
        # Basic domain term extraction
        domain_terms = set()
        # Common domain indicators (e.g. technical terms, named entities)
        indicators = [
            r'\b[A-Z][A-Za-z]+ ?(?:[A-Z][A-Za-z]+)*\b',  # CamelCase or PascalCase words
            r'\b[A-Z]{2,}\b',  # Acronyms
            r'\b(?:api|sdk|url|gui|cli|db|ui|ux|ai|ml)\b',  # Common tech abbreviations
            r'\b(?:database|server|client|endpoint|interface)\b'  # Common tech terms
        ]
        
        for pattern in indicators:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            domain_terms.update(match.group(0) for match in matches)
        
        return list(domain_terms)  # Convert set to list for JSON serialization

    def _extract_task_type(self, task_description: str) -> str:
        """Map task description to general task type"""
        task_lower = task_description.lower()
        
        if "image" in task_lower or "picture" in task_lower:
            return "analysis"
            
        if "route" in task_lower or "path" in task_lower:
            return "navigation"
            
        if "time" in task_lower or "duration" in task_lower:
            return "calculation"
        
        return "general"

    async def create_pattern_from_execution(
        self, 
        task_description: str,
        execution_steps: List[Dict],
        task_logger: Optional[TaskLogger] = None
    ) -> str:
        """Create a new pattern from successful task execution"""
        logger.info(f"Creating pattern from execution: {task_description}")
        
        try:
            # Extract task type and structure 
            task_type = self._extract_task_type(task_description)
            problem_structure = self._extract_problem_structure(task_description)
            
            # Create abstract pattern using AbstractionAgent
            pattern = {
                "steps": execution_steps,
                "name": f"Pattern for {task_description}",
                "placeholders": {},  # Initialize empty placeholders
                "created_at": datetime.now().isoformat(),
                "metadata": {
                    "task_type": task_type,
                    "domain_terms": list(self._extract_domain_terms(task_description))  # Convert set to list
                }
            }
            try:
                abstract_pattern = await self.abstraction_agent.abstract_pattern(pattern)
                if not abstract_pattern.get("steps"):
                    logger.warning("Abstract pattern has no steps, falling back to concrete steps")
                    abstract_pattern["steps"] = execution_steps
            except Exception as e:
                logger.error(f"Error abstracting pattern: {str(e)}")
                # Fallback to using concrete steps if abstraction fails
                abstract_pattern = {
                    **pattern,
                    "steps": execution_steps
                }

            # Create new task with both concrete and abstract steps
            task = {
                "name": f"task_{task_type}_{generate_unique_id(task_description)[:8]}",
                "description": task_description,
                "type": task_type,
            }

            # Store task with both solution patterns
            task_id = await self.pattern_store.store_task_with_solutions(
                task=task,
                concrete_steps=execution_steps,
                abstract_steps=abstract_pattern["steps"]
            )
            logger.info(f"Created new task with both solution types, ID: {task_id}")

            # Log pattern creation and store metrics
            if task_logger:
                task_logger.log_pattern_learning({
                    "type": "new_task",
                    "id": task_id,
                    "task_type": task_type,
                    "problem_structure": problem_structure,
                    "has_abstract": True,
                    "has_concrete": True
                })

                task_logger.log_execution_step({
                    "type": "store_solution",
                    "input": task_id,
                    "result": "task created with both solution patterns"
                })

                await self._store_task_result(
                    task_description=task_description,
                    task_id=task_id,
                    success=True,
                    metadata={
                        "task_id": task_id,
                        "problem_structure": problem_structure,
                        "execution_steps": execution_steps
                    },
                    task_logger=task_logger,
                    task_type=task_type
                )
            
            return task_id
            
        except Exception as e:
            logger.error(f"Error creating pattern from execution: {str(e)}")
            if task_logger:
                task_logger.log_execution_step({
                    "type": "pattern_creation_error",
                    "input": task_description,
                    "error": str(e)
                })
            raise RuntimeError(f"Failed to create pattern: {str(e)}")
