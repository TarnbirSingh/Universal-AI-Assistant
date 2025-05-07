import os
import logging
import re  # Added for regex operations
from typing import List, Dict, Optional, Any
from neo4j import GraphDatabase, Driver
from src.utils.id_utils import generate_unique_id

logger = logging.getLogger(__name__)

class GraphStore:
    def __init__(self):
        """Initialize Neo4j connection"""
        # Use fixed credentials since we're running Neo4j locally
        self.driver: Driver = GraphDatabase.driver(
            "bolt://localhost:7687",
            auth=("neo4j", "password123"),
            database="neo4j"
        )
        try:
            self._init_schema()
        except Exception as e:
            logger.error(f"Error initializing schema: {str(e)}")
            self.close()
            raise
        
    def _init_schema(self):
        """Initialize Neo4j schema and constraints"""
        with self.driver.session() as session:
            # Create constraints
            session.run("""
                CREATE CONSTRAINT pattern_name IF NOT EXISTS
                FOR (p:Pattern) REQUIRE p.name IS UNIQUE
            """)
            
            session.run("""
                CREATE CONSTRAINT task_id IF NOT EXISTS
                FOR (t:Task) REQUIRE t.id IS UNIQUE
            """)
            
    async def query_patterns(self, query: str, params: Dict = None) -> List[Dict]:
        """Execute a direct pattern-related query"""
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters=params or {})
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Error querying patterns: {str(e)}")
            return []

    async def find_patterns(self, task_description: str, threshold: float = 0.5) -> List[Dict]:
        """Find patterns with enhanced similarity matching"""
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (p:Pattern)-[:HAS_STEP]->(s)
                    WHERE p.success_rate >= $threshold
                    AND p.operations IS NOT NULL
                    AND any(op in $required_ops WHERE op in p.operations)
                    WITH DISTINCT p,
                         COLLECT({
                             type: s.type,
                             description: s.description,
                             order: toInteger(s.order)
                         }) as steps,
                         p.operations as ops,
                         COALESCE(p.success_rate, 0.0) as base_score
                    WITH p, steps, ops, base_score,
                         base_score * (1 + size([op in ops WHERE op in $required_ops]) * 0.2) as score
                    OPTIONAL MATCH (p)-[:SPECIALIZES*]->(parent:Pattern)
                    WITH p, steps, ops, score, COLLECT(parent) as parents
                    RETURN {
                        id: elementId(p),
                        name: COALESCE(p.name, ''),
                        description: COALESCE(p.description, ''),
                        abstraction_level: COALESCE(p.abstraction_level, 1),
                        core_concept: COALESCE(p.core_concept, 'general'),
                        success_rate: COALESCE(p.success_rate, 0.0),
                        uses: COALESCE(p.uses, 0),
                        input_type: COALESCE(p.input_type, 'unknown'),
                        output_type: COALESCE(p.output_type, 'unknown'),
                        operations: ops,
                        constraints: COALESCE(p.constraints, []),
                        steps: steps,
                        parent_pattern: CASE 
                            WHEN size(parents) > 0 
                            THEN COALESCE(parents[0].name, null)
                            ELSE null 
                        END,
                        similarity_score: score
                    } as pattern
                    ORDER BY pattern.similarity_score DESC
                    LIMIT 10
                """, {
                    "threshold": threshold,
                    "required_ops": self._extract_operations(task_description)
                })
                
                patterns = [record["pattern"] for record in result]
                result.consume()  # Ensure resources are freed
                return patterns
        except Exception as e:
            logger.error(f"Error finding patterns: {str(e)}")
            return []

    def _extract_operations(self, task_description: str) -> List[str]:
        """Extract required operations from task description"""
        operations = []
        if task_description:
            task_lower = task_description.lower()
            
            # Time-related operations
            if any(term in task_lower for term in ["time", "duration", "between"]):
                operations.append("time_calculation")
                
            # Image-related operations
            if any(term in task_lower for term in ["image", "picture", "photo"]):
                operations.append("image_analysis")
                
            # Navigation operations
            if any(term in task_lower for term in ["route", "path", "direction"]):
                operations.append("route_finding")
                
            # Math operations
            if any(term in task_lower for term in ["calculate", "compute", "sum"]):
                operations.append("mathematical_operation")
                
        return operations

    async def store_pattern(self, pattern_data: Dict) -> str:
        """Store a new pattern with enhanced problem structure metadata"""
        try:
            with self.driver.session() as session:
                # First check if pattern exists
                check_result = session.run(
                    "MATCH (p:Pattern {name: $name}) RETURN elementId(p) as id",
                    {"name": pattern_data["name"]}
                )
                if record := check_result.single():
                    check_result.consume()
                    logger.info(f"Pattern already exists with ID: {record['id']}")
                    return record["id"]

                logger.info(f"Creating new pattern: {pattern_data['name']}")

                # Create pattern with enhanced metadata
                result = session.run("""
                    CREATE (p:Pattern {
                        name: $name,
                        description: $description,
                        abstraction_level: $abstraction_level,
                        core_concept: $core_concept,
                        success_rate: $success_rate,
                        uses: $uses,
                        input_type: $input_type,
                        output_type: $output_type,
                        operations: $operations,
                        constraints: $constraints,
                        created_at: datetime()
                    })
                    RETURN elementId(p) as id
                """, {
                    "name": pattern_data["name"],
                    "description": pattern_data["description"],
                    "abstraction_level": pattern_data.get("abstraction_level", 1),
                    "core_concept": pattern_data.get("core_concept", "general"),
                    "success_rate": pattern_data.get("success_rate", 0.0),
                    "uses": pattern_data.get("uses", 0),
                    "input_type": pattern_data.get("input_type", "unknown"),
                    "output_type": pattern_data.get("output_type", "unknown"),
                    "operations": pattern_data.get("operations", []),
                    "constraints": pattern_data.get("constraints", [])
                })

                if not (record := result.single()):
                    result.consume()
                    raise ValueError("Failed to create pattern - no ID returned")

                pattern_id = record["id"]
                result.consume()  # Clean up resources
                logger.info(f"Created pattern with ID: {pattern_id}")

                # Store steps with method information
                if "steps" in pattern_data:
                    prev_step = None
                    for i, step in enumerate(pattern_data["steps"]):
                        # Create step node
                        logger.info(f"Creating step {i}: {step['description']}")
                        step_result = session.run("""
                            MATCH (p:Pattern)
                            WHERE elementId(p) = $pattern_id
                            CREATE (s:Step {
                                order: $order,
                                type: $type,
                                description: $description
                            })
                            CREATE (p)-[:HAS_STEP]->(s)
                            RETURN elementId(s) as id
                        """, {
                            "pattern_id": pattern_id,
                            "order": i,
                            "type": step["type"],
                            "description": step["description"]
                        })

                        if not (record := step_result.single()):
                            step_result.consume()
                            raise ValueError(f"Failed to create step {i}")

                        current_step_id = record["id"]
                        step_result.consume()  # Clean up resources

                        # Link steps in sequence
                        if prev_step:
                            logger.info(f"Linking step {prev_step} to {current_step_id}")
                            link_result = session.run("""
                                MATCH (prev:Step)
                                WHERE elementId(prev) = $prev_id
                                MATCH (curr:Step)
                                WHERE elementId(curr) = $curr_id
                                CREATE (prev)-[:NEXT]->(curr)
                                RETURN true as success
                            """, {
                                "prev_id": prev_step,
                                "curr_id": current_step_id
                            })
                            link_result.consume()  # Clean up resources

                        prev_step = current_step_id

                # Link to parent pattern if specified
                if "parent_pattern" in pattern_data:
                    logger.info(f"Linking to parent pattern: {pattern_data['parent_pattern']}")
                    link_result = session.run("""
                        MATCH (child:Pattern)
                        WHERE elementId(child) = $child_id
                        MATCH (parent:Pattern {name: $parent_name})
                        CREATE (child)-[:SPECIALIZES]->(parent)
                        RETURN true as success
                    """, {
                        "child_id": pattern_id,
                        "parent_name": pattern_data["parent_pattern"]
                    })
                    link_result.consume()  # Clean up resources

                return pattern_id

        except Exception as e:
            logger.error(f"Error storing pattern: {str(e)}")
            raise

    async def update_pattern_success(self, pattern_name: str, success: bool, execution_data: Dict = None):
        """Update pattern success metrics with execution insights"""
        try:
            with self.driver.session() as session:
                # Update success rate and execution metadata
                result = session.run("""
                    MATCH (p:Pattern {name: $name})
                    SET p.uses = COALESCE(p.uses, 0) + 1,
                        p.success_rate = ((COALESCE(p.success_rate, 0.0) * COALESCE(p.uses, 0)) + $success_value) / (COALESCE(p.uses, 0) + 1),
                        p.last_execution = datetime(),
                        p.execution_time = CASE 
                            WHEN $exec_time IS NOT NULL THEN 
                                (COALESCE(p.execution_time, 0) * COALESCE(p.uses, 0) + $exec_time) / (COALESCE(p.uses, 0) + 1)
                            ELSE p.execution_time
                        END,
                        p.adaptations = CASE 
                            WHEN $adaptations IS NOT NULL THEN 
                                COALESCE(p.adaptations, []) + $adaptations
                            ELSE p.adaptations
                        END
                    RETURN p.success_rate as new_rate
                """, {
                    "name": pattern_name,
                    "success_value": 1.0 if success else 0.0,
                    "exec_time": execution_data.get("execution_time") if execution_data else None,
                    "adaptations": execution_data.get("adaptations", []) if execution_data else []
                })
                result.consume()  # Ensure resources are freed
                logger.info(f"Updated pattern {pattern_name} success rate with value {success}")
        except Exception as e:
            logger.error(f"Error updating pattern success: {str(e)}")
    
    async def store_task_execution(self, task_description: str, pattern_name: str, success: bool, metadata: Optional[Dict] = None):
        """Store detailed task execution information"""
        try:
            with self.driver.session() as session:
                # Store execution with enhanced metadata
                result = session.run("""
                    MATCH (p:Pattern {name: $pattern_name})
                    CREATE (t:Task {
                        description: $task_description,
                        timestamp: datetime(),
                        success: $success,
                        input_type: $input_type,
                        output_type: $output_type,
                        operations: $operations,
                        execution_time: $exec_time,
                        steps_completed: $steps_completed,
                        adaptations: $adaptations,
                        error_details: $error_details
                    })
                    CREATE (t)-[:USED_PATTERN]->(p)
                    FOREACH (step IN $step_results |
                        CREATE (s:StepExecution {
                            order: step.order,
                            type: step.type,
                            success: step.success,
                            duration: step.duration,
                            error: step.error
                        })
                        CREATE (t)-[:EXECUTED_STEP]->(s)
                    )
                    RETURN elementId(t) as id
                """, {
                    "pattern_name": pattern_name,
                    "task_description": task_description,
                    "success": success,
                    "input_type": metadata.get("input_type", "unknown") if metadata else "unknown",
                    "output_type": metadata.get("output_type", "unknown") if metadata else "unknown",
                    "operations": metadata.get("operations", []) if metadata else [],
                    "exec_time": metadata.get("execution_time", 0) if metadata else 0,
                    "steps_completed": metadata.get("steps_completed", 0) if metadata else 0,
                    "adaptations": metadata.get("adaptations", []) if metadata else [],
                    "error_details": metadata.get("error") if metadata and not success else None,
                    "step_results": metadata.get("step_results", []) if metadata else []
                })
                
                if record := result.single():
                    task_id = record["id"]
                    result.consume()
                    return task_id
                else:
                    raise ValueError("Failed to store task execution - no ID returned")
        except Exception as e:
            logger.error(f"Error storing task execution: {str(e)}")
            return None
    
    def _adapt_existing_pattern(self, pattern: Dict, task_description: str) -> Dict:
        """Adapt an existing pattern to a new task"""
        # Create a copy of the pattern
        adapted = pattern.copy()
        # Ensure valid text for ID generation
        id_text = task_description if task_description else "unknown_adaptation"
        adapted["name"] = f"adapted_{pattern['name']}_{generate_unique_id(id_text)[:8]}"
        adapted["description"] = task_description
        
        # Keep same structure but adapt steps
        new_steps = []
        for step in pattern["steps"]:
            new_step = step.copy()
            
            # Replace task-specific parts in description
            if "analyze" in step["type"].lower():
                new_step["description"] = f"Analyze: {task_description}"
            elif "process" in step["type"].lower():
                new_step["description"] = f"Process {pattern['input_type']} input: {task_description}"
            elif "generate" in step["type"].lower():
                new_step["description"] = f"Generate {pattern['output_type']} output"
            elif any(key in step["description"].lower() for key in ["route", "time", "image"]):
                # Keep structure but update specific values
                ctx = self._extract_context(task_description)
                if "times" in ctx:
                    new_step["description"] = new_step["description"].replace(
                        pattern.get("context", {}).get("times", [""])[0], 
                        ctx["times"][0]
                    )
                if "from_location" in ctx and "to_location" in ctx:
                    new_step["description"] = (
                        new_step["description"]
                        .replace(pattern.get("context", {}).get("from_location", ""), ctx["from_location"])
                        .replace(pattern.get("context", {}).get("to_location", ""), ctx["to_location"])
                    )
                    
            new_steps.append(new_step)
            
        adapted["steps"] = new_steps
        return adapted
        
    def _extract_context(self, task_description: str) -> Dict[str, Any]:
        """Extract contextual information from task description"""
        context = {}
        if task_description:
            task_lower = task_description.lower()
            
            # Time context
            time_matches = re.findall(r'\d{1,2}:\d{2}', task_description)
            if time_matches:
                context["times"] = time_matches
                
            # Location context
            if "from" in task_lower and "to" in task_lower:
                words = task_lower.split()
                from_idx = words.index("from")
                to_idx = words.index("to")
                
                if from_idx < to_idx:
                    # Extract locations
                    from_loc = []
                    to_loc = []
                    
                    i = from_idx + 1
                    while i < to_idx:
                        from_loc.append(words[i])
                        i += 1
                        
                    i = to_idx + 1
                    while i < len(words) and words[i] not in ["from", "to", "via", "using"]:
                        to_loc.append(words[i])
                        i += 1
                        
                    if from_loc:
                        context["from_location"] = " ".join(from_loc)
                    if to_loc:
                        context["to_location"] = " ".join(to_loc)
                    
        return context

    def close(self):
        """Close the database connection"""
        self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
