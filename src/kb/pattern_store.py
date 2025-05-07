import logging
from typing import Dict, List, Optional
from datetime import datetime
import json
import neo4j
from neo4j import GraphDatabase
from src.utils.id_utils import generate_unique_id

logger = logging.getLogger(__name__)

class PatternStore:
    """Interface for storing and retrieving patterns in Neo4j"""
    
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
    async def initialize(self):
        """Initialize Neo4j schema and constraints"""
        with self.driver.session() as session:
            # Create uniqueness constraints
            session.run("""
                CREATE CONSTRAINT pattern_id IF NOT EXISTS
                FOR (p:Pattern) REQUIRE p.id IS UNIQUE
            """)
            
            session.run("""
                CREATE CONSTRAINT task_id IF NOT EXISTS
                FOR (t:Task) REQUIRE t.id IS UNIQUE
            """)
            
            session.run("""
                CREATE CONSTRAINT step_id IF NOT EXISTS
                FOR (s:Step) REQUIRE s.id IS UNIQUE
            """)
            
            # Create indexes
            session.run("""
                CREATE INDEX pattern_success_rate IF NOT EXISTS
                FOR (p:Pattern) ON (p.success_rate)
            """)
            
            session.run("""
                CREATE INDEX pattern_type IF NOT EXISTS
                FOR (p:Pattern) ON (p.type)
            """)
            
            session.run("""
                CREATE INDEX task_type IF NOT EXISTS
                FOR (t:Task) ON (t.type)
            """)
            
            session.run("""
                CREATE INDEX step_order IF NOT EXISTS
                FOR (s:Step) ON (s.order)
            """)
    
    async def store_task_with_solutions(self, task: Dict, concrete_steps: List[Dict], abstract_steps: List[Dict]) -> str:
        """Store task and both its concrete and abstract solution patterns"""
        with self.driver.session() as session:
            # Create unique IDs
            task_id = task.get("id") or f"task_{generate_unique_id(task['name'])}"
            concrete_id = f"concrete_{task_id}"
            abstract_id = f"abstract_{task_id}"
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

            # Create or merge Task node
            session.run("""
                MERGE (t:Task {id: $task_id})
                SET t = $task_properties
            """, {
                "task_id": task_id,
                "task_properties": task
            })
            
            # Create or update Concrete Pattern
            session.run("""
                MERGE (p:Pattern {id: $pattern_id})
                SET p.name = $pattern_name,
                    p.type = 'concrete'
                WITH p
                MATCH (t:Task {id: $task_id})
                MERGE (t)-[:HAS_CONCRETE_PATTERN]->(p)
                WITH p
                OPTIONAL MATCH (p)-[r:HAS_STEP]->(s:Step)
                DELETE r, s
            """, {
                "pattern_id": concrete_id,
                "pattern_name": f"Concrete Pattern for {task['name']}",
                "task_id": task_id
            })
            
            # Create or update Abstract Pattern
            session.run("""
                MERGE (p:Pattern {id: $pattern_id})
                SET p.name = $pattern_name,
                    p.type = 'abstract'
                WITH p
                MATCH (t:Task {id: $task_id})
                MERGE (t)-[:HAS_ABSTRACT_PATTERN]->(p)
                WITH p
                OPTIONAL MATCH (p)-[r:HAS_STEP]->(s:Step)
                DELETE r, s
            """, {
                "pattern_id": abstract_id,
                "pattern_name": f"Abstract Pattern for {task['name']}",
                "task_id": task_id
            })
            
            # Create Concrete Steps
            for i, step in enumerate(concrete_steps):
                # Generate unique step ID based on timestamp and content
                step_id = f"step_concrete_{generate_unique_id(f'{timestamp}_{step['description']}')}"
                session.run("""
                    MATCH (p:Pattern {id: $pattern_id})
                    CREATE (s:Step {
                        id: $step_id,
                        description: $description,
                        order: $order,
                        context: $context,
                        type: 'concrete'
                    })
                    CREATE (p)-[:HAS_STEP]->(s)
                """, {
                    "pattern_id": concrete_id,
                    "step_id": step_id,
                    "description": step["description"],
                    "order": i,
                    "context": json.dumps(step.get("context", {}))
                })
            
            # Create Abstract Steps
            for i, step in enumerate(abstract_steps):
                # Generate unique step ID based on timestamp and content
                step_id = f"step_abstract_{generate_unique_id(f'{timestamp}_{step['description']}')}"
                # Extract semantic details
                semantic_details = step.get("semantic_details", {})
                description = step.get("abstract_description", step.get("description", ""))
                
                # Create context with semantic information
                context = {
                    "core_action": semantic_details.get("core_action"),
                    "intent": semantic_details.get("intent"),
                    "target_type": semantic_details.get("target_type"),
                    "interaction_pattern": semantic_details.get("interaction_pattern")
                }
                
                session.run("""
                    MATCH (p:Pattern {id: $pattern_id})
                    CREATE (s:Step {
                        id: $step_id,
                        description: $description,
                        order: $order,
                        context: $context,
                        type: 'abstract',
                        core_action: $core_action,
                        intent: $intent,
                        target_type: $target_type,
                        interaction_pattern: $interaction_pattern
                    })
                    CREATE (p)-[:HAS_STEP]->(s)
                """, {
                    "pattern_id": abstract_id,
                    "step_id": step_id,
                    "description": description,
                    "order": i,
                    "context": json.dumps(context),
                    "core_action": semantic_details.get("core_action"),
                    "intent": semantic_details.get("intent"),
                    "target_type": semantic_details.get("target_type"),
                    "interaction_pattern": semantic_details.get("interaction_pattern")
                })
            
            return task_id
    
    async def link_task_to_existing_pattern(self, task: Dict, pattern_id: str) -> str:
        """Link a new task to an existing abstract pattern"""
        with self.driver.session() as session:
            # Create task node
            task_id = task.get("id") or f"task_{generate_unique_id(task['name'])}"
            
            # Create task and link to existing abstract pattern
            session.run("""
                MERGE (t:Task {id: $task_id})
                SET t = $task_properties
                WITH t
                MATCH (p:Pattern {id: $pattern_id, type: 'abstract'})
                MERGE (t)-[:HAS_ABSTRACT_PATTERN]->(p)
            """, {
                "task_id": task_id,
                "task_properties": task,
                "pattern_id": pattern_id
            })
            
            return task_id

    async def get_task_with_solutions(self, task_id: str) -> Optional[Dict]:
        """Retrieve a task with both its concrete and abstract patterns and steps"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (t:Task {id: $task_id})
                OPTIONAL MATCH (t)-[:HAS_CONCRETE_PATTERN]->(cp:Pattern)-[:HAS_STEP]->(cs:Step)
                WITH t, cp, cs
                ORDER BY cs.order
                WITH t, cp, collect({
                    id: cs.id,
                    description: cs.description,
                    order: cs.order,
                    context: cs.context,
                    type: cs.type
                }) as concrete_steps
                OPTIONAL MATCH (t)-[:HAS_ABSTRACT_PATTERN]->(ap:Pattern)
                WITH t, cp, concrete_steps, ap
                OPTIONAL MATCH (ap)-[:HAS_STEP]->(as:Step)
                WITH t, cp, concrete_steps, ap, as
                ORDER BY as.order
                WITH t, cp, concrete_steps, ap, collect({
                    id: as.id,
                    description: as.description,
                    order: as.order,
                    context: as.context,
                    type: as.type,
                    semantic_details: {
                        core_action: as.core_action,
                        intent: as.intent,
                        target_type: as.target_type,
                        interaction_pattern: as.interaction_pattern
                    }
                }) as abstract_steps
                RETURN {
                    task: properties(t),
                    concrete_pattern: CASE WHEN cp IS NULL THEN null ELSE properties(cp) END,
                    concrete_steps: concrete_steps,
                    abstract_pattern: CASE WHEN ap IS NULL THEN null ELSE {
                        id: ap.id,
                        name: ap.name,
                        type: ap.type,
                        success_rate: ap.success_rate
                    } END,
                    abstract_steps: abstract_steps
                } as result
            """, {"task_id": task_id})
            
            if record := result.single():
                pattern_data = record["result"]
                # Parse JSON context in steps
                for steps in [pattern_data["concrete_steps"], pattern_data["abstract_steps"]]:
                    if steps:
                        for step in steps:
                            try:
                                step["context"] = json.loads(step["context"]) if step["context"] else {}
                            except json.JSONDecodeError:
                                step["context"] = {}
                return pattern_data
            return None
    
    async def update_solution_metrics(
        self,
        task_id: str,
        success: bool,
        execution_time: float,
        pattern_type: str = 'concrete'
    ) -> None:
        """Update success metrics for the task's pattern"""
        with self.driver.session() as session:
            relationship = "HAS_CONCRETE_PATTERN" if pattern_type == "concrete" else "HAS_ABSTRACT_PATTERN"
            session.run(f"""
                MATCH (t:Task {{id: $task_id}})-[:{relationship}]->(p:Pattern)
                SET p.executions = coalesce(p.executions, 0) + 1,
                    p.successful_executions = coalesce(p.successful_executions, 0) + $success_int,
                    p.success_rate = toFloat(coalesce(p.successful_executions, 0) + $success_int) /
                                   (coalesce(p.executions, 0) + 1),
                    p.avg_execution_time = (coalesce(p.avg_execution_time, 0.0) * coalesce(p.executions, 0) +
                                         $exec_time) / (coalesce(p.executions, 0) + 1),
                    p.last_used = datetime()
            """, {
                "task_id": task_id,
                "success_int": 1 if success else 0,
                "exec_time": execution_time
            })
    
    async def close(self):
        """Close the Neo4j driver"""
        self.driver.close()
