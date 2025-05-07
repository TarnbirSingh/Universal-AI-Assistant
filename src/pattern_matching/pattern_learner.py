import logging
from typing import Dict, List, Optional
from datetime import datetime
import uuid

from .task_analyzer import TaskAnalysis
from .pattern_executor import ExecutionResult

logger = logging.getLogger(__name__)

class PatternLearner:
    """Learns and improves patterns from execution results"""
    
    def __init__(self, neo4j_client):
        self.neo4j = neo4j_client
    
    async def learn_from_execution(
        self,
        task: TaskAnalysis,
        pattern: Dict,
        execution_result: ExecutionResult
    ) -> None:
        """Learn from a pattern execution"""
        try:
            # 1. Analyze execution results
            success_analysis = await self._analyze_execution(
                task,
                pattern,
                execution_result
            )
            
            # 2. Update pattern metrics
            await self._update_pattern_metrics(
                pattern_id=pattern["id"],
                success=execution_result.success,
                execution_time=execution_result.duration,
                adaptations=execution_result.adaptations
            )
            
            # 3. Create pattern variations if needed
            if success_analysis["should_create_variation"]:
                try:
                    with self.neo4j.driver.session() as session:
                        # Create new pattern variation
                        variation_id = f"pattern_{uuid.uuid4().hex[:8]}"
                        session.run("""
                            CREATE (p:Pattern {
                                id: $pattern_id,
                                name: $name,
                                type: 'abstract',
                                parent_id: $parent_id,
                                created_at: datetime()
                            })
                        """, {
                            "pattern_id": variation_id,
                            "name": f"Variation of {pattern['name']}",
                            "parent_id": pattern["id"]
                        })
                        
                        # Apply adaptations to steps
                        adapted_steps = self._apply_adaptations(
                            pattern["steps"],
                            execution_result.adaptations
                        )
                        
                        # Create adapted steps
                        for i, step in enumerate(adapted_steps):
                            step_id = f"step_{variation_id}_{i}"
                            session.run("""
                                MATCH (p:Pattern {id: $pattern_id})
                                CREATE (s:Step {
                                    id: $step_id,
                                    description: $description,
                                    order: $order
                                })
                                CREATE (p)-[:HAS_STEP]->(s)
                            """, {
                                "pattern_id": variation_id,
                                "step_id": step_id,
                                "description": step["description"],
                                "order": i
                            })
                            
                except Exception as e:
                    logger.error(f"Error creating pattern variation: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error in pattern learning: {str(e)}")
            raise
    
    async def _analyze_execution(
        self,
        task: TaskAnalysis,
        pattern: Dict,
        execution_result: ExecutionResult
    ) -> Dict:
        """Simple pattern analysis without LLM"""
        # For now, just return some basic metrics
        return {
            "should_create_variation": False,
            "adaptation_success": 1.0 if execution_result.success else 0.0,
            "pattern_applicability": 0.8,  # Default high applicability
            "suggested_improvements": [],
            "abstraction_potential": 0.0
        }
    
    async def _update_pattern_metrics(
        self,
        pattern_id: str,
        success: bool,
        execution_time: float,
        adaptations: List[str]
    ) -> None:
        """Update pattern success metrics"""
        query = """
        MATCH (p:Pattern {id: $pattern_id})
        SET
            p.executions = p.executions + 1,
            p.successful_executions = p.successful_executions + $success_int,
            p.success_rate = toFloat(p.successful_executions + $success_int) / (p.executions + 1),
            p.avg_execution_time = (p.avg_execution_time * p.executions + $exec_time) / (p.executions + 1),
            p.last_used = datetime()
        """
        
        await self.neo4j.execute_query(
            query,
            pattern_id=pattern_id,
            success_int=1 if success else 0,
            exec_time=execution_time
        )
    
    async def _create_pattern_variation(
        self,
        original_pattern: Dict,
        execution_result: ExecutionResult,
        analysis: Dict
    ) -> Dict:
        """Create a new variation of a pattern"""
        variation = original_pattern.copy()
        
        # Generate new ID
        variation["id"] = f"var_{original_pattern['id']}_{uuid.uuid4().hex[:8]}"
        variation["parent_pattern"] = original_pattern["id"]
        variation["created_at"] = datetime.now().isoformat()
        
        # Apply successful adaptations
        if execution_result.adaptations:
            variation["steps"] = self._apply_adaptations(
                variation["steps"],
                execution_result.adaptations
            )
        
        # Apply suggested improvements
        if analysis["suggested_improvements"]:
            variation["steps"] = self._apply_improvements(
                variation["steps"],
                analysis["suggested_improvements"]
            )
        
        # Update metadata
        variation["variation_type"] = "execution_based"
        variation["success_rate"] = 0.0  # New variation starts fresh
        variation["executions"] = 0
        variation["adaptation_context"] = {
            "original_task_concepts": execution_result.context.get("concepts", []),
            "adaptations_applied": execution_result.adaptations,
            "improvements_applied": analysis["suggested_improvements"]
        }
        
        return variation
    
    
    def _apply_adaptations(self, steps: List[Dict], adaptations: List[str]) -> List[Dict]:
        """Apply adaptations to pattern steps"""
        adapted_steps = []
        for step in steps:
            adapted_step = step.copy()
            
            # Apply relevant adaptations
            for adaptation in adaptations:
                if adaptation.get("step_id") == step.get("id"):
                    adapted_step.update(adaptation.get("changes", {}))
            
            adapted_steps.append(adapted_step)
        
        return adapted_steps
    
    def _apply_improvements(self, steps: List[Dict], improvements: List[Dict]) -> List[Dict]:
        """Apply suggested improvements to pattern steps"""
        improved_steps = []
        for step in steps:
            improved_step = step.copy()
            
            # Apply relevant improvements
            for improvement in improvements:
                if improvement["type"] == "step_modification" and improvement["step_id"] == step.get("id"):
                    improved_step.update(improvement["modifications"])
                elif improvement["type"] == "validation_addition":
                    improved_step.setdefault("validations", []).append(improvement["validation"])
            
            improved_steps.append(improved_step)
        
        return improved_steps
