from dataclasses import dataclass
from typing import List, Dict, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class ExecutionResult:
    success: bool
    error: Optional[str] = None
    results: List[Dict] = None
    adaptations: List[str] = None
    duration: float = 0.0

@dataclass
class StepResult:
    success: bool
    output: Optional[Dict] = None
    error: Optional[str] = None
    adaptations: List[str] = None
    context: Dict = None

class ExecutionValidator:
    """Validates execution results for each step"""
    
    async def validate_step(self, step: Dict, result: StepResult) -> bool:
        """Validate a step's execution result"""
        try:
            # Check basic success flag
            if not result.success:
                return False
                
            # Check required outputs
            if step.get("required_outputs"):
                if not result.output:
                    return False
                for required in step["required_outputs"]:
                    if required not in result.output:
                        return False
            
            # Check output constraints if any
            if step.get("output_constraints"):
                for constraint in step["output_constraints"]:
                    if not self._validate_constraint(constraint, result.output):
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error in step validation: {str(e)}")
            return False
    
    def _validate_constraint(self, constraint: Dict, output: Dict) -> bool:
        """Validate a single output constraint"""
        constraint_type = constraint["type"]
        value = output.get(constraint["field"])
        
        if constraint_type == "type":
            return isinstance(value, eval(constraint["value"]))
        elif constraint_type == "range":
            return constraint["min"] <= value <= constraint["max"]
        elif constraint_type == "pattern":
            import re
            return bool(re.match(constraint["pattern"], str(value)))
        elif constraint_type == "custom":
            # Custom validation function
            validation_func = eval(constraint["function"])
            return validation_func(value)
        
        return False

class PatternExecutor:
    """Executes patterns and validates results"""
    
    def __init__(self, agent_pool):
        self.agent_pool = agent_pool
        self.validator = ExecutionValidator()
        
    async def execute_pattern(self, pattern: Dict, context: Dict) -> ExecutionResult:
        """Execute a pattern with given context"""
        start_time = datetime.now()
        
        try:
            # Initialize execution
            results = []
            adaptations = []
            current_context = context.copy()
            
            # Execute each step
            for step in pattern["steps"]:
                # Get appropriate agent
                agent = await self.agent_pool.get_agent(step.get("required_capabilities", []))
                
                # Execute step with retries if needed
                step_result = await self._execute_step_with_retry(
                    agent=agent,
                    step=step,
                    context=current_context
                )
                
                # Validate step result
                if not await self.validator.validate_step(step, step_result):
                    if step.get("required", True):  # If step is required
                        return ExecutionResult(
                            success=False,
                            error=f"Step validation failed: {step.get('name', 'Unknown step')}",
                            results=results,
                            adaptations=adaptations
                        )
                
                # Update context with step results
                if step_result.output:
                    current_context.update(step_result.output)
                
                # Track results and adaptations
                results.append(step_result)
                if step_result.adaptations:
                    adaptations.extend(step_result.adaptations)
            
            # Calculate execution time
            duration = (datetime.now() - start_time).total_seconds()
            
            return ExecutionResult(
                success=True,
                results=results,
                adaptations=adaptations,
                duration=duration
            )
            
        except Exception as e:
            logger.error(f"Error executing pattern: {str(e)}")
            return ExecutionResult(
                success=False,
                error=str(e),
                duration=(datetime.now() - start_time).total_seconds()
            )
    
    async def _execute_step_with_retry(
        self,
        agent,
        step: Dict,
        context: Dict,
        max_retries: int = 3
    ) -> StepResult:
        """Execute a step with retry logic"""
        attempts = 0
        last_error = None
        
        while attempts < max_retries:
            try:
                # Execute step
                result = await agent.execute(step, context)
                
                # If successful, return result
                if result.success:
                    return result
                
                # If failed but can be retried
                if self._can_retry(step, result.error):
                    attempts += 1
                    last_error = result.error
                    # Wait before retry (exponential backoff)
                    await asyncio.sleep(2 ** attempts)
                    continue
                
                # If failed and cannot be retried
                return result
                
            except Exception as e:
                attempts += 1
                last_error = str(e)
                if attempts >= max_retries:
                    break
                await asyncio.sleep(2 ** attempts)
        
        return StepResult(
            success=False,
            error=f"Step failed after {attempts} attempts. Last error: {last_error}"
        )
    
    def _can_retry(self, step: Dict, error: str) -> bool:
        """Determine if a step can be retried based on error"""
        # Don't retry if step explicitly disables it
        if step.get("no_retry", False):
            return False
            
        # Don't retry certain types of errors
        non_retryable = [
            "InvalidInput",
            "ValidationError",
            "AuthenticationError"
        ]
        
        for error_type in non_retryable:
            if error_type in error:
                return False
        
        return True
