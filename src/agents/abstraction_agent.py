import logging
import os
from typing import Dict, List, Optional
from datetime import datetime
import json
from src.utils.id_utils import generate_unique_id

logger = logging.getLogger(__name__)

class AbstractionAgent:
    """Agent specialized in converting concrete steps into meaningful abstract patterns"""
    
    def __init__(self, memory: Optional[Dict] = None):
        self.memory = memory or {}
        self.model_client = None  # Set by telegram bot
        
    async def _abstract_description(self, step: Dict) -> Dict:
        """Use LLM to create meaningful abstraction of step"""
        prompt = """Your task is to convert concrete steps into reusable abstract patterns while preserving meaningful context.

Example abstractions:
1. Concrete: "Go to https://www.sap.com/products/s4hana.html"
   Abstract: "Navigate to the specified product page"
   (NOT "Access knowledge repository" - too abstract)

2. Concrete: "Search for SAP S/4HANA Cloud in the search bar"
   Abstract: "Search for the specified product"
   (NOT "Query information system" - too abstract)

3. Concrete: "Click the 'Trial' button"
   Abstract: "Select the specified action"
   (NOT "Interact with element" - too abstract)

4. Concrete: "Type 'John Smith' into the username field"
   Abstract: "Enter the provided value in the specified field"
   (NOT "Process input data" - too abstract)

5. Concrete: "Press the 'Submit' button on the form"
   Abstract: "Confirm the current action"
   (NOT "Execute operation" - too abstract)

Key rules:
1. Remove specific values but preserve the step's purpose
2. Keep the action clear and contextual
3. Don't over-abstract to the point of losing meaning
4. Think about what makes this step reusable for similar tasks

Input step: "{step}"
Please provide the abstracted version following these examples.
"""

        # Get abstraction from LLM
        messages = [
            {"role": "system", "content": "You are an expert at creating meaningful abstract patterns from concrete steps."},
            {"role": "user", "content": prompt.format(step=step.get('description', ''))}
        ]

        # Use create method for OpenAI client
        try:
            # Debug the model client and call create
            print("=== DEBUG: Model Client ===")
            print(f"Model client type: {type(self.model_client)}")
            print(f"Messages: {messages}")
            
            # Attempt to create a custom chat completion
            print("=== DEBUG: Using custom chat completion ===")
            
            try:
                # Extract any client initialization parameters we might need
                model = "gpt-4"  # Use base model name without version suffix
                base_url = getattr(self.model_client, '_api_base', None)
                api_key = getattr(self.model_client, '_api_key', None)
                
                # Try to get the API key from environment if not in client
                if not api_key and 'OPENAI_API_KEY' in os.environ:
                    api_key = os.environ['OPENAI_API_KEY']
                
                # Print what we found
                print(f"Model: {model}")
                print(f"Base URL: {base_url}")
                print(f"Has API Key: {bool(api_key)}")
                
                # Try to get underlying OpenAI client
                if hasattr(self.model_client, '_client'):
                    raw_client = self.model_client._client
                    print(f"=== DEBUG: Raw client: {type(raw_client)} ===")
                    # Try using raw client directly
                    response = await raw_client.chat.completions.create(
                        messages=messages,
                        model=model
                    )
                    result = response.choices[0].message
                else:
                    # Fall back to regular create
                    result = await self.model_client.create(messages=messages)
                
                # Print raw result for debugging
                print(f"Raw result: {result}")
                
                # Extract the response text from ChatCompletionMessage
                if hasattr(result, 'content'):
                    # Get content directly from the message object
                    raw_content = result.content
                    # Strip the "Abstract: " prefix and quotes if present
                    abstract_description = raw_content.replace('Abstract: ', '').strip('"')
                else:
                    # Fallback for dict or other formats
                    if isinstance(result, dict):
                        abstract_description = (
                            result.get('content', '') or 
                            result.get('text', '') or 
                            result.get('response', '')
                        ).strip()
                    else:
                        # Last resort - convert to string
                        abstract_description = str(result)
                
                if not abstract_description:
                    raise ValueError("No content found in response")
                
                print(f"=== DEBUG: Got response: {abstract_description} ===")
                    
            except Exception as parse_error:
                print(f"=== DEBUG: Error during LLM call or parsing: {str(parse_error)} ===")
                abstract_description = step.get('description', '').replace("'", "").replace('"', '')
                
        except Exception as e:
            logger.error(f"LLM call failed: {str(e)}")
            # Fallback to simple abstraction on any other error
            abstract_description = step.get('description', '').replace("'", "").replace('"', '')
            
        return {
            "abstract_description": abstract_description,
            "semantic_details": {
                "type": step.get('type', 'action'),
                "context": step.get('context', {})
            }
        }

    async def abstract_step(self, step: Dict) -> Dict:
        """Process a single step for abstraction"""
        try:
            abstract = await self._abstract_description(step)
            return {
                "input_step": step,
                "description": abstract["abstract_description"],
                "semantic_details": abstract["semantic_details"]
            }
        except Exception as e:
            logger.error(f"Step abstraction failed: {str(e)}")
            return {
                "error": str(e)
            }

    async def abstract_pattern(self, pattern: Dict) -> Dict:
        """Process entire pattern for abstraction"""
        try:
            # Abstract each step
            abstract_steps = []
            for step in pattern.get("steps", []):
                abstract_step = await self.abstract_step(step)
                if "error" in abstract_step:
                    continue
                abstract_steps.append({
                    "description": abstract_step["description"],  # Use the abstract description
                    "context": step.get("context", {}),
                    "semantic_details": abstract_step["semantic_details"]  # Use abstracted semantic details
                })
            
            # Keep task metadata and steps concrete
            task_metadata = pattern.get("task", {}) or {
                "id": pattern.get("id") or generate_unique_id(pattern.get("name", "")),
                "name": pattern.get("name", ""),
                "description": pattern.get("description", ""),
                "type": pattern.get("type", "task")
            }
            
            return {
                "task": task_metadata,
                "abstract_steps": [
                    {
                        "description": step["description"],
                        "context": step["context"],
                        "order": i,
                        "semantic_details": step["semantic_details"]
                    }
                    for i, step in enumerate(abstract_steps)
                ]
            }
            
        except Exception as e:
            logger.error(f"Pattern abstraction failed: {str(e)}")
            return {"error": str(e)}
