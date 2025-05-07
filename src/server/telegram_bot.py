import asyncio
import json
import logging
import os
import time
from typing import Dict, List, Optional, Any, AsyncIterator
from src.utils.progress_tracker import ProgressTracker
from src.utils.pattern_utils import (
    extract_url_from_description,
    extract_search_term,
    extract_analysis_target,
    extract_element,
    extract_field,
    extract_input_value,
    enrich_step_context
)
from dataclasses import dataclass
from datetime import datetime

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from openai import OpenAIError

from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import UserProxyAgent
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
from autogen_core import CancellationToken
from autogen_agentchat.messages import TextMessage

from src.agents.abstraction_agent import AbstractionAgent
from src.agents.pattern_search_agent import PatternSearchAgent
from src.kb.pattern_store import PatternStore
from src.utils.task_logger import TaskLogger
from src.utils.id_utils import generate_unique_id

logger = logging.getLogger(__name__)

@dataclass
class ExecutionStep:
    type: str
    description: str
    result: str
    timestamp: float

class EnhancedMagenticOneGroupChat(MagenticOneGroupChat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.execution_steps: List[ExecutionStep] = []
        self._start_time = time.time()
        
    def record_step(self, step_type: str, description: str, result: str):
        """Record an execution step with timestamp"""
        self.execution_steps.append(
            ExecutionStep(
                type=step_type,
                description=description,
                result=result,
                timestamp=time.time() - self._start_time  # Relative time from start
            )
        )
        logger.info(f"Recorded step: {step_type} - {description}")

def get_message_content(message: Any) -> Optional[dict]:
    """Extract message content and determine its type"""
    # Extract raw content
    raw_content = None
    if hasattr(message, 'content'):
        raw_content = message.content
    elif hasattr(message, 'chat_message'):
        raw_content = message.chat_message.content
    else:
        raw_content = str(message)
        
    # Skip empty content
    if not raw_content:
        return None
        
    content = str(raw_content)
    content_lower = content.lower()

    # Skip system artifacts, templates, and technical details
    is_system = any(skip in content for skip in [
        'RequestUsage(',
        'prompt_tokens',
        'completion_tokens',
        'Traceback',
        'Core behaviors:',
        '### ',
        'task:',
        'input:',
        'output:',
        'system:',
        '<autogen',
        'object at 0x',
        'returning websurfer',
        'chatbot.ask(',
        'executing command',
        'human:',
        'assistant:',
        'received input',
        'generated output'
    ]) or (
        '[' in content and ']' in content and any(x in content for x in [
            "screenshot",
            "browser",
            "viewport",
            "following text",
            "metadata",
            "extracted from",
            "autogen",
            "chatbot",
            "image object",
            "object at"
        ])
    )

    if is_system:
        return None

    # Detect completion messages first - check for exact marker and variations
    has_result = (
        "TASK_COMPLETE" in content or
        any(completion in content_lower for completion in [
            "task complete",
            "completed the task",
            "here's what i found",
            "here is what i found",
            "task finished",
            "search complete",
            "completed successfully",
            "i have completed"
        ])
    )
    
    if has_result:
        # Extract just the final result without browser details
        result = []
        for line in content.split('\n'):
            line = line.strip()
            # Skip lines with technical details
            if any(x in line.lower() for x in [
                'screenshot',
                'browser',
                'viewport',
                'following text',
                'metadata',
                'extracted from',
                'image object at'
            ]):
                continue
            if line and not line.startswith('[') and not line.startswith('{'):
                result.append(line)
        
        # Clean up and join the result
        clean_result = ' '.join(result)
        # Remove common markers
        for marker in [
                "TASK_COMPLETE:",
                "TASK_COMPLETE",
                "Task completed:",
                "I have completed the task:",
                "Here is what I found:",  
                "Here's what I found:",
                "Result:",
                "Found:",
                "Successfully completed:",
                "Task finished:",
                "I have completed"
        ]:
            clean_result = clean_result.replace(marker, "").strip()
            clean_result = clean_result.replace(marker.lower(), "").strip()
            
        if clean_result:
            return {
                "type": "completion",
                "content": clean_result
            }
        return None
    
    # Filter action messages to remove technical details
    content = content.strip()
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    filtered_lines = []
    
    for line in lines:
        if not any(skip in line.lower() for skip in [
            'screenshot',
            'browser',
            'viewport',
            'following text',
            'metadata',
            'extracted',
            'image object',
            'returning websurfer',
            'object at',
            'task:',
            'input:',
            'core',
            '###'
        ]):
            filtered_lines.append(line)
    
    # Return filtered content
    if filtered_lines:
        return {
            "type": "action",
            "content": ' '.join(filtered_lines)
        }
    return None

def should_display_message(content: str) -> bool:
    """Determine if a message should be displayed in Telegram progress"""
    content_lower = content.lower()

    # Always show messages that require user interaction
    if "please provide" in content_lower or "would you like" in content_lower:
        return True
    
    # Show messages with clear actions (navigation, interaction, etc)
    has_action = any(action in content_lower for action in [
        "navigating to",
        "clicking on", 
        "clicking the",
        "searching for",
        "entering",
        "typing",
        "extracting",
        "analyzing",
        "found the",
        "selecting",
        "processing",
        "waiting for",
        "going to",
        "opening",
        "reading",
        "checking",
        "looking up",
        "processing",
        "calculating",
        "comparing",
        "evaluating"
    ])

    # Skip system noise but keep interactions and results
    is_noise = any(skip in content for skip in [
        'RequestUsage(',
        'TextMessage(',
        'Task:',
        'Input:',
        'Core behaviors',
        '### ',
        'System:',
        'prompt_tokens',
        'completion_tokens'
    ])
    
    # Only show meaningful interactions, not technical messages
    return has_action and not is_noise and len(content.strip()) < 300

async def aiter_to_list(aiter: AsyncIterator) -> List:
    """Convert an async iterator to a list"""
    if hasattr(aiter, '__aiter__'):
        return [x async for x in aiter]
    else:
        # If it's a coroutine, await it to get the async iterator
        result = await aiter
        return [x async for x in result]

class TelegramBot:
    def __init__(self):
        """Initialize basic properties"""
        self._app = None
        self._input_futures: Dict[int, asyncio.Future] = {}
        self._current_user_id: Optional[int] = None
        self._current_update: Optional[Update] = None
        self._current_task: Dict[int, str] = {}  # Current task per user
        self._initial_tasks: Dict[int, str] = {}  # Initial task that started the interaction
        self.model_client = None
        self.pattern_store = None
        self.abstraction_agent = None
        self.pattern_search_agent = None
        self.browser = None
        self._latest_pattern_id = None  # Track last created pattern ID

    async def _update_execution_progress(self, progress: ProgressTracker, content: str) -> None:
        """Update progress based on message content"""
        content_lower = content.lower()
        
        # Detect execution substage
        if "navigate" in content_lower or "go to" in content_lower:
            substage = "NAVIGATION"
        elif "click" in content_lower or "select" in content_lower:
            substage = "INTERACTION"
        elif "extract" in content_lower or "collect" in content_lower:
            substage = "EXTRACTION"
        elif "found" in content_lower:
            substage = "PROCESSING"
        else:
            substage = "PROCESSING"
            
        # Just update substage and details
        await progress.update_progress(
            stage="EXECUTING",
            substage=substage,
            details=content.strip()
        )

    async def initialize(self):
        """Initialize the Telegram bot with persistent browser and pattern storage"""
        # Get OpenAI settings
        api_key = os.getenv("OPENAI_API_KEY")
        api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        model = os.getenv("OPENAI_MODEL", "gpt-4")
        
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY environment variable must be set")
            
        try:
            # Initialize OpenAI client with custom endpoint configuration
            self.model_client = OpenAIChatCompletionClient(
                model=model,
                base_url=api_base,
                default_headers={
                    "Authorization": f"Bearer {api_key}"
                }
            )
            
            # Initialize pattern store and set up Neo4j schema
            self.pattern_store = PatternStore(
                uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                user=os.getenv("NEO4J_USER", "neo4j"),
                password=os.getenv("NEO4J_PASSWORD", "password123")
            )
            await self.pattern_store.initialize()  # Set up Neo4j schema
            
            # Initialize abstraction agent
            self.abstraction_agent = AbstractionAgent(memory=None)
            # Ensure abstraction agent uses same model client
            self.abstraction_agent.model_client = self.model_client
            
            # Initialize pattern search agent
            self.pattern_search_agent = PatternSearchAgent(
                name="pattern_searcher",
                neo4j_client=self.pattern_store
            )
            
            # Create persistent browser
            self.browser = MultimodalWebSurfer(
                "WebSurfer",
                model_client=self.model_client,
                headless=False,
                debug_dir=os.getenv("TEMP_DIR", "tmp/screenshots")
            )
            
        except OpenAIError as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command"""
        await update.message.reply_text(
            "I'm your personal Universal AI Assistant 🤖 — I learn and evolve with every task you give me, ready to execute anything from simple clicks to complex missions 🚀. Just tell me what you need!"
        )
        # Clear all task context on /start
        user_id = update.effective_user.id
        if user_id in self._current_task:
            del self._current_task[user_id]
        if user_id in self._initial_tasks:
            del self._initial_tasks[user_id]

    async def clear_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Clear current task context"""
        user_id = update.effective_user.id
        if user_id in self._current_task:
            del self._current_task[user_id]
        if user_id in self._initial_tasks:
            del self._initial_tasks[user_id]
        await update.message.reply_text("Task context cleared. Ready for new task.")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming messages"""
        try:
            user_id = update.effective_user.id
            text = update.message.text.strip()
            
            # Handle input request responses
            if user_id in self._input_futures and not self._input_futures[user_id].done():
                self._input_futures[user_id].set_result(text)
                return

            await self._handle_task(text, update)

        except Exception as e:
            logger.error(f"Error handling message: {str(e)}", exc_info=True)
            await update.message.reply_text(f"Error: {str(e)}")

    async def _send_message(self, content: str, update: Update) -> None:
        """Send message with filtering and length handling"""
        try:
            # Skip template sections and system content
            if any(skip in content for skip in [
                '### 🚀 Core Directives',
                '### 🛠️ Smart Interaction Protocol',
                '### 🔥 Execution Strategy',
                '### Core Behaviors',
                'Task:',
                'Input:',
                'prompt_tokens',
                'completion_tokens',
                'TaskResult(',
                'TextMessage(',
                'MultiModalMessage(',
                'RequestUsage(',
                'Task completed in',
                '⚡ Execute the task',
                'Core behaviors:',
                '🔹 When',
                '✔ Act like',
                '1️⃣ Gather',
                '2️⃣ Follow',
                '3️⃣ Adapt',
                '4️⃣ Respect',
                '5️⃣ Self-check'
            ]):
                return

            # Handle long messages with chunking
            if len(content) > 4000:
                parts = [content[i:i+4000] for i in range(0, len(content), 4000)]
                for part in parts:
                    if not any(skip in part for skip in ['Task:', 'Input:', '###']):
                        await update.message.reply_text(part)
            else:
                await update.message.reply_text(content)
        except Exception as e:
            logger.error(f"Error sending message: {str(e)}")

    async def _get_telegram_input(self, prompt: str = None, cancellation_token: Optional[CancellationToken] = None) -> str:
        """Get user input through Telegram"""
        if prompt and self._current_update:
            await self._send_message(prompt, self._current_update)
        
        # Create future for this user
        user_id = self._current_user_id
        if not user_id:
            raise ValueError("No current user ID")
            
        self._input_futures[user_id] = asyncio.Future()
        
        try:
            try:
                # Wait for input while respecting cancellation 
                while True:
                    try:
                        if cancellation_token and cancellation_token.cancelled():
                            raise asyncio.CancelledError()
                        # Add 5-minute timeout for user input
                        return await asyncio.wait_for(self._input_futures[user_id], timeout=300)
                    except asyncio.TimeoutError:
                        logger.warning(f"Input timeout for user {user_id}")
                        raise asyncio.CancelledError("Input timeout")
            except asyncio.CancelledError as e:
                # Log cancellation for debugging
                logger.info(f"Input cancelled for user {user_id}: {str(e)}")
                raise
            
        finally:
            # Cleanup
            if user_id in self._input_futures:
                future = self._input_futures[user_id]
                if not future.done():
                    future.cancel()
                    # Ensure the future is properly cancelled and cleaned up
                    try:
                        await future
                    except asyncio.CancelledError:
                        pass
                del self._input_futures[user_id]

    async def _format_pattern_guidance(self, pattern_data: dict) -> str:
        """Format pattern data into guidance section for prompt"""
        if not pattern_data:
            return ""
            
        # Get pattern steps
        abstract_steps = []
        if "abstract_steps" in pattern_data:
            for step in pattern_data["abstract_steps"]:
                if isinstance(step, dict):
                    desc = step.get('description', '')
                    if desc:
                        abstract_steps.append(desc)
                else:
                    abstract_steps.append(step)
            
        # Format steps as numbered list
        steps_text = "\n".join(f"{i+1}) {step}" for i, step in enumerate(abstract_steps))
            
        # Format concrete examples if available
        concrete_examples = ""
        if "concrete_examples" in pattern_data and pattern_data["concrete_examples"]:
            concrete_examples = "\nConcrete Examples:\n" + "\n".join(
                f"- {example}" for example in pattern_data["concrete_examples"][:3]  # Show up to 3 examples
            )
            
        return f"""
### 🎯 Reference Pattern ###
Use these patterns from a similar task ({pattern_data.get('similarity', 0):.0%} match) as guidance:

Abstract Steps:
{steps_text}
{concrete_examples}

---"""

    async def _handle_task(self, task: str, update: Update) -> None:
        """Handle web navigation task"""
        try:
            user_id = update.effective_user.id
            self._current_user_id = user_id
            self._current_update = update
            
            # Save task context
            if user_id not in self._current_task:
                # New task - save both current and initial task
                self._current_task[user_id] = task
                self._initial_tasks[user_id] = task  # Store initial task
                
            # Search for similar patterns
            messages = [TextMessage(content=task, source="user")]
            pattern_id = None
            similar_pattern_data = None
            
            # Initialize progress tracker
            progress = ProgressTracker(self._app.bot, update.message.chat_id)
            progress.task_description = task
            await progress.update_progress(stage="ANALYZING")
            
            # Get message stream and convert to list
            await progress.update_progress(
                stage="PATTERN_SEARCH",
                details="Searching for similar patterns..."
            )
            
            pattern_messages = await aiter_to_list(
                self.pattern_search_agent.on_messages_stream(messages, CancellationToken())
            )
            
            for msg in pattern_messages:
                if msg.source == "pattern_searcher":
                    try:
                        # Extract pattern data from JSON in message
                        content_parts = msg.content.split("\nDetailed Pattern Data:\n")
                        if len(content_parts) > 1:
                            pattern_data = content_parts[1].strip()
                            similar_pattern_data = json.loads(pattern_data)
                            pattern_id = similar_pattern_data.get("abstract_pattern_id")
                            # Start execution with pattern info
                            # Extract abstract steps
                            abstract_steps = []
                            for step in similar_pattern_data.get("abstract_steps", []):
                                if isinstance(step, dict):
                                    desc = step.get('description', '')
                                    if desc:
                                        abstract_steps.append(desc)
                                else:
                                    abstract_steps.append(step)

                            await progress.update_progress(
                                stage="EXECUTING",
                                pattern_found=True,
                                pattern_similarity=similar_pattern_data.get("similarity", 0),
                                similar_task_description=similar_pattern_data.get("task_description", ""),
                                pattern_abstract_steps=abstract_steps,
                                pattern_concrete_steps=[step['description'] if isinstance(step, dict) else step
                                                     for step in similar_pattern_data.get("concrete_steps", [])[:-1]]  # Exclude last step
                            )
                    except Exception as e:
                        logger.error(f"Error parsing pattern data: {e}")
                        # Start execution without pattern
                        await progress.update_progress(
                            stage="EXECUTING",
                            pattern_found=False
                        )
                        
            
            # Create user proxy for interaction
            user_proxy = UserProxyAgent(
                "human",
                input_func=self._get_telegram_input,
                description="Human providing guidance when needed"
            )
            
            # Create team with browser and step recording
            team = EnhancedMagenticOneGroupChat(
                participants=[self.browser, user_proxy],
                model_client=self.model_client
            )
            
            # Format pattern guidance if available
            pattern_guidance = ""
            if similar_pattern_data:
                pattern_guidance = await self._format_pattern_guidance(similar_pattern_data)
            
            # Build task prompt with completion trigger
            task_prompt = f"""Task: {task}

### 🚀 Core Directives 🚀 ###

✔ **Act like a real user:** Navigate, click, and interact naturally.  

✔ **Never assume—ask when uncertain:** If any key detail is missing, request clarification before proceeding.  

✔ **Stay on track:** Remain focused on the task's goal without deviating.  

✔ **Confirm critical actions:** Before making irreversible changes, ask the user for explicit approval.  

✔ **Communicate efficiently:** Share only the **most relevant** progress updates—avoid unnecessary details.  

✔ **Complete tasks decisively:** When the objective is achieved, state **"TASK_COMPLETE"** to finalize the process.  



---



### 🛠️ Smart Interaction Protocol 🛠️ ###

🔹 **When missing information** → Prompt the user with a **clear and direct question.**  

🔹 **When user input is required** → Formulate a **specific** and **actionable** request.  

🔹 **When a decision is needed** → Present **concise, distinct options** and wait for confirmation.  

🔹 **When providing updates** → Keep it **short, relevant, and focused** on progress.  

🔹 **When encountering an issue** → State the problem **clearly** and suggest solutions instead of stopping.  



---



### 🔥 Execution Strategy 🔥 ###

1️⃣ **Gather all required information** before acting.  

2️⃣ **Follow a step-by-step approach**, ensuring accuracy at each stage.  

3️⃣ **Adapt intelligently**—if the task changes based on new input, update the approach accordingly.  

4️⃣ **Respect the user's intent**—prioritize their requests and confirmations.  

5️⃣ **Self-check your output**—if a response seems incomplete, refine it before sending.  



- Very important: When a Task was sucessfully completed, say "TASK_COMPLETE" to end the Task



⚡ *Execute the task efficiently while maintaining a natural and user-friendly interaction.* ⚡  
"""

            # Combine task and pattern guidance
            enhanced_task = task_prompt + "\n" + (pattern_guidance if pattern_guidance else "")

            # Create task logger with initial task
            initial_task = self._initial_tasks[user_id]
            task_logger = TaskLogger(generate_unique_id(initial_task), initial_task)

            # Run task and record steps
            completed = False
            team_messages = await aiter_to_list(team.run_stream(task=enhanced_task))
            
            for message in team_messages:
                content = str(message.content if hasattr(message, 'content') else str(message))
                
                # Get message content
                result = get_message_content(message)
                if result:
                    # Record every action as a step, but only show relevant ones in progress
                    message_type = result["type"]
                    is_displayable = should_display_message(result["content"])
                    team.record_step(
                        "completion" if message_type == "completion" else "assistant_step",
                        result["content"],
                        "Task result" if message_type == "completion" else
                        "Navigation action" if is_displayable else "Background step"
                    )
                    
                    # Handle completion
                    if result["type"] == "completion":
                        completed = True
                        # Mark execution complete and show final status
                        await progress.update_progress(
                            stage="COMPLETE",
                            details="✅ Task completed successfully!"
                        )
                        # Extract and format final result for display
                        clean_result = result["content"].strip()
                        if len(clean_result.split('\n')) > 1:
                            # If multiple lines, extract key details
                            result_lines = [line for line in clean_result.split('\n') 
                                          if not line.startswith(('I ', 'The ', 'A ', 'An ')) 
                                          and len(line) > 10]
                            clean_result = ' '.join(result_lines) if result_lines else clean_result
                        
                        await update.message.reply_text(clean_result)
                        break
                    # Handle action messages - only display if they show meaningful progress
                    elif should_display_message(result["content"]):
                        self._update_execution_progress(progress, result["content"])

            if completed and team.execution_steps:
                # Update status for pattern phase
                pattern_status = "📝 Creating pattern from solution..." if not pattern_id else "✨ Updating existing pattern..."
                await progress.update_progress(
                    stage="STORING",
                    details=pattern_status
                )
                try:
                    # Convert execution steps to pattern format
                    pattern_steps = []
                    order_index = 0
                    
                    # Add task start step using initial task
                    initial_task = self._initial_tasks[user_id]
                    pattern_steps.append({
                        "type": "task_start",
                        "description": initial_task,  # Use initial task description
                        "context": {
                            "task_type": "web_navigation",
                            "user_id": user_id
                        },
                        "order": order_index
                    })
                    order_index += 1
                    
                    # Add assistant steps with richer context
                    for step in team.execution_steps:
                        if step.type == "assistant_step":
                            # Detect step type and enrich context
                            desc = step.description.lower()
                            step_type = "navigation"  # Default type
                            
                            if "go to" in desc or "navigate" in desc or "open" in desc:
                                step_type = "navigation"
                            elif "search" in desc or "find" in desc or "look for" in desc:
                                step_type = "search"
                            elif "extract" in desc or "collect" in desc or "summarize" in desc:
                                step_type = "analysis"
                            elif "click" in desc or "select" in desc or "choose" in desc:
                                step_type = "interaction"
                            elif "type" in desc or "enter" in desc or "input" in desc:
                                step_type = "input"
                                
                            # Enrich step context using utility function
                            step_context = enrich_step_context(step_type, desc)
                            
                            pattern_steps.append({
                                "type": step_type,
                                "description": step.description,
                                "context": step_context,
                                "order": order_index,
                                "semantic_details": {
                                    "type": step_type,
                                    "core_action": step_context["action_type"],
                                    "target_type": step_context["target_type"],
                                    "parameters": step_context["parameters"]
                                }
                            })
                            order_index += 1
                            
                    # Add completion step if present
                    completion_steps = [s for s in team.execution_steps if s.type == "completion"]
                    if completion_steps:
                        pattern_steps.append({
                            "type": "completion",
                            "description": completion_steps[0].description,
                            "context": {
                                "status": "success",
                                "timestamp": datetime.now().isoformat(),
                                "result": completion_steps[0].result
                            },
                            "order": order_index
                        })
                    
                    # Get execution time
                    execution_time = team.execution_steps[-1].timestamp if team.execution_steps else 0
                    
                    if pattern_id:  # If we used a pattern successfully
                        # Create task entry
                        task_data = {
                            "name": initial_task,
                            "description": initial_task,
                            "type": "web_navigation",
                            "timestamp": datetime.now().isoformat(),
                            "user_id": user_id
                        }
                        
                        # Link task to existing abstract pattern
                        if not pattern_id:
                            raise ValueError("Could not find abstract pattern ID in similar pattern data")
                            
                        new_task_id = await self.pattern_store.link_task_to_existing_pattern(
                            task=task_data,
                            pattern_id=pattern_id
                        )
                        logger.info(f"Linked task {new_task_id} to abstract pattern {pattern_id}")
                        
                        await self.pattern_store.update_solution_metrics(
                            task_id=new_task_id,
                            success=True,
                            execution_time=execution_time,
                            pattern_type='abstract'
                        )
                        
                        # Create stored pattern structure for existing pattern
                        abstract_steps = []
                        for step in similar_pattern_data.get("abstract_steps", []):
                            if isinstance(step, dict):
                                desc = step.get('description', '')
                                if desc:
                                    abstract_steps.append(desc)
                            else:
                                abstract_steps.append(step)

                        stored_pattern = {
                            'task_description': task,
                            'abstract_steps': abstract_steps,
                            'concrete_steps': [step['description'] for step in pattern_steps[:-1] if isinstance(step, dict) and 'description' in step]  # Exclude last step
                        }
                        
                        # Complete the task when using existing pattern
                        await progress.update_progress(
                            stage="STORING",
                            details="✨ Successfully updated solution pattern!"
                        )
                        await progress.update_progress(
                            stage="COMPLETE",
                            details="✅ Task completed! Pattern updated with your solution.",
                            final_pattern=stored_pattern
                        )
                        
                        logger.info(f"Linked successful task {new_task_id} to pattern {pattern_id}")
                    else:
                        # No pattern was used - create new pattern
                        # Skip intermediate status - will be covered by abstraction process
                        await progress.update_progress(
                            stage="STORING",
                            details="✨ Processing steps and creating pattern..."
                        )
                        
                        # Create pattern and get abstraction
                        pattern_result = await self._create_execution_pattern(
                            task=task,
                            execution_steps=pattern_steps,
                            task_logger=task_logger,
                            user_id=user_id
                        )
                        
                        if pattern_result and pattern_result.get('pattern_id') and pattern_result.get('abstracted'):
                            pattern_id = pattern_result['pattern_id']
                            # Create final pattern structure
                            # Extract descriptions from abstract steps and exclude last step from concrete
                            abstract_steps = []
                            for step in pattern_result['abstracted'].get('abstract_steps', []):
                                if isinstance(step, dict):
                                    desc = step.get('description', '')
                                    if desc:
                                        abstract_steps.append(desc)
                                else:
                                    abstract_steps.append(step)

                            final_pattern = {
                                'task_description': task,
                                'abstract_steps': abstract_steps,
                                'concrete_steps': [step['description'] for step in pattern_steps[:-1] if isinstance(step, dict) and 'description' in step]  # Exclude last step
                            }
                        
                        # Complete the task with new pattern details
                        await progress.update_progress(
                            stage="COMPLETE",
                            details="✅ New pattern created from your solution!",
                            final_pattern=final_pattern
                        )
                    
                except Exception as e:
                    error_msg = f"Error storing pattern: {str(e)}"
                    logger.error(error_msg)
                    await progress.update_progress(
                        stage="STORING",
                        details=f"❌ Error: {error_msg}"
                    )
                    # Log the error but don't re-raise
                    task_logger.log_execution_step({
                        "type": "pattern_creation_error",
                        "result": error_msg
                    })
                finally:
                    self._current_update = None
                    self._current_user_id = None

        except Exception as e:
            # Clear all task context on error
            if user_id in self._current_task:
                del self._current_task[user_id]
            if user_id in self._initial_tasks:
                del self._initial_tasks[user_id]
            error_msg = f"Error: {str(e)}"
            await update.message.reply_text(error_msg)
            logger.error(f"Task error: {error_msg}", exc_info=True)
            # Log failure if we have a task logger
            if 'task_logger' in locals():
                task_logger.log_execution_step({
                    "type": "task_error",
                    "result": error_msg
                })
                task_logger.set_result(False)
                task_logger.save()
        finally:
            self._current_update = None
            self._current_user_id = None

    async def _create_execution_pattern(self, task: str, execution_steps: List[Dict], task_logger: TaskLogger, user_id: int) -> Optional[Dict]:
        """Create pattern from execution steps using AbstractionAgent"""
        start_time = time.time()
        try:
            # Create pattern object
            initial_task = self._initial_tasks[user_id]  # Use original task
            pattern = {
                "name": f"Pattern from task {initial_task[:30]}...",
                "description": initial_task,  # Use initial task description
                "steps": execution_steps
            }
            
            # Use abstraction agent to get full abstraction
            abstracted = await self.abstraction_agent.abstract_pattern(pattern)
            
            if "error" in abstracted:
                raise Exception(abstracted["error"])
                
            # Store both concrete and abstract versions
            pattern_id = await self.pattern_store.store_task_with_solutions(
                task=abstracted["task"],
                concrete_steps=execution_steps,
                abstract_steps=abstracted["abstract_steps"]
            )
            
            logger.info(f"Created pattern {pattern_id} with {len(execution_steps)} steps")
            self._latest_pattern_id = pattern_id  # Store latest pattern ID
            return {
                'pattern_id': pattern_id,
                'abstracted': abstracted
            }
            
        except Exception as e:
            error_msg = f"Pattern creation failed: {str(e)}"
            logger.error(error_msg)
            task_logger.log_execution_step({
                "type": "pattern_creation_error",
                "result": error_msg,
                "time_taken": f"{time.time() - start_time:.2f}s"
            })
            return None

    async def start(self) -> None:
        """Start the Telegram bot"""
        try:
            # Initialize components
            await self.initialize()
            
            # Set up Telegram bot
            token = os.getenv("TELEGRAM_TOKEN")
            if not token:
                raise EnvironmentError("TELEGRAM_TOKEN must be set")

            self._app = ApplicationBuilder().token(token).build()
            
            # Add handlers
            self._app.add_handler(CommandHandler("start", self.start_command))
            self._app.add_handler(CommandHandler("clear", self.clear_command))
            self._app.add_handler(MessageHandler(
                filters.TEXT & ~filters.COMMAND,
                self.handle_message
            ))

            # Start the bot
            await self._app.initialize()
            await self._app.start()
            await self._app.updater.start_polling()
            
            logger.info("Bot started and ready")
            
        except Exception as e:
            logger.error(f"Failed to start bot: {str(e)}")
            # Clean up any initialized components
            await self.stop()
            raise

    async def stop(self) -> None:
        """Stop the Telegram bot"""
        if self._app:
            try:
                if self._app.updater and self._app.updater.running:
                    await self._app.updater.stop()
                if self._app.running:
                    await self._app.stop()
                await self._app.shutdown()
            except Exception as e:
                logger.error(f"Error stopping bot: {str(e)}")
