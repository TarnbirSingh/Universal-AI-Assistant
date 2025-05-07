from typing import Optional, Dict, Any, List
from telegram import Bot
import logging

logger = logging.getLogger(__name__)

class ProgressTracker:
    """
    Track and display task progress with user-friendly messages.
    No progress bars or percentages, just clear status messages.
    """
    
    STAGES = {
        "ANALYZING": {
            "emoji": "🧠",
            "message": "Analyzing and executing your task..."
        },
        "EXECUTING": {  # Combined stage
            "emoji": "⚡️",
            "message": "Executing task",
            "substages": {
                "NAVIGATION": "🌐 Navigating web page",
                "INTERACTION": "👆 Interacting with page",
                "EXTRACTION": "📝 Collecting information",
                "PROCESSING": "⚙️ Processing data"
            }
        },
        "STORING": {
            "emoji": "💾",
            "message": "Saving results..."
        },
        "COMPLETE": {
            "emoji": "✨",
            "message": "Task completed!"
        }
    }

    def __init__(self, bot: Bot, chat_id: int):
        self.bot = bot
        self.chat_id = chat_id
        self.message_id: Optional[int] = None
        self.current_stage = "ANALYZING"
        self.current_substage: Optional[str] = None
        
        # Task info
        self.task_description = ""
        
        # Pattern matching info
        self.pattern_found = False
        self.pattern_similarity: Optional[float] = None
        self.similar_task_description: Optional[str] = None
        self.pattern_abstract_steps: List[str] = []
        self.pattern_concrete_steps: List[str] = []
        
        # Final pattern info
        self.final_pattern: Optional[Dict[str, Any]] = None

    def _format_progress_message(self, details: Optional[str] = None) -> str:
        """Format a user-friendly progress message"""
        stage_info = self.STAGES[self.current_stage]
        
        # Start with task description
        message_parts = [
            f"🎯 Task: {self.task_description}",
            ""
        ]
        
        # Add pattern match info during execution
        if self.current_stage == "EXECUTING" and self.pattern_found:
            # Pattern match section
            similarity_percent = int(self.pattern_similarity * 100) if self.pattern_similarity else 0
            message_parts.extend([
                f"🌟 Knowledge Base Match Found! ({similarity_percent}% similarity)",
                f'Similar task: "{self.similar_task_description}"' if self.similar_task_description else "",
                ""
            ])
            
            # Pattern abstract steps
            if self.pattern_abstract_steps:
                for i, step in enumerate(self.pattern_abstract_steps, 1):
                    if isinstance(step, dict):
                        desc = step.get('description', '')
                        if desc:
                            message_parts.append(f"{i}) {desc}")
                    else:
                        message_parts.append(f"{i}) {step}")
            
            message_parts.append("")
        elif self.current_stage == "EXECUTING" and not self.pattern_found:
            message_parts.append("📝 Creating new solution pattern...")
        
        # Status line
        if self.current_stage == "PATTERN_SEARCH":
            message_parts.append(f"⚡️ Status: {details if details else 'Searching for similar patterns...'}")
        elif self.current_stage == "EXECUTING":
            if self.pattern_found:
                message_parts.append(f"⚡️ Status: Using existing pattern to complete task")
            else:
                message_parts.append(f"⚡️ Status: {stage_info['message']}")
        else:
            message_parts.append(f"⚡️ Status: {stage_info['message']}")
            
        # Add substage if in execution phase
        if self.current_substage and "substages" in stage_info:
            substage_info = stage_info["substages"].get(self.current_substage, "")
            if substage_info:
                message_parts.append(substage_info)
        
        # Add final pattern details if complete
        if self.current_stage == "COMPLETE" and self.final_pattern:
            message_parts.extend([
                "",
                "📝 Pattern created from your solution:" if not self.pattern_found else "📝 Pattern updated with your solution:",
                f"Task: {self.final_pattern.get('task_description', 'N/A')}",
                "",
                "Abstract Steps:"
            ])
            
            # Final abstract steps - extract descriptions from steps
            abstract_steps = self.final_pattern.get('abstract_steps', [])
            for i, step in enumerate(abstract_steps, 1):
                if isinstance(step, dict):
                    desc = step.get('description', str(step))
                    # Clean up description if it's still a dict
                    if isinstance(desc, dict):
                        desc = desc.get('description', str(desc))
                    message_parts.append(f"{i}) {desc}")
                else:
                    message_parts.append(f"{i}) {step}")
                
            message_parts.extend([
                "",
                "Concrete Steps:"
            ])
            
            # Final concrete steps - exclude the last step (completion)
            concrete_steps = self.final_pattern.get('concrete_steps', [])[:-1]  # Exclude last step
            for i, step in enumerate(concrete_steps, 1):
                if isinstance(step, dict):
                    desc = step.get('description', str(step))
                    message_parts.append(f"{i}) {desc}")
                else:
                    message_parts.append(f"{i}) {step}")
        
        # Add any additional details
        if details:
            message_parts.append(f"\n{details}")
            
        return "\n".join(message_parts)

    async def update_progress(self, 
                            stage: Optional[str] = None,
                            substage: Optional[str] = None,
                            details: Optional[str] = None,
                            # Pattern match params
                            pattern_found: Optional[bool] = None,
                            pattern_similarity: Optional[float] = None,
                            similar_task_description: Optional[str] = None,
                            pattern_abstract_steps: Optional[List[str]] = None,
                            pattern_concrete_steps: Optional[List[str]] = None,
                            # Final pattern params
                            final_pattern: Optional[Dict[str, Any]] = None) -> None:
        """Update progress message with new status"""
        try:
            # Update state
            if stage:
                self.current_stage = stage
            if substage:
                self.current_substage = substage
                
            # Update pattern match info
            if pattern_found is not None:
                self.pattern_found = pattern_found
            if pattern_similarity is not None:
                self.pattern_similarity = pattern_similarity
            if similar_task_description is not None:
                self.similar_task_description = similar_task_description
            if pattern_abstract_steps is not None:
                self.pattern_abstract_steps = pattern_abstract_steps
            if pattern_concrete_steps is not None:
                self.pattern_concrete_steps = pattern_concrete_steps
                
            # Update final pattern
            if final_pattern is not None:
                self.final_pattern = final_pattern
            
            # Format and send message
            message = self._format_progress_message(details)
            
            if not self.message_id:
                sent_message = await self.bot.send_message(
                    chat_id=self.chat_id,
                    text=message
                )
                self.message_id = sent_message.message_id
            else:
                await self.bot.edit_message_text(
                    text=message,
                    chat_id=self.chat_id,
                    message_id=self.message_id
                )
        except Exception as e:
            # Log error but don't raise to prevent task interruption
            logger.error(f"Error updating progress message: {e}")
