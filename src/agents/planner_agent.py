from autogen_agentchat.agents import AssistantAgent

class PlannerAgent(AssistantAgent):
    """Agent responsible for breaking down tasks into steps and planning execution"""
    
    def __init__(self, model, memory):
        super().__init__(
            name="planner",
            model_client=model,
            memory=[memory],
            handoffs=["coordinator", "executor", "user", "vision"],
            system_message="""You are the task planning expert.
            Your role is to:
            1. Analyze the task requirements
            2. Check other agent's capabilities and tools and propose a clear approach to solve the task
            3. Break down the task into clear steps
            4. Assign the first subtask to the best suitable agent
            If you need user clarification, hand it back to the coordinator with request to ask the user."""
        )
