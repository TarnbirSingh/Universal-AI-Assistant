import logging
from typing import AsyncGenerator, Dict, List, Optional, Sequence
from sentence_transformers import SentenceTransformer
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import AgentEvent, ChatMessage, TextMessage
from autogen_core import CancellationToken

from src.pattern_matching.task_analyzer import TaskAnalyzer
from src.kb.pattern_store import PatternStore

logger = logging.getLogger(__name__)

class NumpyFloatEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        return super().default(obj)

class PatternSearchAgent(BaseChatAgent):
    """Agent that finds similar task patterns using semantic similarity"""
    
    def __init__(self, name: str, neo4j_client: PatternStore, task_analyzer: Optional[TaskAnalyzer] = None):
        super().__init__(name, "Agent that finds similar task patterns and returns their abstract solutions")
        self.neo4j = neo4j_client
        self.task_analyzer = task_analyzer or TaskAnalyzer()
        # Initialize without SentenceTransformer for now - use fallback similarity
        self.encoder = None
        logger.info("Using fallback word-overlap similarity calculation")
            
        self.similarity_threshold = 0.3  # Lowered to 30%
        
    @property
    def produced_message_types(self) -> Sequence[type[ChatMessage]]:
        return (TextMessage,)

    async def on_messages(self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken) -> Response:
        response: Response | None = None
        async for message in self.on_messages_stream(messages, cancellation_token):
            if isinstance(message, Response):
                response = message
        assert response is not None
        return response

    def _format_steps(self, steps: List[Dict]) -> str:
        """Format steps into a numbered list"""
        # Sort steps by order
        sorted_steps = sorted(steps, key=lambda x: x.get('order', 0))
        
        # Format as numbered list
        step_list = []
        for i, step in enumerate(sorted_steps, 1):
            step_list.append(f"{i}) {step['description']}")
        
        return "\n".join(step_list)

    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts using embeddings"""
        def normalize_text(text: str) -> str:
            """Normalize text by handling common variants"""
            text = text.lower()
            # Handle common variants
            replacements = {
                'information': ['info', 'facts', 'details', 'data'],
                'navigate': ['go to', 'open', 'access', 'visit'],
                'find': ['search', 'look for', 'locate', 'get'],
                'show': ['display', 'give', 'tell', 'present'],
                'system': ['cloud', 'app', 'portal', 'site']
            }
            for main, variants in replacements.items():
                for variant in variants:
                    text = text.replace(variant, main)
            return text

        def fallback_similarity(t1: str, t2: str) -> float:
            """Calculate fallback similarity using normalized word overlap"""
            # Normalize texts
            t1 = normalize_text(t1)
            t2 = normalize_text(t2)
            
            # Get word sets
            w1 = set(t1.split())
            w2 = set(t2.split())
            
            if not w1 or not w2:
                return 0.0
                
            intersection = w1.intersection(w2)
            union = w1.union(w2)
            
            # Give more weight to matches in shorter text
            base_sim = len(intersection) / len(union)
            short_text_bonus = min(len(w1), len(w2)) / max(len(w1), len(w2))
            
            return base_sim * (1 + 0.2 * short_text_bonus)  # Up to 20% bonus for matching shorter text

        # If no encoder, use fallback
        if self.encoder is None:
            return fallback_similarity(text1, text2)
            
        try:
            # Try using encoder for embeddings
            emb1 = self.encoder.encode([text1])[0]
            emb2 = self.encoder.encode([text2])[0]
        
            # Reshape for sklearn
            emb1 = emb1.reshape(1, -1)
            emb2 = emb2.reshape(1, -1)
            
            # Calculate cosine similarity
            sim = float(cosine_similarity(emb1, emb2)[0][0])
        except Exception as e:
            logger.error(f"Error in semantic similarity calculation: {str(e)}")
            # Fallback to basic text matching if embedding fails
            return fallback_similarity(text1, text2)
        
        # Apply semantic boosting
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        shared_words = words1.intersection(words2)
        
        
        # Boost similarity if key action words are shared
        action_words = {
            "search", "find", "get", "create", "view", "open", "navigate", "access",
            "look", "lookup", "read", "check", "locate", "browse", "explore",
            "show", "display", "see", "list", "fetch", "retrieve", "go", "enter",
            "load", "visit", "bring", "give", "tell", "provide", "present"
        }
        shared_actions = shared_words.intersection(action_words)
        if shared_actions:
            sim = sim * (1 + 0.05 * len(shared_actions))  # 5% boost per shared action word
            
        # Apply domain boost
        domain_terms = {
            "sap", "s/4hana", "cloud", "wiki", "wikipedia", "customer", "master", "data",
            "system", "database", "information", "details", "records", "document", "object",
            "product", "vendor", "sales", "order", "business", "account", "client",
            "info", "facts", "portal", "screen", "page", "site", "app", "application"
        }
        shared_domains = shared_words.intersection(domain_terms)
        if shared_domains:
            sim = sim * (1 + 0.03 * len(shared_domains))  # 3% boost per shared domain term
        
        return min(float(sim), 1.0)  # Cap at 1.0 and ensure Python float

    async def on_messages_stream(
        self,
        messages: Sequence[ChatMessage],
        cancellation_token: CancellationToken
    ) -> AsyncGenerator[AgentEvent | ChatMessage | Response, None]:
        try:
            # Extract task description from last message
            task_desc = messages[-1].content if messages else ""
            
            # Analyze task
            task_analysis = await self.task_analyzer.analyze_task(task_desc)
            
            # Find similar patterns
            try:
                patterns = await self._find_similar_patterns(task_desc, task_analysis)
            except Exception as e:
                logger.error(f"Error finding similar patterns: {str(e)}")
                yield TextMessage(
                    content=f"Error: {str(e)}",
                    source=self.name
                )
                return
            
            if not patterns:
                yield TextMessage(
                    content=f"No similar patterns found with >={self.similarity_threshold*100:.0f}% similarity threshold.",
                    source=self.name
                )
                return
            
            # Sort by similarity score
            patterns.sort(key=lambda x: x["similarity"], reverse=True)
            best_match = patterns[0]
            
            # Get abstract pattern for best match
            try:
                task_id = best_match["pattern_id"]
                # Get both abstract and concrete pattern data
                pattern_data = await self.neo4j.get_task_with_solutions(task_id)
                
                if not pattern_data:
                    raise ValueError("Could not retrieve pattern data")
                    
                abstract_pattern_info = pattern_data.get("abstract_pattern", {})
                if not abstract_pattern_info:
                    raise ValueError("Could not find abstract pattern information")
                
                abstract_steps = pattern_data.get("abstract_steps", [])
                concrete_steps = pattern_data.get("concrete_steps", [])
                if not abstract_steps:
                    raise ValueError(f"No abstract steps found for pattern {task_id}")
                    
                # Format steps in ordered list
                formatted_abstract_steps = self._format_steps(abstract_steps)
                
                # Get concrete examples (descriptions from concrete steps)
                concrete_examples = []
                for step in concrete_steps:
                    if step.get('type') != 'task_start' and step.get('type') != 'completion':
                        desc = step.get('description', '').strip()
                        if desc:
                            concrete_examples.append(desc)
                
                response_data = {
                    "similarity": float(best_match['similarity']),  # Ensure Python float
                    "task_id": task_id,
                    "abstract_pattern_id": abstract_pattern_info["id"],
                    "task_type": pattern_data.get("task", {}).get("type", "unknown"),
                    "abstract_pattern": formatted_abstract_steps,
                    "abstract_pattern_name": abstract_pattern_info.get("name", "Unknown Pattern"),
                    "concrete_examples": concrete_examples[:5]  # Include up to 5 concrete examples
                }
                
                # Format response text
                response_text = [
                    f"Found similar pattern ({response_data['similarity']:.0%} match)",
                    "Abstract Solution Pattern:",
                    formatted_abstract_steps,
                    "\nDetailed Pattern Data:",
                    json.dumps(response_data, indent=2, cls=NumpyFloatEncoder)
                ]
                
                yield TextMessage(
                    content="\n".join(response_text),
                    source=self.name
                )
            except Exception as e:
                logger.error(f"Error retrieving pattern data: {str(e)}")
                yield TextMessage(
                    content=f"Error: {str(e)}",
                    source=self.name
                )
            
        except Exception as e:
            logger.error(f"Error in pattern search: {str(e)}")
            yield TextMessage(
                content=f"Error: {str(e)}",
                source=self.name
            )
            
    async def _find_similar_patterns(
        self, 
        task_desc: str,
        task_analysis: Dict
    ) -> List[Dict]:
        """Find patterns similar to the given task using semantic similarity"""
        try:
            # Query Neo4j for tasks with abstract patterns
            query = """
            MATCH (t:Task)-[:HAS_ABSTRACT_PATTERN]->(p:Pattern)
            WHERE p.type = 'abstract'
            RETURN t.id as id, t.description as description, t.type as type, p.id as pattern_id
            """
            
            similar_patterns = []
            
            with self.neo4j.driver.session() as session:
                results = session.run(query)
                
                for record in results:
                    pattern_desc = record["description"]
                    if not pattern_desc:
                        continue
                    
                    # Calculate semantic similarity
                    similarity = self._calculate_semantic_similarity(task_desc, pattern_desc)
                    
                    # Check if meets threshold
                    if similarity >= self.similarity_threshold:
                        similar_patterns.append({
                            "pattern_id": record["id"],  # Task ID for fetching full pattern
                            "abstract_pattern_id": record["pattern_id"],  # Abstract pattern ID for linking
                            "similarity": float(similarity),  # Ensure Python float
                            "type": record.get("type", "unknown")
                        })
                        
            return similar_patterns
            
        except Exception as e:
            logger.error(f"Error finding similar patterns: {str(e)}")
            raise

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        """Reset agent state"""
        pass
