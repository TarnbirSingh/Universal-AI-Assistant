from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
from collections import Counter
import logging

from .task_analyzer import TaskAnalysis

logger = logging.getLogger(__name__)

@dataclass
class PatternMatch:
    pattern_id: str
    score: float
    adaptations_needed: List[str]
    pattern: Dict
    analysis: Dict

class AlgorithmicMatcher:
    """Fast pattern matching using algorithmic approaches"""
    
    def __init__(self):
        # Common words to ignore in similarity comparison
        self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'can', 'you', 'please'}
        self.word_weight = 0.00  # Minimal weight for word similarity
        self.action_weight = 0.00  # Minimal weight for action matching
        self.similarity_threshold = 0.00  # Very low threshold to match almost anything
    
    def _get_word_vector(self, text: str) -> Counter:
        # Convert text to lowercase and split into words
        words = text.lower().split()
        # Remove stop words and create word frequency vector
        words = [w for w in words if w not in self.stop_words]
        return Counter(words)
    
    def _cosine_similarity_counters(self, counter1: Counter, counter2: Counter) -> float:
        # Get all unique words
        all_words = set(counter1.keys()) | set(counter2.keys())
        
        # Convert counters to vectors using the same word ordering
        vec1 = np.array([counter1.get(word, 0) for word in all_words])
        vec2 = np.array([counter2.get(word, 0) for word in all_words])
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return float(dot_product / (norm1 * norm2))
    
    async def find_candidates(
        self,
        task_analysis: TaskAnalysis,
        stored_patterns: List[Dict],
        limit: int = 10
    ) -> List[Dict]:
        """Find candidate patterns using fast algorithmic matching"""
        # Convert task description to word vector
        task_word_vector = self._get_word_vector(task_analysis.task_description)
        
        matches = []
        for pattern in stored_patterns:
            # Get word vector for pattern
            pattern_word_vector = self._get_word_vector(pattern.get('description', ''))
            
            # Calculate word similarity
            word_similarity = self._cosine_similarity_counters(
                task_word_vector,
                pattern_word_vector
            )
            
            # Compare actions
            pattern_actions = set(pattern.get('actions', []))
            task_actions = set(task_analysis.actions)
            
            action_match = len(
                pattern_actions.intersection(task_actions)
            ) / max(len(pattern_actions), len(task_actions)) if pattern_actions and task_actions else 0.0
            
            # Check for key action pairs
            action_boost = 0.0
            key_pairs = [{'navigate', 'find'}, {'find', 'show'}, {'navigate', 'show'}]
            for pair in key_pairs:
                if pair.issubset(task_actions) and pair.issubset(pattern_actions):
                    action_boost = 0.2
                    break
            
            # Calculate final score
            score = (word_similarity * self.word_weight) + (action_match * self.action_weight) + action_boost
            
            if score >= self.similarity_threshold:
                pattern_copy = pattern.copy()
                pattern_copy['score'] = score
                pattern_copy['word_similarity'] = word_similarity
                pattern_copy['action_match'] = action_match
                matches.append(pattern_copy)
        
        # Sort by score and return top matches
        matches.sort(key=lambda x: x['score'], reverse=True)
        return matches[:limit]

class HybridPatternMatcher:
    """Pattern matcher combining algorithmic and AI approaches"""
    
    def __init__(self, neo4j_client, llm_service):
        self.neo4j = neo4j_client
        self.llm = llm_service
        self.algorithmic_matcher = AlgorithmicMatcher()
        
    async def find_patterns(self, task_analysis: TaskAnalysis) -> List[PatternMatch]:
        """Find matching patterns using hybrid approach"""
        try:
            # Get patterns from database
            patterns = await self._get_stored_patterns()
            
            # 1. Get initial candidates using fast algorithmic matching
            candidates = await self.algorithmic_matcher.find_candidates(task_analysis, patterns)
            logger.info(f"Found {len(candidates)} initial candidates")
            
            if not candidates:
                logger.info("No matching patterns found")
                return []
            
            # 2. Detailed analysis of top candidates
            matches = []
            for candidate in candidates[:5]:  # Analyze top 5 candidates
                # Get AI analysis of pattern applicability
                analysis = await self._analyze_pattern_match(task_analysis, candidate)
                
                matches.append(PatternMatch(
                    pattern_id=candidate["id"],
                    score=candidate["score"],
                    adaptations_needed=analysis.get("adaptations", []),
                    pattern=candidate,
                    analysis=analysis
                ))
            
            # Sort by score
            matches.sort(key=lambda x: x.score, reverse=True)
            return matches
            
        except Exception as e:
            logger.error(f"Error in pattern matching: {str(e)}")
            raise
    
    async def _get_stored_patterns(self) -> List[Dict]:
        """Get all patterns from database"""
        query = """
        MATCH (p:Pattern)
        RETURN p
        """
        result = await self.neo4j.execute_query(query)
        return [record["p"] for record in result]
    
    async def _analyze_pattern_match(
        self,
        task_analysis: TaskAnalysis,
        pattern: Dict
    ) -> Dict:
        """Use LLM to analyze pattern applicability"""
        try:
            # Create analysis prompt
            prompt = self._create_analysis_prompt(task_analysis, pattern)
            
            # Get LLM analysis
            analysis = await self.llm.analyze(prompt)
            
            # Add algorithmic scores to analysis
            analysis["word_similarity"] = pattern.get("word_similarity", 0.0)
            analysis["action_match"] = pattern.get("action_match", 0.0)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in pattern analysis: {str(e)}")
            return {
                "confidence": 0.0,
                "explanation": f"Analysis failed: {str(e)}",
                "adaptations": [],
                "word_similarity": pattern.get("word_similarity", 0.0),
                "action_match": pattern.get("action_match", 0.0)
            }
    
    def _create_analysis_prompt(self, task_analysis: TaskAnalysis, pattern: Dict) -> str:
        """Create prompt for LLM analysis"""
        return f"""
        Analyze the applicability of this pattern to the given task:
        
        Task Details:
        - Description: {task_analysis.task_description}
        - Actions: {task_analysis.actions}
        - Requirements: {task_analysis.requirements}
        - Constraints: {task_analysis.constraints}
        
        Pattern:
        - Description: {pattern.get('description', '')}
        - Core Actions: {pattern.get('actions', [])}
        - Requirements: {pattern.get('requirements', {})}
        - Previous Success Rate: {pattern.get('success_rate', 0.5)}
        - Word Similarity Score: {pattern.get('word_similarity', 0.0)}
        - Action Match Score: {pattern.get('action_match', 0.0)}
        
        Provide:
        1. Confidence score (0-1) for pattern applicability
        2. Explanation of reasoning
        3. Required adaptations to make pattern work for this task
        """
