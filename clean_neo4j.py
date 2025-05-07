from neo4j import GraphDatabase
import logging
from typing import Optional, Dict, List
from setup_neo4j_indexes import setup_neo4j_indexes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            "bolt://localhost:7687",
            auth=("neo4j", "password123"),
            database="neo4j"
        )

    def clean_database(self, preserve_patterns: bool = False):
        """Clean database with option to preserve successful patterns"""
        try:
            with self.driver.session() as session:
                if preserve_patterns:
                    # Store successful patterns before cleaning
                    patterns = self._extract_successful_patterns(session)
                    
                    # Clean database
                    session.run("MATCH (n) DETACH DELETE n")
                    logger.info("Database cleaned")
                    
                    # Restore successful patterns
                    self._restore_patterns(session, patterns)
                    logger.info(f"Restored {len(patterns)} successful patterns")
                else:
                    # Clean everything
                    session.run("MATCH (n) DETACH DELETE n")
                    logger.info("Database completely cleaned")
                
                # Rebuild indexes
                setup_neo4j_indexes()
                logger.info("Indexes rebuilt")
                
        except Exception as e:
            logger.error(f"Error during database cleaning: {str(e)}")
        finally:
            self.close()

    def _extract_successful_patterns(self, session) -> List[Dict]:
        """Extract patterns with high success rate"""
        result = session.run("""
            MATCH (p:Pattern)-[:HAS_STEP]->(s)
            WHERE p.success_rate > 0.7 AND p.uses > 5
            WITH p, collect(s) as steps
            RETURN {
                name: p.name,
                description: p.description,
                abstraction_level: p.abstraction_level,
                core_concept: p.core_concept,
                input_type: p.input_type,
                output_type: p.output_type,
                operations: p.operations,
                constraints: p.constraints,
                success_rate: p.success_rate,
                uses: p.uses,
                steps: steps
            } as pattern
        """)
        return [record["pattern"] for record in result]

    def _restore_patterns(self, session, patterns: List[Dict]):
        """Restore extracted patterns to database"""
        for pattern in patterns:
            # Create pattern node
            result = session.run("""
                CREATE (p:Pattern {
                    name: $name,
                    description: $description,
                    abstraction_level: $abstraction_level,
                    core_concept: $core_concept,
                    input_type: $input_type,
                    output_type: $output_type,
                    operations: $operations,
                    constraints: $constraints,
                    success_rate: $success_rate,
                    uses: $uses
                })
                RETURN elementId(p) as id
            """, pattern)
            
            pattern_id = result.single()["id"]
            
            # Create and link steps
            prev_step_id = None
            for step in pattern["steps"]:
                step_result = session.run("""
                    MATCH (p:Pattern)
                    WHERE elementId(p) = $pattern_id
                    CREATE (s:Step {
                        type: $type,
                        description: $description,
                        order: $order
                    })
                    CREATE (p)-[:HAS_STEP]->(s)
                    RETURN elementId(s) as id
                """, {
                    "pattern_id": pattern_id,
                    "type": step["type"],
                    "description": step["description"],
                    "order": step["order"]
                })
                
                current_step_id = step_result.single()["id"]
                
                # Link steps in sequence
                if prev_step_id:
                    session.run("""
                        MATCH (prev:Step), (curr:Step)
                        WHERE elementId(prev) = $prev_id 
                        AND elementId(curr) = $curr_id
                        CREATE (prev)-[:NEXT]->(curr)
                    """, {
                        "prev_id": prev_step_id,
                        "curr_id": current_step_id
                    })
                    
                prev_step_id = current_step_id

    def close(self):
        """Close the database connection"""
        self.driver.close()

if __name__ == "__main__":
    db_manager = DatabaseManager()
    # By default, preserve successful patterns when cleaning
    db_manager.clean_database(preserve_patterns=True)
