import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import logging configuration first
from src.knowledge_graph.logging_config import setup_logging, get_logger

# Configure logging before other imports
setup_logging(log_level='DEBUG')
logger = get_logger(__name__)

import asyncio
from src.knowledge_graph.advanced_entity_extractor import AdvancedEntityExtractor
from src.knowledge_graph.builder import KnowledgeGraphBuilder

# Sample scientific texts for entity extraction
SAMPLE_TEXTS = [
    "Quantum entanglement is a fundamental principle of quantum mechanics where two particles become correlated in such a way that the quantum state of each particle cannot be described independently.",
    
    "The development of warp drive technology could revolutionize space exploration by enabling faster-than-light travel through the manipulation of spacetime geometry.",
    
    "Gravitational wave detection using advanced laser interferometry has opened up new frontiers in observational astrophysics, allowing scientists to study cosmic events like black hole mergers.",
    
    "Experimental quantum computing platforms are rapidly advancing, with superconducting qubits and topological quantum computation emerging as promising approaches to scalable quantum information processing.",
    
    "The interplay between quantum mechanics and general relativity remains a critical challenge in theoretical physics, with potential implications for our understanding of fundamental spacetime structure."
]

async def demonstrate_advanced_entity_extraction():
    """
    Demonstrate advanced entity extraction capabilities.
    """
    try:
        # Initialize advanced entity extractor
        logger.info("Initializing Advanced Entity Extractor")
        extractor = AdvancedEntityExtractor()
        
        # Extract entities from sample texts
        logger.info("Starting advanced entity extraction demonstration...")
        
        for text in SAMPLE_TEXTS:
            logger.info(f"\n--- Extracting Entities from Text: ---\n{text}\n")
            
            # Extract entities
            entities = extractor.extract_entities(text)
            
            # Display extracted entities
            for entity in entities:
                logger.info(f"Entity: {entity.name}")
                logger.info(f"Type: {entity.type.value}")
                logger.info(f"Properties: {entity.properties}\n")
        
        # Demonstrate knowledge graph construction
        logger.info("Demonstrating Knowledge Graph Construction...")
        
        # Prepare sample papers
        sample_papers = [
            {
                'title': 'Quantum Mechanics and Warp Drive Theory',
                'abstract': SAMPLE_TEXTS[0] + ' ' + SAMPLE_TEXTS[1],
                'content': ' '.join(SAMPLE_TEXTS)
            }
        ]
        
        # Initialize knowledge graph builder
        graph_builder = KnowledgeGraphBuilder()
        
        # Build knowledge graph
        graph = await graph_builder.build_graph(sample_papers)
        
        if graph:
            logger.info(f"Knowledge Graph Created:")
            logger.info(f"Nodes: {graph.number_of_nodes()}")
            logger.info(f"Edges: {graph.number_of_edges()}")
        else:
            logger.warning("Failed to create knowledge graph")
    
    except Exception as e:
        logger.error(f"Error in entity extraction demonstration: {e}", exc_info=True)

async def main():
    """Main async entry point."""
    logger.info("Starting Advanced Entity Extraction Demonstration")
    await demonstrate_advanced_entity_extraction()
    logger.info("Demonstration Complete")

if __name__ == '__main__':
    asyncio.run(main())
