import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from tqdm import tqdm

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.knowledge_graph.builder import KnowledgeGraphBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('knowledge_graph_build.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

async def load_papers(data_dir: Path) -> list:
    """Load papers from the dataset."""
    papers = []
    try:
        # Load papers from metadata
        metadata_file = data_dir / "metadata" / "papers_metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
            
        with open(metadata_file, 'r', encoding='utf-8') as f:
            papers_metadata = json.load(f)
            
        logger.info(f"Found {len(papers_metadata)} papers in metadata")
        
        # Load full paper contents
        for paper in tqdm(papers_metadata, desc="Loading papers"):
            paper_path = data_dir / paper.get('pdf_path', '')
            if paper_path.exists():
                try:
                    with open(paper_path, 'r', encoding='utf-8') as f:
                        paper['content'] = f.read()
                except Exception as e:
                    logger.warning(f"Error reading paper {paper_path}: {str(e)}")
                    paper['content'] = ''
            else:
                logger.warning(f"Paper file not found: {paper_path}")
                paper['content'] = ''
                
            papers.append(paper)
            
        return papers
        
    except Exception as e:
        logger.error(f"Error loading papers: {str(e)}")
        return []

async def main():
    try:
        # Setup paths
        data_dir = project_root / "data"
        output_dir = project_root / "output" / "knowledge_graph"
        
        # Load papers
        logger.info("Loading papers from dataset...")
        papers = await load_papers(data_dir)
        
        if not papers:
            logger.error("No papers loaded. Exiting.")
            return
            
        logger.info(f"Successfully loaded {len(papers)} papers")
        
        # Initialize knowledge graph builder
        builder = KnowledgeGraphBuilder(str(output_dir))
        
        # Build knowledge graph
        logger.info("Building knowledge graph...")
        success = await builder.build_graph(papers)
        
        if success:
            logger.info("Knowledge graph built successfully!")
            logger.info(f"Output files available in: {output_dir}")
            
            # Load and display statistics
            stats_file = output_dir / 'graph_statistics.json'
            if stats_file.exists():
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                    
                logger.info("\nKnowledge Graph Statistics:")
                logger.info(f"Number of entities: {stats['num_entities']}")
                logger.info(f"Number of relationships: {stats['num_relationships']}")
                logger.info("\nEntity types:")
                for entity_type, count in stats['entity_types'].items():
                    logger.info(f"  {entity_type}: {count}")
                logger.info("\nRelationship types:")
                for rel_type, count in stats['relationship_types'].items():
                    logger.info(f"  {rel_type}: {count}")
                logger.info(f"\nAverage confidence: {stats['avg_confidence']:.2%}")
        else:
            logger.error("Failed to build knowledge graph")
            
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
