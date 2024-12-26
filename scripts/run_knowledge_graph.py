import os
import sys
import json
import logging
from pathlib import Path
from tqdm import tqdm
import asyncio

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.knowledge_graph.entity_extractor import EntityExtractor
from src.knowledge_graph.relationship_mapper import RelationshipMapper
from src.knowledge_graph.builder import KnowledgeGraphBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('knowledge_graph.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_papers(data_dir: Path) -> list:
    """Load papers from the dataset directory."""
    papers = []
    try:
        # Load papers from metadata
        metadata_file = data_dir / "papers" / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                papers_metadata = json.load(f)
                
            logger.info(f"Found {len(papers_metadata)} papers in metadata")
            
            # Load full paper contents
            for paper in tqdm(papers_metadata, desc="Loading papers"):
                paper_path = data_dir / "papers" / paper['filename']
                if paper_path.exists():
                    try:
                        with open(paper_path, 'r', encoding='utf-8') as f:
                            paper['content'] = f.read()
                    except Exception as e:
                        logger.warning(f"Error reading paper {paper_path}: {str(e)}")
                        paper['content'] = ''
                else:
                    logger.warning(f"Paper file not found: {paper_path}")
                papers.append(paper)
        else:
            logger.error(f"Metadata file not found: {metadata_file}")
    except Exception as e:
        logger.error(f"Error loading papers: {str(e)}")
    
    return papers

async def main():
    """Main async function to build knowledge graph."""
    try:
        # Set up paths
        project_root = Path(__file__).parent.parent
        data_dir = project_root / "data"
        output_dir = project_root / "outputs" / "knowledge_graph"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load papers
        papers = load_papers(data_dir)
        
        if not papers:
            logger.error("No papers loaded. Exiting.")
            return
        
        # Initialize knowledge graph builder
        builder = KnowledgeGraphBuilder(str(output_dir))
        
        # Build knowledge graph
        graph = await builder.build_graph(papers)
        
        if graph is None:
            logger.error("Failed to build knowledge graph")
            return
        
        # Log graph statistics
        logger.info(f"Knowledge Graph Statistics:")
        logger.info(f"Number of Nodes: {graph.number_of_nodes()}")
        logger.info(f"Number of Edges: {graph.number_of_edges()}")
        
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
