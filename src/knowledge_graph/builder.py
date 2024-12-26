import logging
from typing import List, Dict, Optional, Union, Generator, Any
from pathlib import Path
import json
import asyncio
from .schema import KnowledgeGraphSchema, Entity, Relationship, EntityType
from .advanced_entity_extractor import AdvancedEntityExtractor
from .relationship_mapper import RelationshipMapper
from .graph_versioning import GraphVersionControl
from .logging_config import get_logger, log_performance
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

logger = get_logger(__name__)

class KnowledgeGraphBuilder:
    def __init__(
        self, 
        output_dir: Union[str, Path] = 'knowledge_graph_output', 
        extractor: Optional[AdvancedEntityExtractor] = None,
        mapper: Optional[RelationshipMapper] = None
    ):
        """
        Initialize the knowledge graph builder.
        
        Args:
            output_dir: Directory to save knowledge graph outputs
            extractor: Custom entity extractor (defaults to AdvancedEntityExtractor)
            mapper: Custom relationship mapper
        """
        # Convert to Path object if string
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use advanced entity extractor by default
        self.extractor = extractor or AdvancedEntityExtractor()
        
        # Use default relationship mapper if not provided
        self.mapper = mapper or RelationshipMapper()
        
        # Initialize logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'knowledge_graph_builder.log'),
                logging.StreamHandler()
            ]
        )
        
        logger.info("Knowledge Graph Builder initialized")
        
        self.schema = KnowledgeGraphSchema()
        
    async def build_graph(self, papers: List[Dict]) -> Optional[nx.Graph]:
        """Build knowledge graph from research papers."""
        try:
            logger.info("Starting knowledge graph construction...")
            
            # Extract entities from papers
            entities = await self._extract_entities([paper.get('abstract', '') + ' ' + paper.get('content', '') for paper in papers])
            logger.info(f"Extracted {len(entities)} entities")
            
            # Add entities to schema
            for entity in entities:
                self.schema.add_entity(entity)
                
            # Map relationships between entities
            texts = [paper.get('abstract', '') + ' ' + paper.get('content', '') 
                    for paper in papers]
            relationships = await self._map_relationships(entities, texts)
            logger.info(f"Mapped {len(relationships)} relationships")
            
            # Add relationships to schema
            for relationship in relationships:
                self.schema.add_relationship(relationship)
                
            # Validate graph
            if not self.schema.validate_graph():
                logger.error("Graph validation failed")
                return None
                
            # Create NetworkX graph
            graph = nx.Graph()
            
            # Add nodes
            for entity in entities:
                graph.add_node(entity.id, **entity.dict())
            
            # Add edges
            for relationship in relationships:
                graph.add_edge(relationship.source_id, relationship.target_id, 
                               type=relationship.type, 
                               confidence=relationship.confidence)
            
            # Save graph
            await self._save_graph(graph)
            
            return graph
        
        except Exception as e:
            logger.error(f"Error building knowledge graph: {str(e)}")
            return None
        
    async def _extract_entities(
        self, 
        texts: Union[List[str], Generator[str, None, None]], 
        additional_context: Optional[Dict[str, Any]] = None
    ) -> List[Entity]:
        """
        Extract entities from multiple texts using advanced techniques.
        
        Args:
            texts: List or generator of text sources
            additional_context: Optional context for entity extraction
        
        Returns:
            List of extracted entities
        """
        # Convert generator to list if needed
        if not isinstance(texts, list):
            texts = list(texts)
        
        # Prepare for parallel entity extraction
        entities = []
        
        # Use asyncio for potentially faster processing
        async def process_text(text):
            try:
                # Use advanced extractor with additional context
                text_entities = self.extractor.extract_entities(
                    text, 
                    additional_context
                )
                return text_entities
            except Exception as e:
                logger.warning(f"Error extracting entities from text: {e}")
                return []
        
        # Parallel entity extraction
        entity_tasks = [process_text(text) for text in texts]
        entity_results = await asyncio.gather(*entity_tasks)
        
        # Flatten results
        for result in entity_results:
            entities.extend(result)
        
        # Remove duplicates while preserving order
        unique_entities = []
        seen = set()
        for entity in entities:
            if entity.name.lower() not in seen:
                unique_entities.append(entity)
                seen.add(entity.name.lower())
        
        logger.info(f"Extracted {len(unique_entities)} unique entities")
        return unique_entities
    
    async def _map_relationships(self, entities: List[Entity], texts: List[str]) -> List[Relationship]:
        """Map relationships between entities."""
        # Ensure texts is a list of strings
        processed_texts = []
        for text in texts:
            # Handle different input types
            if isinstance(text, dict):
                # If it's a paper dictionary, combine abstract and content
                text_content = text.get('abstract', '') + ' ' + text.get('content', '')
                processed_texts.append(text_content)
            elif isinstance(text, str):
                processed_texts.append(text)
            else:
                # Convert other types to string
                processed_texts.append(str(text))
        
        return await self.mapper.map_relationships(entities, processed_texts)
    
    async def _save_graph(self, graph: nx.Graph):
        """Save knowledge graph to files."""
        # Initialize graph version control
        version_control = GraphVersionControl()
        
        # Save graph version with metadata
        version_id = version_control.save_graph_version(
            graph, 
            metadata={
                'extraction_method': 'paper_analysis',
                'entity_count': graph.number_of_nodes(),
                'relationship_count': graph.number_of_edges()
            }
        )
        
        # Prune old versions to manage storage
        version_control.prune_versions(max_versions=10)
        
        # Traditional graph saving methods
        try:
            # Save as NetworkX graph
            nx.write_gexf(graph, os.path.join(self.output_dir, 'knowledge_graph.gexf'))
            
            # Save as JSON for easier parsing
            graph_data = {
                'nodes': [
                    {**node[1], 'id': node[0]} 
                    for node in graph.nodes(data=True)
                ],
                'edges': [
                    {**edge[2], 'source': edge[0], 'target': edge[1]} 
                    for edge in graph.edges(data=True)
                ]
            }
            
            with open(os.path.join(self.output_dir, 'knowledge_graph.json'), 'w') as f:
                json.dump(graph_data, f, indent=2)
            
            # Optional: Visualization
            plt.figure(figsize=(20, 20))
            pos = nx.spring_layout(graph, k=0.5, iterations=50)
            nx.draw(
                graph, 
                pos, 
                with_labels=False, 
                node_color='lightblue', 
                node_size=50, 
                edge_color='gray', 
                alpha=0.6
            )
            plt.title('Knowledge Graph Visualization')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'knowledge_graph_viz.png'), dpi=300)
            plt.close()
            
            logger.info(f"Knowledge graph saved. Version ID: {version_id}")
        except Exception as e:
            logger.error(f"Error saving knowledge graph: {str(e)}")
