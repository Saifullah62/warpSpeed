import os
import json
import hashlib
import networkx as nx
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class GraphVersionControl:
    """
    Manages versioning and tracking of knowledge graph changes.
    
    Provides capabilities to:
    - Save graph snapshots
    - Track graph modifications
    - Retrieve historical graph states
    """
    
    def __init__(self, base_path: str = 'knowledge_graph_versions'):
        """
        Initialize graph version control system.
        
        Args:
            base_path: Directory to store graph version history
        """
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
    
    def _generate_version_id(self, graph: nx.Graph) -> str:
        """
        Generate a unique version ID based on graph structure and content.
        
        Args:
            graph: NetworkX graph to generate ID for
        
        Returns:
            Unique version identifier
        """
        # Create a deterministic representation of the graph
        graph_repr = {
            'nodes': sorted(list(graph.nodes(data=True))),
            'edges': sorted(list(graph.edges(data=True)))
        }
        
        # Generate hash based on graph representation
        graph_hash = hashlib.md5(
            json.dumps(graph_repr, sort_keys=True).encode()
        ).hexdigest()
        
        # Incorporate timestamp for additional uniqueness
        timestamp = datetime.now().isoformat()
        return f"{graph_hash}_{timestamp}"
    
    def save_graph_version(
        self, 
        graph: nx.Graph, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save a version of the knowledge graph.
        
        Args:
            graph: NetworkX graph to save
            metadata: Additional metadata about the graph version
        
        Returns:
            Version ID of the saved graph
        """
        # Generate unique version ID
        version_id = self._generate_version_id(graph)
        
        # Prepare version metadata
        version_metadata = {
            'version_id': version_id,
            'timestamp': datetime.now().isoformat(),
            'node_count': graph.number_of_nodes(),
            'edge_count': graph.number_of_edges(),
            'additional_metadata': metadata or {}
        }
        
        # Save graph structure
        graph_path = os.path.join(self.base_path, f'{version_id}_graph.json')
        with open(graph_path, 'w') as f:
            json.dump({
                'nodes': list(graph.nodes(data=True)),
                'edges': list(graph.edges(data=True))
            }, f, indent=2)
        
        # Save metadata
        metadata_path = os.path.join(self.base_path, f'{version_id}_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(version_metadata, f, indent=2)
        
        logger.info(f"Saved graph version: {version_id}")
        return version_id
    
    def list_versions(self) -> List[Dict[str, Any]]:
        """
        List all saved graph versions.
        
        Returns:
            List of version metadata dictionaries
        """
        versions = []
        for filename in os.listdir(self.base_path):
            if filename.endswith('_metadata.json'):
                try:
                    with open(os.path.join(self.base_path, filename), 'r') as f:
                        versions.append(json.load(f))
                except Exception as e:
                    logger.warning(f"Error reading version metadata {filename}: {e}")
        
        # Sort versions by timestamp
        return sorted(
            versions, 
            key=lambda x: x.get('timestamp', ''), 
            reverse=True
        )
    
    def load_graph_version(self, version_id: str) -> Optional[nx.Graph]:
        """
        Load a specific graph version.
        
        Args:
            version_id: Unique identifier of the graph version
        
        Returns:
            Reconstructed NetworkX graph or None if not found
        """
        graph_path = os.path.join(self.base_path, f'{version_id}_graph.json')
        
        if not os.path.exists(graph_path):
            logger.warning(f"Graph version {version_id} not found")
            return None
        
        try:
            with open(graph_path, 'r') as f:
                graph_data = json.load(f)
            
            # Reconstruct graph
            graph = nx.Graph()
            graph.add_nodes_from(graph_data['nodes'])
            graph.add_edges_from(graph_data['edges'])
            
            return graph
        except Exception as e:
            logger.error(f"Error loading graph version {version_id}: {e}")
            return None
    
    def compare_versions(
        self, 
        version1_id: str, 
        version2_id: str
    ) -> Dict[str, Any]:
        """
        Compare two graph versions.
        
        Args:
            version1_id: First version identifier
            version2_id: Second version identifier
        
        Returns:
            Comparison report
        """
        graph1 = self.load_graph_version(version1_id)
        graph2 = self.load_graph_version(version2_id)
        
        if not graph1 or not graph2:
            raise ValueError("One or both versions not found")
        
        # Compare nodes
        nodes1 = set(graph1.nodes())
        nodes2 = set(graph2.nodes())
        
        # Compare edges
        edges1 = set(graph1.edges())
        edges2 = set(graph2.edges())
        
        return {
            'nodes_added': list(nodes2 - nodes1),
            'nodes_removed': list(nodes1 - nodes2),
            'edges_added': list(edges2 - edges1),
            'edges_removed': list(edges1 - edges2)
        }
    
    def prune_versions(self, max_versions: int = 10):
        """
        Remove older graph versions to manage storage.
        
        Args:
            max_versions: Maximum number of versions to keep
        """
        versions = self.list_versions()
        
        if len(versions) <= max_versions:
            return
        
        # Sort versions and remove older ones
        versions_to_remove = versions[max_versions:]
        
        for version in versions_to_remove:
            version_id = version['version_id']
            
            # Remove graph and metadata files
            graph_file = os.path.join(self.base_path, f'{version_id}_graph.json')
            metadata_file = os.path.join(self.base_path, f'{version_id}_metadata.json')
            
            try:
                if os.path.exists(graph_file):
                    os.remove(graph_file)
                if os.path.exists(metadata_file):
                    os.remove(metadata_file)
                
                logger.info(f"Removed old graph version: {version_id}")
            except Exception as e:
                logger.warning(f"Error removing version {version_id}: {e}")
