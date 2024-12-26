from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
import networkx as nx

class EntityType(str, Enum):
    CONCEPT = "concept"
    TECHNOLOGY = "technology"
    THEORY = "theory"
    EXPERIMENT = "experiment"
    MATERIAL = "material"
    PHENOMENON = "phenomenon"
    EQUATION = "equation"

class RelationType(str, Enum):
    DEPENDS_ON = "depends_on"
    PROVES = "proves"
    IMPLEMENTS = "implements"
    USES = "uses"
    RELATES_TO = "relates_to"
    CONTRADICTS = "contradicts"
    EXTENDS = "extends"
    MEASURES = "measures"
    PREDICTS = "predicts"

class Property(BaseModel):
    name: str
    value: str
    unit: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    source: Optional[str] = None

class Entity(BaseModel):
    id: str
    type: EntityType
    name: str
    description: str
    properties: List[Property] = []
    references: List[str] = []
    confidence: float = Field(ge=0.0, le=1.0)
    metadata: Dict[str, str] = {}

class Relationship(BaseModel):
    id: str
    type: RelationType
    source_id: str
    target_id: str
    properties: List[Property] = []
    confidence: float = Field(ge=0.0, le=1.0)
    metadata: Dict[str, str] = {}

class KnowledgeGraphSchema:
    """Schema definition for the Warp Drive Knowledge Graph."""
    
    def __init__(self):
        """Initialize the knowledge graph schema."""
        self.graph = nx.Graph()
        self.entities: Dict[str, Entity] = {}
        self.relationships: Dict[str, Relationship] = {}
        
    def add_entity(self, entity: Entity) -> None:
        """Add an entity to the graph."""
        if entity.id not in self.entities:
            self.entities[entity.id] = entity
        else:
            # Update existing entity with new information
            existing_entity = self.entities[entity.id]
            for key, value in entity.dict().items():
                if value and not getattr(existing_entity, key):
                    setattr(existing_entity, key, value)
            
    def add_relationship(self, relationship: Relationship) -> None:
        """Add a relationship to the knowledge graph."""
        if relationship.id in self.relationships:
            raise ValueError(f"Relationship with ID {relationship.id} already exists")
            
        # Validate that source and target entities exist
        if relationship.source_id not in self.entities:
            raise ValueError(f"Source entity {relationship.source_id} does not exist")
        if relationship.target_id not in self.entities:
            raise ValueError(f"Target entity {relationship.target_id} does not exist")
            
        self.relationships[relationship.id] = relationship
        
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by its ID."""
        return self.entities.get(entity_id)
        
    def get_relationships(self, entity_id: str) -> List[Relationship]:
        """Get all relationships involving an entity."""
        return [
            rel for rel in self.relationships.values()
            if rel.source_id == entity_id or rel.target_id == entity_id
        ]
        
    def get_related_entities(self, entity_id: str) -> List[Entity]:
        """Get all entities related to a given entity."""
        related_ids = set()
        for rel in self.get_relationships(entity_id):
            if rel.source_id == entity_id:
                related_ids.add(rel.target_id)
            else:
                related_ids.add(rel.source_id)
        
        return [self.entities[rid] for rid in related_ids]
        
    def validate_graph(self) -> bool:
        """Validate the consistency of the knowledge graph."""
        try:
            # Check for orphaned relationships
            for rel in self.relationships.values():
                if rel.source_id not in self.entities:
                    raise ValueError(f"Relationship {rel.id} has invalid source entity")
                if rel.target_id not in self.entities:
                    raise ValueError(f"Relationship {rel.id} has invalid target entity")
                    
            # Check for circular dependencies
            self._check_circular_dependencies()
            
            return True
            
        except Exception as e:
            print(f"Graph validation failed: {str(e)}")
            return False
            
    def _check_circular_dependencies(self) -> None:
        """Check for circular dependencies in the graph."""
        def dfs(node: str, visited: set, path: set) -> None:
            if node in path:
                raise ValueError(f"Circular dependency detected involving {node}")
                
            path.add(node)
            visited.add(node)
            
            for rel in self.get_relationships(node):
                next_node = rel.target_id if rel.source_id == node else rel.source_id
                if next_node not in visited:
                    dfs(next_node, visited, path.copy())
                    
        visited = set()
        for entity_id in self.entities:
            if entity_id not in visited:
                dfs(entity_id, visited, set())
