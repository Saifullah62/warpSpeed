import os
import json
from pathlib import Path

# Create data directory structure
project_root = Path(__file__).parent.parent
data_dir = project_root / "data"
papers_dir = data_dir / "papers"

data_dir.mkdir(exist_ok=True)
papers_dir.mkdir(exist_ok=True)

# Create sample papers metadata
sample_papers = [
    {
        "id": "paper1",
        "title": "Quantum Mechanics and Warp Drive Theory",
        "filename": "quantum_mechanics_warp_drive.txt",
        "authors": ["John Smith", "Jane Doe"],
        "year": 2023,
        "abstract": "This paper explores the theoretical foundations of warp drive technology through the lens of quantum mechanics..."
    },
    {
        "id": "paper2", 
        "title": "Subspace Field Dynamics",
        "filename": "subspace_field_dynamics.txt",
        "authors": ["Alice Johnson", "Bob Wilson"],
        "year": 2023,
        "abstract": "An analysis of subspace field dynamics and their implications for faster-than-light travel..."
    }
]

# Save metadata
metadata_file = papers_dir / "metadata.json"
with open(metadata_file, 'w', encoding='utf-8') as f:
    json.dump(sample_papers, f, indent=2)

# Create sample paper content
paper1_content = """
Title: Quantum Mechanics and Warp Drive Theory

Abstract:
This paper explores the theoretical foundations of warp drive technology through the lens of quantum mechanics. We examine the relationship between quantum field theory and space-time curvature, with particular focus on the Alcubierre metric and its quantum mechanical implications.

1. Introduction
The development of faster-than-light propulsion systems remains one of the most significant challenges in theoretical physics. The concept of warp drive, first proposed by Miguel Alcubierre, suggests the possibility of achieving superluminal travel by manipulating space-time geometry.

2. Quantum Mechanical Framework
2.1 Quantum Field Theory
The quantum field theoretical approach to warp drive mechanics involves understanding the interaction between quantum fields and curved space-time. Key concepts include:
- Vacuum energy density
- Quantum fluctuations
- Field operator algebra

2.2 Space-time Curvature
The manipulation of space-time geometry requires:
- Negative energy density
- Stable warp bubble formation
- Quantum tunneling effects

3. Theoretical Results
Our analysis shows that quantum mechanical effects play a crucial role in:
- Warp bubble stability
- Energy requirements
- Field containment

4. Conclusion
The integration of quantum mechanics with warp drive theory provides new insights into potential propulsion mechanisms.
"""

paper2_content = """
Title: Subspace Field Dynamics

Abstract:
An analysis of subspace field dynamics and their implications for faster-than-light travel. This paper presents new mathematical models for understanding subspace field behavior and its potential applications in advanced propulsion systems.

1. Introduction
Subspace field theory represents a promising avenue for achieving faster-than-light travel. This paper examines the fundamental properties of subspace fields and their interaction with normal space-time.

2. Theoretical Framework
2.1 Field Equations
The basic equations governing subspace field dynamics:
- Subspace metric tensor
- Field strength tensors
- Energy-momentum relationships

2.2 Interaction Mechanisms
Key interaction mechanisms include:
- Field coupling
- Energy transfer
- Dimensional transitions

3. Applications
Potential applications in propulsion technology:
- Warp field generation
- Subspace bubble formation
- Field stability control

4. Conclusion
Subspace field dynamics offer a promising theoretical framework for advanced propulsion systems.
"""

# Save paper contents
with open(papers_dir / "quantum_mechanics_warp_drive.txt", 'w', encoding='utf-8') as f:
    f.write(paper1_content)
    
with open(papers_dir / "subspace_field_dynamics.txt", 'w', encoding='utf-8') as f:
    f.write(paper2_content)

print("Data directory structure and sample papers created successfully!")
