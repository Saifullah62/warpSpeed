import os

def create_directory_structure():
    base_dir = "data"
    
    # Main categories
    structure = {
        "quantum_physics": [
            "quantum_field_theory",
            "quantum_entanglement",
            "quantum_tunneling",
            "quantum_computing",
            "quantum_teleportation",
            "quantum_gravity",
            "quantum_vacuum",
            "quantum_chromodynamics"
        ],
        "spacetime": [
            "general_relativity",
            "special_relativity",
            "wormholes",
            "metric_engineering",
            "causal_structure",
            "exotic_matter",
            "field_propulsion"
        ],
        "metadata": []
    }
    
    # Create base directory
    os.makedirs(base_dir, exist_ok=True)
    print(f"Created {base_dir}/")
    
    # Create subdirectories
    for category, subcategories in structure.items():
        category_path = os.path.join(base_dir, category)
        os.makedirs(category_path, exist_ok=True)
        print(f"Created {category_path}/")
        
        for subcategory in subcategories:
            subcategory_path = os.path.join(category_path, subcategory)
            os.makedirs(subcategory_path, exist_ok=True)
            print(f"Created {subcategory_path}/")
    
    # Create metadata files
    metadata_files = ["papers.json", "experiments.json", "citations.json"]
    for file in metadata_files:
        file_path = os.path.join(base_dir, "metadata", file)
        with open(file_path, "w") as f:
            f.write("{}")
        print(f"Created {file_path}")

if __name__ == "__main__":
    create_directory_structure()
