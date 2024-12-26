"""Script to run the NASA NTRS scraper with specific warp drive related keywords."""

import os
import sys
import logging
from pathlib import Path
from itertools import product

# Add src directory to Python path
src_dir = str(Path(__file__).parent.parent / "src")
sys.path.append(src_dir)

from data_collection.nasa_scraper import NASAScraper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nasa_scraper.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def get_search_combinations():
    """Generate comprehensive search combinations for warp drive research."""
    
    # Core concepts
    core_terms = [
        "warp drive",
        "alcubierre drive",
        "faster than light",
        "ftl propulsion",
        "superluminal"
    ]
    
    # Physics concepts
    physics_terms = [
        "space time manipulation",
        "negative energy density",
        "exotic matter",
        "quantum vacuum",
        "spacetime metric",
        "gravitational field",
        "quantum field theory",
        "general relativity",
        "quantum gravity",
        "quantum tunneling",
        "casimir effect"
    ]
    
    # Propulsion concepts
    propulsion_terms = [
        "antimatter propulsion",
        "quantum propulsion",
        "advanced propulsion",
        "breakthrough propulsion",
        "field propulsion",
        "space propulsion",
        "plasma propulsion"
    ]
    
    # Energy concepts
    energy_terms = [
        "zero point energy",
        "vacuum energy",
        "dark energy",
        "negative energy",
        "antimatter",
        "energy density"
    ]
    
    # Materials
    materials_terms = [
        "exotic materials",
        "metamaterials",
        "negative mass",
        "negative matter",
        "quantum materials"
    ]
    
    # Combine terms in meaningful ways
    combinations = []
    
    # Add all individual terms
    combinations.extend(core_terms + physics_terms + propulsion_terms + energy_terms + materials_terms)
    
    # Combine core terms with physics concepts
    combinations.extend([f"{core} {physics}" 
                       for core, physics in product(core_terms, physics_terms)])
    
    # Combine core terms with propulsion
    combinations.extend([f"{core} {prop}" 
                       for core, prop in product(core_terms, propulsion_terms)])
    
    # Combine propulsion with energy
    combinations.extend([f"{prop} {energy}" 
                       for prop, energy in product(propulsion_terms, energy_terms)])
    
    # Combine propulsion with materials
    combinations.extend([f"{prop} {material}" 
                       for prop, material in product(propulsion_terms, materials_terms)])
    
    # Add specific multi-term combinations
    specific_combinations = [
        "warp drive negative energy",
        "alcubierre drive exotic matter",
        "ftl quantum tunneling",
        "warp drive metamaterials",
        "faster than light quantum gravity",
        "warp field propulsion",
        "spacetime manipulation propulsion",
        "quantum vacuum propulsion",
        "negative mass propulsion",
        "antimatter warp drive",
        "quantum gravity propulsion",
        "casimir effect propulsion",
        "zero point energy propulsion",
        "exotic matter warp drive",
        "quantum field propulsion",
        "space time warping",
        "metric engineering propulsion",
        "negative energy warp field",
        "quantum vacuum fluctuation",
        "relativistic space drive"
    ]
    combinations.extend(specific_combinations)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_combinations = []
    for item in combinations:
        if item not in seen:
            seen.add(item)
            unique_combinations.append(item)
    
    return unique_combinations

def main():
    try:
        # Initialize scraper
        output_dir = os.path.join("data", "nasa")
        scraper = NASAScraper(output_dir=output_dir)
        
        # Get search combinations
        search_terms = get_search_combinations()
        logger.info(f"Generated {len(search_terms)} unique search combinations")
        
        # Process each combination in batches
        batch_size = 5
        for i in range(0, len(search_terms), batch_size):
            batch = search_terms[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1} of {(len(search_terms) + batch_size - 1)//batch_size}")
            
            # Scrape all categories with current batch of keywords
            scraper.scrape_all_categories(
                keywords=batch,
                max_results=100,  # Increased from previous runs
                start_year=1950  # Include historical papers
            )
            
            logger.info(f"Completed batch {i//batch_size + 1}")
        
        logger.info("NASA scraping completed successfully")
        
    except Exception as e:
        logger.error(f"Error during NASA scraping: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
