import os
import logging
from datetime import datetime
from typing import List, Type, Dict, Any
from src.scrapers.base_scraper import BaseScraper
from src.scrapers.harvard_dataverse_scraper import HarvardDataverseScraper
from src.scrapers.nasa_scraper import NASAScraper
from src.scrapers.nist_scraper import NISTScraper
from src.scrapers.physics_forums_scraper import PhysicsForumsScraper
from src.scrapers.researchgate_scraper import ResearchGateScraper
from src.scrapers.sciencedirect_scraper import ScienceDirectScraper
from src.scrapers.antimatter_scraper import AntimatterScraper
from src.scrapers.ieee_scraper import IEEEScraper
from src.scrapers.materials_scraper import MaterialsScraper
from src.scrapers.patent_scraper import PatentScraper
from src.utils.retry_utils import validate_sites_before_scraping, ScraperRetryPolicy
from tqdm import tqdm
import time
from dotenv import load_dotenv
import random
from huggingface_hub import HfApi, create_repo
import pandas as pd
import json
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraping.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class DatasetManager:
    """Manages dataset organization and upload to Hugging Face."""
    
    REPO_ID = "GotThatData/warp-speed"
    DATASET_STRUCTURE = {
        "quantum_physics": {
            "description": "Quantum physics research and experiments",
            "subcategories": [
                "quantum_field_theory",
                "quantum_entanglement",
                "quantum_tunneling",
                "quantum_computing",
                "quantum_teleportation",
                "quantum_gravity",
                "quantum_vacuum",
                "quantum_chromodynamics"
            ]
        },
        "spacetime": {
            "description": "Research on spacetime, gravity, and related phenomena",
            "subcategories": [
                "gravitational_waves",
                "black_holes",
                "wormholes",
                "spacetime_curvature",
                "alcubierre_metrics",
                "causal_structure",
                "exotic_matter",
                "negative_energy"
            ]
        },
        "experimental_physics": {
            "description": "Experimental physics data and results",
            "subcategories": [
                "particle_physics",
                "high_energy_physics",
                "quantum_materials",
                "plasma_physics",
                "superconductivity",
                "antimatter",
                "fusion_research",
                "field_generation"
            ]
        },
        "propulsion_theory": {
            "description": "Theoretical and experimental propulsion research",
            "subcategories": [
                "warp_field_mechanics",
                "space_folding",
                "quantum_propulsion",
                "field_manipulation",
                "inertial_dampening",
                "subspace_dynamics",
                "tachyonic_particles",
                "zero_point_energy"
            ]
        },
        "materials_science": {
            "description": "Advanced materials research and development",
            "subcategories": [
                "metamaterials",
                "exotic_matter_states",
                "quantum_dots",
                "topological_materials",
                "smart_materials",
                "energy_crystals",
                "field_containment",
                "high_temp_superconductors"
            ]
        },
        "energy_systems": {
            "description": "Energy generation and manipulation research",
            "subcategories": [
                "matter_antimatter",
                "zero_point_extraction",
                "quantum_energy_states",
                "field_coupling",
                "power_generation",
                "energy_conversion",
                "containment_systems",
                "stability_control"
            ]
        },
        "theoretical_physics": {
            "description": "Theoretical frameworks and mathematical models",
            "subcategories": [
                "unified_field_theory",
                "m_theory",
                "string_theory",
                "loop_quantum_gravity",
                "supersymmetry",
                "causal_dynamics",
                "quantum_cosmology",
                "dimensional_theory"
            ]
        },
        "computational_physics": {
            "description": "Computational models and simulations",
            "subcategories": [
                "field_simulations",
                "quantum_calculations",
                "spacetime_modeling",
                "warp_field_analysis",
                "stability_predictions",
                "trajectory_optimization",
                "energy_efficiency",
                "safety_protocols"
            ]
        }
    }
    
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.api = HfApi()
        self.metadata = {
            "description": "A comprehensive dataset for warp drive research",
            "license": "MIT",
            "citations": [],
            "last_updated": datetime.now().isoformat(),
            "version": "1.0.0",
            "maintainers": ["GotThatData"],
            "keywords": [
                "warp drive",
                "spacetime manipulation",
                "quantum physics",
                "propulsion",
                "theoretical physics",
                "experimental physics"
            ]
        }
    
    def setup_directory_structure(self):
        """Create the directory structure for the dataset."""
        # Create main categories
        for category, info in self.DATASET_STRUCTURE.items():
            category_dir = os.path.join(self.base_dir, category)
            os.makedirs(category_dir, exist_ok=True)
            
            # Create subcategories
            for subcategory in info["subcategories"]:
                os.makedirs(os.path.join(category_dir, subcategory), exist_ok=True)
            
            # Create category metadata
            with open(os.path.join(category_dir, "metadata.json"), "w") as f:
                json.dump({
                    "description": info["description"],
                    "subcategories": info["subcategories"]
                }, f, indent=2)
    
    def categorize_data(self, scraper_name: str, data_file: str) -> str:
        """Determine the appropriate category for the data."""
        df = pd.read_csv(data_file)
        
        # Enhanced mapping of scrapers to categories
        categorization = {
            "HarvardDataverseScraper": ("spacetime", "black_holes"),
            "LIGOScraper": ("spacetime", "gravitational_waves"),
            "EHTScraper": ("spacetime", "black_holes"),
            "NASAScraper": ("spacetime", "propulsion"),
            "NISTScraper": ("experimental_physics", "quantum_materials")
        }
        
        category, subcategory = categorization.get(
            scraper_name, 
            ("experimental_physics", "particle_physics")
        )
        
        # Create target directory if it doesn't exist
        target_dir = os.path.join(self.base_dir, category, subcategory)
        os.makedirs(target_dir, exist_ok=True)
        
        # Copy file to appropriate directory
        target_file = os.path.join(target_dir, os.path.basename(data_file))
        shutil.copy2(data_file, target_file)
        
        # Add category-specific metadata
        metadata_file = os.path.join(target_dir, "source_metadata.json")
        source_metadata = {
            "source": scraper_name,
            "timestamp": datetime.now().isoformat(),
            "record_count": len(df),
            "columns": list(df.columns),
            "data_types": df.dtypes.astype(str).to_dict(),
            "category": category,
            "subcategory": subcategory
        }
        
        with open(metadata_file, "w") as f:
            json.dump(source_metadata, f, indent=2)
        
        return target_file
    
    def update_metadata(self, scraper_results: Dict[str, Any]):
        """Update dataset metadata with new information."""
        metadata_file = os.path.join(self.base_dir, "dataset_info.json")
        
        self.metadata["last_updated"] = datetime.now().isoformat()
        self.metadata["total_records"] = sum(
            len(pd.read_csv(file)) for file in scraper_results.values() if file
        )
        self.metadata["data_sources"] = list(scraper_results.keys())
        
        with open(metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)
    
    def upload_to_huggingface(self) -> bool:
        """Upload the dataset to Hugging Face."""
        try:
            # Ensure repository exists
            create_repo(
                repo_id=self.REPO_ID,
                repo_type="dataset",
                exist_ok=True
            )
            
            # Upload all files
            self.api.upload_folder(
                folder_path=self.base_dir,
                repo_id=self.REPO_ID,
                repo_type="dataset"
            )
            
            logger.info(f"Successfully uploaded dataset to {self.REPO_ID}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload dataset: {str(e)}")
            return False

class ScraperOrchestrator:
    """Orchestrates the sequential execution of scrapers with validation and progress tracking."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.scraper_classes: List[Type[BaseScraper]] = [
            NASAScraper,
            NISTScraper,
            PhysicsForumsScraper,
            ResearchGateScraper,
            ScienceDirectScraper,
            AntimatterScraper,
            IEEEScraper,
            MaterialsScraper,
            PatentScraper
        ]
        self.retry_policy = ScraperRetryPolicy()
        self.failed_scrapers: List[Type[BaseScraper]] = []
    
    def validate_and_prepare(self) -> bool:
        """Validate sites and prepare for scraping."""
        logger.info("Validating all sites before starting the scraping process...")
        return validate_sites_before_scraping()
    
    def run_scraper(self, scraper: BaseScraper) -> bool:
        """Run a single scraper with progress tracking and retry mechanism."""
        scraper_name = scraper.__class__.__name__
        attempt = self.retry_policy.get_attempt_count(scraper_name) + 1
        
        try:
            if attempt > 1:
                logger.info(f"Retry attempt {attempt} for {scraper_name}")
            else:
                logger.info(f"Starting {scraper_name}")
            
            # Fetch data with progress tracking
            with tqdm(total=100, desc=f"Fetching {scraper_name} data (Attempt {attempt})") as pbar:
                pbar.update(10)  # Initial connection
                data = scraper.fetch_data()
                pbar.update(40)  # Data fetching complete
                
                # Preprocess data
                df = scraper.preprocess_data(data)
                pbar.update(30)  # Preprocessing complete
                
                # Validate data
                if df.empty:
                    raise ValueError("No data collected")
                
                # Save to CSV
                output_file = os.path.join(
                    self.output_dir,
                    f"{scraper_name.lower()}_data.csv"
                )
                df.to_csv(output_file, index=False)
                pbar.update(20)  # Saving complete
            
            logger.info(f"Successfully completed {scraper_name}")
            logger.info(f"Saved data to {output_file}")
            logger.info(f"Records collected: {len(df)}")
            
            # Record successful attempt
            self.retry_policy.record_attempt(scraper_name, True)
            
            # Add a small delay between scrapers
            time.sleep(2)
            return True
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error in {scraper_name} (Attempt {attempt}): {error_msg}")
            
            # Record failed attempt
            self.retry_policy.record_attempt(scraper_name, False, error_msg)
            
            # Check if we should retry
            if self.retry_policy.should_retry(scraper_name):
                delay = self.retry_policy.get_delay(scraper_name)
                logger.info(f"Will retry {scraper_name} in {delay:.1f} seconds...")
                time.sleep(delay)
                return False
            else:
                logger.error(f"All retry attempts exhausted for {scraper_name}")
                return False
    
    def retry_failed_scrapers(self) -> None:
        """Retry any failed scrapers."""
        if not self.failed_scrapers:
            return
        
        logger.info(f"Attempting to retry {len(self.failed_scrapers)} failed scrapers...")
        still_failed = []
        
        for scraper_class in self.failed_scrapers:
            try:
                scraper = scraper_class(self.output_dir)
                if not self.run_scraper(scraper):
                    still_failed.append(scraper_class)
            except Exception as e:
                logger.error(f"Failed to initialize {scraper_class.__name__} for retry: {str(e)}")
                still_failed.append(scraper_class)
        
        self.failed_scrapers = still_failed
    
    def run_all_scrapers(self) -> Dict[str, str]:
        """Run all scrapers sequentially with retry mechanism."""
        total_scrapers = len(self.scraper_classes)
        successful_scrapers = 0
        self.failed_scrapers = []
        scraper_results = {}
        
        logger.info(f"Starting scraping process with {total_scrapers} scrapers")
        
        # First pass: try all scrapers
        for scraper_class in self.scraper_classes:
            try:
                scraper = scraper_class(self.output_dir)
                if self.run_scraper(scraper):
                    successful_scrapers += 1
                    scraper_results[scraper_class.__name__] = os.path.join(
                        self.output_dir,
                        f"{scraper_class.__name__.lower()}_data.csv"
                    )
                else:
                    self.failed_scrapers.append(scraper_class)
                    scraper_results[scraper_class.__name__] = None
            except Exception as e:
                logger.error(f"Failed to initialize {scraper_class.__name__}: {str(e)}")
                self.failed_scrapers.append(scraper_class)
                scraper_results[scraper_class.__name__] = None
        
        # Retry failed scrapers
        while self.failed_scrapers and self.retry_policy.should_retry(self.failed_scrapers[0].__name__):
            self.retry_failed_scrapers()
            successful_scrapers = total_scrapers - len(self.failed_scrapers)
            
            # Update results for any newly successful scrapers
            for scraper_class in self.scraper_classes:
                if scraper_class not in self.failed_scrapers:
                    scraper_results[scraper_class.__name__] = os.path.join(
                        self.output_dir,
                        f"{scraper_class.__name__.lower()}_data.csv"
                    )
        
        # Log final statistics
        logger.info("Scraping process completed")
        logger.info(f"Successful scrapers: {successful_scrapers}/{total_scrapers}")
        
        if self.failed_scrapers:
            logger.warning(
                f"Some scrapers failed after all retry attempts. "
                f"Failed scrapers: {[s.__name__ for s in self.failed_scrapers]}"
            )
        
        # Log retry statistics
        for scraper_class in self.scraper_classes:
            scraper_name = scraper_class.__name__
            attempts = self.retry_policy.get_attempt_count(scraper_name)
            if attempts > 1:
                logger.info(f"{scraper_name} required {attempts} attempts")
        
        return scraper_results

def main():
    """Main function to run the scraping process."""
    try:
        # Load environment variables
        load_dotenv()
        
        # Create output directory with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join('data', f'scrape_{timestamp}')
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize dataset manager
        dataset_manager = DatasetManager(output_dir)
        dataset_manager.setup_directory_structure()
        
        # Initialize orchestrator
        orchestrator = ScraperOrchestrator(output_dir)
        
        # Validate sites
        if not orchestrator.validate_and_prepare():
            logger.error("Site validation failed. Aborting scraping process.")
            return
        
        # Run scrapers sequentially with retry mechanism
        scraper_results = orchestrator.run_all_scrapers()
        
        # Organize and categorize the data
        for scraper_name, data_file in scraper_results.items():
            if data_file and os.path.exists(data_file):
                dataset_manager.categorize_data(scraper_name, data_file)
        
        # Update dataset metadata
        dataset_manager.update_metadata(scraper_results)
        
        # Upload to Hugging Face if token exists
        if os.getenv('HUGGINGFACE_TOKEN'):
            if dataset_manager.upload_to_huggingface():
                logger.info("Dataset successfully uploaded to Hugging Face")
            else:
                logger.error("Failed to upload dataset to Hugging Face")
        else:
            logger.warning("HUGGINGFACE_TOKEN not found. Skipping upload.")
        
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        raise

if __name__ == "__main__":
    main()
