#!/usr/bin/env python3
"""
Multi-Modal Entity Extraction Demonstration Script

This script showcases the advanced multi-modal entity extraction capabilities
of our knowledge graph system, demonstrating entity recognition across
different modalities and sources.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Ensure project root is in Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import logging configuration
from src.knowledge_graph.logging_config import setup_logging, get_logger

# Configure logging
setup_logging(log_level='DEBUG')
logger = get_logger(__name__)

# Import multi-modal entity extractor
from src.knowledge_graph.multi_modal_entity_extractor import (
    MultiModalEntityExtractor, 
    EntityExtractionStrategy
)

# Sample scientific texts and resources
SCIENTIFIC_TEXTS = [
    "Quantum entanglement is a fundamental principle of quantum mechanics where two particles become correlated in such a way that the quantum state of each particle cannot be described independently.",
    
    "The development of warp drive technology could revolutionize space exploration by enabling faster-than-light travel through the manipulation of spacetime geometry.",
    
    "Gravitational wave detection using advanced laser interferometry has opened up new frontiers in observational astrophysics, allowing scientists to study cosmic events like black hole mergers.",
    
    "Experimental quantum computing platforms are rapidly advancing, with superconducting qubits and topological quantum computation emerging as promising approaches to scalable quantum information processing."
]

def create_demo_images():
    """
    Create sample images for visual entity extraction demo.
    
    Returns:
        List of paths to created demo images
    """
    from PIL import Image, ImageDraw, ImageFont
    
    # Create demo images directory
    demo_dir = Path(os.path.dirname(__file__)) / 'demo_images'
    demo_dir.mkdir(exist_ok=True)
    
    demo_images = []
    
    # Create images with scientific diagrams and text
    image_configs = [
        {
            'filename': 'quantum_circuit.png',
            'text': 'Superconducting Quantum Circuit\nQubit Entanglement Diagram',
            'size': (600, 400)
        },
        {
            'filename': 'warp_drive_concept.png',
            'text': 'Theoretical Warp Drive Spacetime Manipulation',
            'size': (600, 400)
        }
    ]
    
    for config in image_configs:
        # Create image
        img = Image.new('RGB', config['size'], color='white')
        draw = ImageDraw.Draw(img)
        
        # Use a default font
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()
        
        # Draw text
        draw.text((50, 50), config['text'], fill='black', font=font)
        
        # Save image
        image_path = demo_dir / config['filename']
        img.save(image_path)
        demo_images.append(str(image_path))
    
    return demo_images

async def demonstrate_multi_modal_entity_extraction():
    """
    Demonstrate multi-modal entity extraction capabilities.
    """
    # Initialize multi-modal entity extractor
    logger.info("Initializing Multi-Modal Entity Extractor")
    extractor = MultiModalEntityExtractor()
    
    # Create demo images
    demo_images = create_demo_images()
    
    # Demonstrate different extraction strategies
    logger.info("\n--- Multi-Modal Entity Extraction Demonstration ---")
    
    # 1. Textual Entity Extraction
    logger.info("\n1. Textual Entity Extraction")
    for text in SCIENTIFIC_TEXTS:
        logger.info(f"\nExtracting entities from text: {text}")
        entities = await extractor.extract_entities(
            text=text, 
            strategies=[EntityExtractionStrategy.TEXTUAL]
        )
        
        for entity in entities:
            logger.info(f"Entity: {entity.name}")
            logger.info(f"Type: {entity.type}")
            logger.info(f"Confidence: {entity.properties.get('confidence', 'N/A')}")
    
    # 2. Visual Entity Extraction
    logger.info("\n2. Visual Entity Extraction")
    for image_path in demo_images:
        logger.info(f"\nExtracting entities from image: {image_path}")
        entities = await extractor.extract_entities(
            image_path=image_path, 
            strategies=[EntityExtractionStrategy.VISUAL]
        )
        
        for entity in entities:
            logger.info(f"Entity: {entity.name}")
            logger.info(f"Type: {entity.type}")
            logger.info(f"Confidence: {entity.properties.get('confidence', 'N/A')}")
    
    # 3. Multi-Modal Extraction
    logger.info("\n3. Multi-Modal Entity Extraction")
    for text, image_path in zip(SCIENTIFIC_TEXTS[:2], demo_images):
        logger.info(f"\nCombined extraction for text and image")
        entities = await extractor.extract_entities(
            text=text,
            image_path=image_path,
            strategies=[
                EntityExtractionStrategy.TEXTUAL,
                EntityExtractionStrategy.VISUAL,
                EntityExtractionStrategy.CONTEXTUAL
            ]
        )
        
        logger.info("Extracted Entities:")
        for entity in entities:
            logger.info(f"Entity: {entity.name}")
            logger.info(f"Type: {entity.type}")
            logger.info(f"Confidence: {entity.properties.get('confidence', 'N/A')}")

async def main():
    """
    Main async entry point for the demonstration.
    """
    logger.info("Starting Multi-Modal Entity Extraction Demonstration")
    
    try:
        await demonstrate_multi_modal_entity_extraction()
    except Exception as e:
        logger.error(f"Demonstration failed: {e}", exc_info=True)
    
    logger.info("Multi-Modal Entity Extraction Demonstration Complete")

if __name__ == '__main__':
    asyncio.run(main())
