import os
import json
import datasets
from typing import Dict, List, Optional

logger = datasets.logging.get_logger(__name__)

_CITATION = """\
@dataset{warp_drive_dataset,
  title={Warp Drive Research Dataset},
  author={GotThatData},
  year={2024},
  publisher={Hugging Face},
  url={https://huggingface.co/datasets/GotThatData/warp-speed}
}
"""

_DESCRIPTION = """\
The Warp Drive Research Dataset is a comprehensive collection of scientific research papers, 
experimental data, and theoretical materials focused on physics concepts that could enable 
faster-than-light travel. It aggregates information from leading physics institutions and 
repositories worldwide, covering quantum physics, spacetime manipulation, exotic matter, 
and advanced propulsion concepts.
"""

_HOMEPAGE = "https://huggingface.co/datasets/GotThatData/warp-speed"

_LICENSE = "cc-by-4.0"

_URLS = {
    "default": "https://huggingface.co/datasets/GotThatData/warp-speed",
}

class WarpSpeedDataset(datasets.GeneratorBasedBuilder):
    """Warp Drive Research Dataset: A collection of physics papers and research materials."""

    VERSION = datasets.Version("1.0.0")

    def _info(self) -> datasets.DatasetInfo:
        features = datasets.Features({
            "title": datasets.Value("string"),
            "authors": datasets.Sequence(datasets.Value("string")),
            "abstract": datasets.Value("string"),
            "arxiv_id": datasets.Value("string"),
            "version": datasets.Value("string"),
            "category": datasets.Value("string"),
            "subcategory": datasets.Value("string"),
            "pdf_path": datasets.Value("string"),
            "metadata": datasets.Value("string"),
        })
        
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        urls = _URLS["default"]
        data_dir = dl_manager.download_and_extract(urls)
        
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_dir": data_dir,
                    "split": "train",
                },
            ),
        ]

    def _validate_paper_metadata(self, paper: Dict) -> bool:
        """Validate paper metadata for required fields and format."""
        required_fields = {
            'title': str,
            'authors': list,
            'abstract': str,
            'arxiv_id': str,
            'category': str
        }
        
        try:
            # Check required fields and types
            for field, field_type in required_fields.items():
                if field not in paper:
                    logger.error(f"Missing required field: {field}")
                    return False
                if not isinstance(paper[field], field_type):
                    logger.error(f"Invalid type for {field}: expected {field_type}, got {type(paper[field])}")
                    return False
                    
            # Validate specific field contents
            if not paper['authors']:
                logger.error("Paper must have at least one author")
                return False
                
            if not paper['abstract'].strip():
                logger.error("Paper must have non-empty abstract")
                return False
                
            # Validate arXiv ID format
            if not self._validate_arxiv_id(paper['arxiv_id']):
                logger.error(f"Invalid arXiv ID format: {paper['arxiv_id']}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Metadata validation error: {str(e)}")
            return False
            
    def _validate_arxiv_id(self, arxiv_id: str) -> bool:
        """Validate arXiv ID format."""
        import re
        # Basic arXiv ID format validation (simplified)
        pattern = r'^\d{4}\.\d{4,5}(v\d+)?$'
        return bool(re.match(pattern, arxiv_id))

    def _generate_examples(self, data_dir: str, split: str) -> Dict:
        """Yields examples as (key, example) tuples with enhanced validation."""
        metadata_path = os.path.join(data_dir, "metadata", "papers_metadata.json")
        
        try:
            with open(metadata_path, encoding="utf-8") as f:
                papers_metadata = json.load(f)
                
            valid_papers = 0
            invalid_papers = 0
            
            for idx, paper in enumerate(papers_metadata):
                if not self._validate_paper_metadata(paper):
                    invalid_papers += 1
                    continue
                    
                valid_papers += 1
                yield idx, {
                    "title": paper.get("title", ""),
                    "authors": paper.get("authors", []),
                    "abstract": paper.get("abstract", ""),
                    "arxiv_id": paper.get("arxiv_id", ""),
                    "version": paper.get("version", ""),
                    "category": paper.get("category", ""),
                    "subcategory": paper.get("subcategory", ""),
                    "pdf_path": paper.get("pdf_path", ""),
                    "metadata": json.dumps(paper.get("metadata", {})),
                }
                
            logger.info(f"Processed {valid_papers} valid papers and found {invalid_papers} invalid papers")
            
        except Exception as e:
            logger.error(f"Error generating examples: {str(e)}")
            raise
