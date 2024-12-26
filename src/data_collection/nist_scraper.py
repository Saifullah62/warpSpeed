import requests
import json
import os
from typing import Dict, Any, List, Optional
from bs4 import BeautifulSoup
import re
import urllib.parse

class NISTScraper:
    """Scraper for NIST Physical Measurement Laboratory data."""
    
    def __init__(self):
        self.base_url = "https://physics.nist.gov/cgi-bin/cuu"
        
    def get_constants(self) -> List[Dict[str, Any]]:
        """Fetch all physical constants from NIST."""
        try:
            # Get all constants
            response = requests.get(f"{self.base_url}/Category?view=html&All+values.x=95&All+values.y=11")
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            constants = []
            
            # Navigation links to skip
            skip_links = {
                'values', 'energyequivalents', 'searchablebibliography', 'background',
                'constantsbibliography', 'constants,units & uncertaintyhome page'
            }
            
            # Find all links that contain constant names
            for link in soup.find_all('a'):
                href = link.get('href', '')
                text = link.get_text(strip=True)
                if text and 'Value?' in href:  # Only get links that point to constant values
                    constant_id = text.lower().replace(' ', '_')
                    if constant_id not in skip_links and not any(skip in text.lower() for skip in ['try a new search']):
                        # Get the full URL by joining with base URL
                        full_url = urllib.parse.urljoin(self.base_url + "/", href)
                        constants.append({
                            'name': text,
                            'id': constant_id,
                            'url': full_url
                        })
            
            print(f"Found {len(constants)} constants")
            return constants
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching NIST constants: {str(e)}")
            return []
            
    def clean_text(self, text: str) -> str:
        """Clean up text by removing extra whitespace and standardizing symbols."""
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text.strip()
        text = text.replace('×', 'x')  # Standardize multiplication symbol
        return text

    def clean_unit(self, unit: str) -> str:
        """Clean up unit by removing extra text and standardizing format."""
        # Remove any text after common endings
        unit = re.sub(r'\s*(?:Click here|Source|Standard uncertainty).*$', '', unit, flags=re.IGNORECASE)
        return self.clean_text(unit)

    def get_constant_details(self, constant_id: str, url: str) -> Optional[Dict[str, Any]]:
        """Fetch detailed information about a specific constant."""
        try:
            print(f"\nFetching from URL: {url}")
            
            # Extract the base URL without the search parameters
            base_url = url.split('|')[0]
            response = requests.get(base_url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract the constant details
            details = {'symbol': constant_id}
            
            # Get all text content
            text = soup.get_text()
            
            # Get the constant name from the title, improved pattern
            title_match = re.search(r'CODATA Internationally recommended values?:?\s*(.*?)(?:\s*\(|$)', text, re.DOTALL | re.IGNORECASE)
            if title_match:
                name = self.clean_text(title_match.group(1))
                details['name'] = name
                print(f"Found name: {name}")
            
            # Find numerical value and unit with improved pattern
            value_pattern = r'(?:Value|Numerical value)\s*=?\s*([-\d.\s]+(?:\s*[x×]\s*10[-\d]+)?)\s*\t?\s*([A-Za-z][-A-Za-z\s\d/]*)?'
            value_match = re.search(value_pattern, text, re.DOTALL | re.IGNORECASE)
            
            if value_match:
                value = self.clean_text(value_match.group(1))
                details['value'] = value
                print(f"Found value: {value}")
                
                if value_match.group(2):
                    unit = self.clean_unit(value_match.group(2))
                    details['unit'] = unit
                    print(f"Found unit: {unit}")
            
            # Find uncertainty with improved pattern
            uncert_pattern = r'(?:Standard uncertainty|Uncertainty)\s*=?\s*([-\d.\s]+(?:\s*[x×]\s*10[-\d]+)?)'
            uncert_match = re.search(uncert_pattern, text, re.IGNORECASE)
            if uncert_match:
                uncertainty = self.clean_text(uncert_match.group(1))
                details['uncertainty'] = uncertainty
                print(f"Found uncertainty: {uncertainty}")
            elif re.search(r'(?:Standard uncertainty|Uncertainty).*?\(exact\)', text, re.DOTALL | re.IGNORECASE):
                details['uncertainty'] = 'exact'
                print("Found exact uncertainty")
            
            # Try to get concise form with improved pattern
            concise_pattern = r'Concise form\s*=?\s*([-\d.\s]+(?:\s*[x×]\s*10[-\d]+)?)\s*\t?\s*([A-Za-z][-A-Za-z\s\d/]*)?'
            concise_match = re.search(concise_pattern, text, re.DOTALL | re.IGNORECASE)
            if concise_match:
                concise_value = self.clean_text(concise_match.group(1))
                details['concise_value'] = concise_value
                print(f"Found concise value: {concise_value}")
                
                if concise_match.group(2):
                    concise_unit = self.clean_unit(concise_match.group(2))
                    details['concise_unit'] = concise_unit
                    print(f"Found concise unit: {concise_unit}")
            
            # For debugging, print the raw text if no value was found
            if 'value' not in details:
                print("\nNo value found in text. Raw text content:")
                print(text[:500])  # Print first 500 chars
            
            return details if 'value' in details else None
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching constant details: {str(e)}")
            return None
            
    def save_constants_to_file(self, filename: str = "nist_constants.json", output_dir: str = "data/nist"):
        """Fetch all constants and save them to a JSON file."""
        constants = self.get_constants()
        if constants:
            # Get details for each constant
            constants_with_details = []
            for constant in constants:
                print(f"\nFetching details for {constant['name']}...")
                details = self.get_constant_details(constant['id'], constant['url'])
                if details:
                    constant.update(details)
                    constants_with_details.append(constant)
            
            try:
                # Create output directory if it doesn't exist
                os.makedirs(output_dir, exist_ok=True)
                # Use os.path.join for path construction
                output_path = os.path.normpath(os.path.join(output_dir, filename))
                
                with open(output_path, 'w') as f:
                    json.dump(constants_with_details, f, indent=2)
                print(f"\nSuccessfully saved {len(constants_with_details)} constants with details to {output_path}")
            except IOError as e:
                print(f"Error saving constants to file: {str(e)}")
        else:
            print("No constants to save")
