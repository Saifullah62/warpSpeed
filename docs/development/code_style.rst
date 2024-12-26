Code Style Guide
===============

This document outlines our coding standards and style guidelines.

Python Code Style
----------------

We follow PEP 8 with some specific adaptations:

Line Length
~~~~~~~~~~
- Maximum line length is 100 characters
- Exception for URLs and long strings that cannot be split

Imports
~~~~~~~
- Use absolute imports
- Group imports in the following order:
  1. Standard library
  2. Third-party packages
  3. Local application imports
- Use ``isort`` to automatically sort imports

.. code-block:: python

    # Standard library
    import os
    import sys
    
    # Third-party packages
    import numpy as np
    import pandas as pd
    
    # Local application imports
    from src.utils import helpers

Naming Conventions
~~~~~~~~~~~~~~~~
- Classes: ``UpperCamelCase``
- Functions and variables: ``lower_snake_case``
- Constants: ``UPPER_SNAKE_CASE``
- Private attributes: ``_single_leading_underscore``
- "Internal" attributes: ``__double_leading_underscore``

Type Hints
~~~~~~~~~
Use type hints for all function definitions:

.. code-block:: python

    def process_data(data: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        """Process the input DataFrame.
        
        Args:
            data: Input DataFrame to process
            threshold: Filtering threshold
            
        Returns:
            Processed DataFrame
        """
        return data[data['value'] > threshold]

Documentation
~~~~~~~~~~~~
Use Google-style docstrings:

.. code-block:: python

    def complex_function(param1: str, param2: int) -> bool:
        """Short description of function.
        
        Longer description if needed. Can be multiple lines.
        
        Args:
            param1: Description of param1
            param2: Description of param2
            
        Returns:
            Description of return value
            
        Raises:
            ValueError: Description of when this error occurs
            KeyError: Description of when this error occurs
        """
        pass

Code Organization
~~~~~~~~~~~~~~~
- One class per file (with rare exceptions)
- Related functions should be grouped in modules
- Use meaningful file names that reflect content

Testing
~~~~~~~
- Write tests for all new code
- Use descriptive test names: ``test_when_[condition]_then_[expected]``
- Group related tests in classes
- Use appropriate fixtures and parametrization

.. code-block:: python

    def test_when_input_valid_then_returns_true():
        assert validate_input("valid_input") is True

    def test_when_input_invalid_then_raises_error():
        with pytest.raises(ValueError):
            validate_input(None)

Error Handling
~~~~~~~~~~~~
- Use specific exception types
- Always include error messages
- Handle exceptions at appropriate levels

.. code-block:: python

    try:
        process_data(input_data)
    except ValueError as e:
        logger.error("Invalid data format: %s", str(e))
        raise
    except Exception as e:
        logger.exception("Unexpected error during processing")
        raise ProcessingError(f"Failed to process data: {str(e)}") from e

Logging
~~~~~~
- Use the ``logging`` module
- Include appropriate context
- Use correct log levels

.. code-block:: python

    logger = logging.getLogger(__name__)
    
    logger.debug("Processing started with params: %s", params)
    logger.info("Successfully processed %d records", count)
    logger.warning("Resource usage above 80%")
    logger.error("Failed to connect to database")
