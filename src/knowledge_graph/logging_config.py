import os
import logging
import logging.config
from datetime import datetime
from pathlib import Path

def setup_logging(
    log_dir: str = 'logs/knowledge_graph',
    log_level: str = 'INFO',
    enable_console: bool = True,
    enable_file: bool = True
) -> None:
    """
    Configure comprehensive logging for knowledge graph components.
    
    Args:
        log_dir: Directory to store log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_console: Enable console logging
        enable_file: Enable file logging
    """
    # Ensure log directory exists
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Generate unique log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = log_path / f"knowledge_graph_{timestamp}.log"
    
    # Logging configuration dictionary
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'colored': {
                '()': 'colorlog.ColoredFormatter',
                'format': '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
                'log_colors': {
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white'
                }
            }
        },
        'handlers': {},
        'loggers': {
            'knowledge_graph': {
                'handlers': [],
                'level': log_level.upper(),
                'propagate': True
            }
        }
    }
    
    # Console Handler
    if enable_console:
        logging_config['handlers']['console'] = {
            'level': log_level.upper(),
            'class': 'logging.StreamHandler',
            'formatter': 'colored' if _is_colorlog_available() else 'standard'
        }
        logging_config['loggers']['knowledge_graph']['handlers'].append('console')
    
    # File Handler
    if enable_file:
        logging_config['handlers']['file'] = {
            'level': log_level.upper(),
            'class': 'logging.FileHandler',
            'filename': str(log_filename),
            'mode': 'a',
            'formatter': 'standard',
            'encoding': 'utf-8'
        }
        logging_config['loggers']['knowledge_graph']['handlers'].append('file')
    
    # Configure logging
    logging.config.dictConfig(logging_config)

def _is_colorlog_available() -> bool:
    """
    Check if colorlog is available.
    
    Returns:
        Boolean indicating colorlog availability
    """
    try:
        import colorlog
        return True
    except ImportError:
        return False

def get_logger(name: str = 'knowledge_graph'):
    """
    Get a configured logger for the specified component.
    
    Args:
        name: Logger name (defaults to 'knowledge_graph')
    
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)

# Performance and Error Logging Decorator
def log_performance(logger=None):
    """
    Decorator to log function performance and potential errors.
    
    Args:
        logger: Optional logger instance
    
    Returns:
        Decorated function
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            log = logger or get_logger(func.__module__)
            try:
                import time
                start_time = time.time()
                result = await func(*args, **kwargs)
                end_time = time.time()
                
                # Log performance metrics
                log.info(
                    f"Function {func.__name__} executed in {end_time - start_time:.4f} seconds"
                )
                
                return result
            
            except Exception as e:
                log.error(
                    f"Error in {func.__name__}: {str(e)}",
                    exc_info=True
                )
                raise
        
        return wrapper
    return decorator

# Initialize logging when module is imported
setup_logging()
