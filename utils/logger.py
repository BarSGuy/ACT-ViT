import logging
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler

# Get project root directory (parent of utils directory)
PROJECT_ROOT = Path(__file__).parent.parent
# Ensure logs directory exists in project root
LOG_DIR = PROJECT_ROOT / "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Default log file name
DEFAULT_LOG_FILE = "app.log"

# Dictionary to store loggers by log file name
_loggers = {}

# Log format
formatter = logging.Formatter("%(asctime)s - %(name)s - [%(levelname)s] - %(filename)s:%(lineno)d - %(message)s")

def get_logger(log_file_name=None):
    """
    Returns a configured logger instance.
    
    Args:
        log_file_name (str, optional): Name of the log file (without .log extension).
                                      If None, defaults to "app.log".
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    # Use default if no name provided
    if log_file_name is None:
        log_file_name = DEFAULT_LOG_FILE
    else:
        # Ensure .log extension is present
        if not log_file_name.endswith('.log'):
            log_file_name = f"{log_file_name}.log"
    
    # Return existing logger if already created
    if log_file_name in _loggers:
        return _loggers[log_file_name]
    
    # Create a new logger instance
    logger_name = f"Logger_{log_file_name.replace('.log', '')}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)  # Capture all levels
    
    # File Handler (Rotating logs)
    log_file_path = LOG_DIR / log_file_name
    # Create parent directories if they don't exist (in case log_file_name contains slashes)
    os.makedirs(log_file_path.parent, exist_ok=True)
    file_handler = RotatingFileHandler(str(log_file_path), maxBytes=5_000_000, backupCount=5)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Store logger for future use
    _loggers[log_file_name] = logger
    
    return logger
