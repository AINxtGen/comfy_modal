import logging
import sys

def configure_logging():
    # Check if root logger already has handlers and return immediately to avoid adding new handlers
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return  # Already configured logging, do not add new handlers

    # Set root logger level
    root_logger.setLevel(logging.DEBUG)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Silence noisy loggers
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    logging.getLogger('hpack').setLevel(logging.WARNING)

    # Add other noisy loggers as needed
    logging.getLogger('httpcore.http11').setLevel(logging.WARNING)
    logging.getLogger('httpcore.connection').setLevel(logging.WARNING)
    logging.getLogger('hpack.hpack').setLevel(logging.WARNING)
    logging.getLogger('hpack.table').setLevel(logging.WARNING)
    
    # Add additional noisy loggers
    logging.getLogger('python_multipart.multipart').setLevel(logging.WARNING)
    logging.getLogger('PIL.PngImagePlugin').setLevel(logging.WARNING)

    # Add filelock logger
    logging.getLogger('filelock').setLevel(logging.WARNING)
