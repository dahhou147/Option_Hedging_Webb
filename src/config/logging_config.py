import logging
import colorlog

def setup_logger(name: str = 'mon_logger', log_file: str = None):
    """
    Configure et retourne un logger avec des logs colorés dans la console et optionnellement dans un fichier.

    Args:
        name (str): Nom du logger.
        log_file (str): Chemin vers le fichier de log (optionnel).

    Returns:
        logging.Logger: Logger configuré.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  

    console_handler = colorlog.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger
