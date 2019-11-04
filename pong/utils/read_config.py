import configparser
import logging

from config import Config

logger = logging.getLogger('captioning')

def reading_config(file_path):
    """
    Reads the config settings from config file and makes them accessible to the project using config.py module.
    Args:
        file_path (Path): The path of config file.
    
    Raises:
        FileNotFoundError: If the config file doesn't exist.
    """
    # TODO (aq): Raise Error if file doesn't exist. 
    config = configparser.ConfigParser()
    try:
        config.read(file_path)
        logger.info('Reading the config file from: %s' % file_path)
    except FileNotFoundError:
        logger.exception("Config file doesn't exist.")

    #GPUs
    Config.set("disable_cuda", config.getboolean("GPU", "disable_cuda", fallback=False))
    if not Config.get("disable_cuda") and torch.cuda.is_available():
        Config.set("device", "cuda")
        logger.info('GPU is available')
    else:
        Config.set("device", "cpu")
        logger.info('Only CPU is available')

    #Training
    Config.set("training_batch_size", config.getint("training", "batch_size", fallback=32))
    Config.set("episodes", config.getint("training", "episodes", fallback=1000))
    Config.set("gamma", config.getfloat("training", "gamma", fallback=0.99))
    Config.set("learning_rate", config.getfloat("training", "learning_rate", fallback=0.01))

    #logging
    Config.set("logfile", config.get("logging", "logfile", fallback="output.log"))