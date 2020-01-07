import configparser
import logging
import torch

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
    
    #environment
    Config.set("env_name", config.get("environment", "env_name", fallback = 'PongNoFrameskip-v4'))

    #policy
    Config.set("feature_extraction", config.getboolean("policy", "feature_extraction", fallback = False))

    #target network
    Config.set("target_update", config.getint("target_network", "update_weights", fallback=10))
    #replay memory
    Config.set("memory_size", config.getint("replay_memory", "total_size", fallback=10000))

    #Training
    Config.set("training_batch_size", config.getint("training", "batch_size", fallback=32))
    Config.set("episodes", config.getint("training", "episodes", fallback=1000))
    Config.set("gamma", config.getfloat("training", "gamma", fallback=0.75))
    Config.set("learning_rate", config.getfloat("training", "learning_rate", fallback=0.01))
    Config.set("epsilon_start", config.getfloat("training", "epsilon_start", fallback=0.9))
    Config.set("epsilon_end", config.getfloat("training", "epsilon_end", fallback=0.05))
    Config.set("epsilon_decay", config.getfloat("training", "epsilon_decay", fallback=200))

    #logging
    Config.set("logfile", config.get("logging", "logfile", fallback="output.log"))

    #paths 
    #Config.set("output_dir", config.get("paths", "output_dir", fallback = "output"))
    Config.set("checkpoint_file", config.get("paths", "checkpoint_file", fallback = "checkpoint.pong.pth.tar"))