import torch
from pathlib import Path
import logging

from config import Config

logger = logging.getLogger('pong')

def save_checkpoint(episode, outdir, policy_net, optimizer, criterion, 
                    filename = 'checkpoint.pong.pth.tar'):
    """
    Saves policy network's checkpoint
    Args:
        episode (int): current episode
        outdir (Path): directory to output checkpoint
        policy_net (obj): cnn to map state to actions 
        optimizer (object): optimizer for the model
        filename (Path): for checkpoint in outdir
    """
    filename = outdir / filename
    torch.save({'episode': episode,
                'policy_net': policy_net.state_dict(),
                'optimizer': optimizer.state_dict(),
                }, str(filename))


def load_checkpoint(checkpoint_file):
    """
    Loads the checkpoint of the epoch, policy_net, optimizer.
    
    Args:
        checkpoint_file (Path): File name of latest checkpoint file.
    Returns:
        checkpoint (dict):
    
    Raises:
        warning: If the checkpoint file doesn't exist.
    """
    checkpoint = None
    try:
        checkpoint = torch.load(checkpoint_file, map_location=Config.get("device"))
        logger.info('Loading the checkpoint file')
    except:
        logger.warning('Checkpoint file does not exist')
    
    return checkpoint