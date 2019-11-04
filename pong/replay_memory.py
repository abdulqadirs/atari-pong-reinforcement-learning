import logging
import random

logger = logging.getLogger('pong')

class ReplayMemory():
    """
    Stores the experience(s,a,r,s') in replay memory.
    
    Attributes:
        total_size (int): total size of memory.
    """
    def __init__(self, total_size):
        self.memory = []
        self.total_size = total_size
        self.current_size = 0

    def push(self, experience):
        """
        Pushes the given experience to the replay memory list. 
        If the memory is full then it replaces the oldest experience.

        Args:
            experience (named tuple): Most recent experience.
        """
        if len(self.memory) < self.total_size:
            self.memory.append(experience)
        else:
            self.memory[self.current_size % self.total_size] = experience
        self.current_size += 1
    
    def get_size(self):
        """
        Returns the current size of memory.
        """
        return len(self.memory)
    
    def sample(self, batch_size):
        """
        Randomly samples a batch from memory.

        Args:
            batch_size (int): No of elements in the batch.
        
        Returns:
            List of experiences randomly sampled from memory.
        
        Raises:
            IndexError: If the size of memory is less than batch size.
        """
        try:
            return random.sample(self.memory, batch_size)
        except IndexError:
            logger.exception('Batch size is greater than the size of memory.')