import torchvision
import torchvision.utils
import torch
import torch.nn as nn
import torchvision.models as vision_models

class Resnet18(nn.Module):
    """
    Maps a given batch of game screenshots to actions.

    Attributes:
        n_actions (int): No of actions in the game.
        feature_extraction (bool): Feature extraction or fine-tuning.
    """
    def __init__(self, n_actions, feature_extraction):
        super(Resnet18, self).__init__()
        self.n_actions = n_actions
        self.resnet18 = vision_models.resnet18(pretrained=True)
        #fine-tuning or feature extraction
        if feature_extraction:
            for param in self.resnet18.parameters():
                param.requires_grad = False
        #replace the classifer with fully connected embedding layer
        self.resnet18.classifer = nn.Linear(in_features = 1000,  out_features = 1000)
        #another fully connected layer
        self.embed = nn.Linear(in_features = 1000, out_features = self.n_actions)
        #dropout layer
        self.dropout = nn.Dropout(p = 0.5)
        #activation layer
        self.prelu = nn.PReLU()
    
    def forward(self, images):
        """
        Maps the given images to action space.

        Args:
            images (tensor): Screenshots of the game screen.

        Returns:
            actions (tensor): A float tensor of numbers for each actions.
        """
        resnet18_output = self.dropout(self.resnet18(images))
        actions = self.prelu(self.embed(resnet18_output))

        return actions