# -*- coding: utf-8 -*-

# cifar10_nn
# author - Quentin Ducasse
# https://github.com/QDucasse
# quentin.ducasse@ensta-bretagne.org

# Trained model loading and inference

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from cifar10_nn.networks import LeNet, MLPNet
from cifar10_nn.core     import Trainer # Class definition must be visible to load

class Inferencer(object):
    def __init__(self,batch_size,network):
        self.loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10('data', train=False, download=True,
                                 transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                     (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                 ])),
            batch_size=batch_size, shuffle=True)
        self.classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        self.network = network
        self.load_trained_network()

    def display_6predictions(self):
        '''Displays the output of running the network over 6 inputs from the dataset'''
        # Run the network through some examples
        examples = enumerate(self.loader)
        batch_idx, (example_data, example_targets) = next(examples)
        with torch.no_grad():
            output = network(example_data)

        # Plot the predictions
        fig = plt.figure()
        for i in range(6):
            plt.subplot(2,3,i+1)
            plt.tight_layout()
            img = example_data[i] / 2 + 0.5
            npimg = np.transpose(img.numpy(), (1,2,0))
            plt.imshow(npimg, interpolation='none')
            plt.title("Prediction: {}".format(
            self.classes[output.data.max(1, keepdim=True)[1][i].item()]))
            plt.xticks([])
            plt.yticks([])
        plt.show()

    def load_trained_network(self):
        '''Load a saved trained network model'''
        path = 'results/trained_' + self.network.name + '.tar'
        dict = torch.load(path)
        self.network.load_state_dict(dict['network_state_dict'])
        self.network.eval()

if __name__ == "__main__":
    network = LeNet()
    inferencer = Inferencer(batch_size = 64, network = network)
    inferencer.display_6predictions()
