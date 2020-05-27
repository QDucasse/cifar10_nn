# -*- coding: utf-8 -*-

# cifar10_nn
# author - Quentin Ducasse
# https://github.com/QDucasse
# quentin.ducasse@ensta-bretagne.org

# Dataset exploration and visualisation

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt


class Visualiser(object):
    def __init__(self,batch_size):
        self.loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10('data', download=True,
                                        transform=torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(
                                            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])),
                            batch_size=batch_size, shuffle=True, num_workers = 2)
        self.classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    def display_6items(self):
        '''Shows six items along with their labels out of the dataset'''
        examples = enumerate(self.loader)
        batch_idx, (example_data, example_targets) = next(examples)
        fig = plt.figure()
        for i in range(6):
            plt.subplot(2,3,i+1)
            plt.tight_layout()
            img = example_data[i] / 2 + 0.5
            npimg = np.transpose(img.numpy(), (1,2,0))
            plt.imshow(npimg, interpolation='none')
            plt.title("Ground Truth: {}".format(self.classes[example_targets[i]]))
            plt.xticks([])
            plt.yticks([])
        plt.show()

if __name__ == "__main__":
    # Visualiser initialization and image display
    visualiser = Visualiser(batch_size = 64)
    visualiser.display_6items()
