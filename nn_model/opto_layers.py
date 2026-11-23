"""
This source code defines a convolutional neural network module
that transforms optogenetic video stimuli into external inputs
for excitatory neurons in layer 2/3 of the primary visual cortex.
"""

import torch
import torch.nn as nn

class OptoConvToExcL23(nn.Module):
    """
    Convolutional neural network module for processing optogenetic stimulation.

    Transforms spatial optogenetic video frames into external input
    for excitatory neurons in layer 2/3 of the primary visual cortex.
    """

    def __init__(
        self,
        n_e23_neurons: int,
        video_height: int = 400,
        video_width: int = 400,
        n_filters: int = 8,
    ):
        """
        Initialize the optogenetic input processing module.

        :param n_e23_neurons: Number of excitatory neurons in layer 2/3.
        :param video_height: Height of the input video frames.
        :param video_width: Width of the input video frames.
        :param n_filters: Number of convolutional filters.
        """
        super().__init__()

        self.n_e23_neurons = n_e23_neurons
        self.video_height = video_height
        self.video_width = video_width

        self.conv = nn.Sequential(
            nn.Conv2d(1, n_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        self.to_e23 = nn.Linear(
            n_filters * (video_height // 8) * (video_width // 8),
            n_e23_neurons,
        )

    def forward(self, opto_video: torch.Tensor) -> torch.Tensor:
        """
        Process a sequence of optogenetic stimulation frames.

        :param opto_video: Input tensor of shape (batch, time, height, width).
        :return: Tensor of external inputs to excitatory layer 2/3 neurons,
            with shape (batch, time, n_e23_neurons).
        """

        # Merge batch and time dimensions for convolutional processing.
        batch_size, time_steps, height, width = opto_video.shape

        if height != self.video_height or width != self.video_width:
            raise ValueError(
                f"Expected opto video size "
                f"({self.video_height}, {self.video_width}), "
                f"got ({height}, {width})."
            )

        x = opto_video.reshape(batch_size * time_steps, 1, height, width)

        # Extract spatial features and project them onto excitatory neurons.
        x = self.conv(x)
        x = x.reshape(batch_size * time_steps, -1)
        x = self.to_e23(x)

        # Restore the batch and time dimensions.
        x = x.reshape(batch_size, time_steps, self.n_e23_neurons)
        return x
