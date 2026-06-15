import torch
import torch.nn as nn

class OptoConvToExcL23(nn.Module):
    """
    Converts optogenetic video into external input for V1_Exc_L23 neurons.

    Input:
        opto_video: (batch, time, height, width)

    Output:
        opto_e23:   (batch, time, n_e23_neurons)
    """

    def __init__(
        self,
        n_e23_neurons: int,
        video_height: int = 400,
        video_width: int = 400,
        n_filters: int = 8,
    ):
        super().__init__()

        self.n_e23_neurons = n_e23_neurons
        self.video_height = video_height
        self.video_width = video_width

        self.conv = nn.Sequential(
            nn.Conv2d(1, n_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.to_e23 = nn.Linear(
            n_filters * video_height * video_width,
            n_e23_neurons,
        )

    def forward(self, opto_video: torch.Tensor) -> torch.Tensor:
        """
        opto_video: (batch, time, height, width)
        """

        batch_size, time_steps, height, width = opto_video.shape

        if height != self.video_height or width != self.video_width:
            raise ValueError(
                f"Expected opto video size "
                f"({self.video_height}, {self.video_width}), "
                f"got ({height}, {width})."
            )

        x = opto_video.reshape(batch_size * time_steps, 1, height, width)

        x = self.conv(x)
        x = x.reshape(batch_size * time_steps, -1)
        x = self.to_e23(x)

        x = x.reshape(batch_size, time_steps, self.n_e23_neurons)
        return x
