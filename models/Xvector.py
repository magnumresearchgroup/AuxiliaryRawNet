import torch  # noqa: F401
import torch.nn as nn
import speechbrain as sb
from speechbrain.nnet.pooling import StatisticsPooling
from speechbrain.nnet.CNN import Conv1d
from speechbrain.nnet.linear import Linear
from speechbrain.nnet.normalization import BatchNorm1d
from speechbrain.nnet.normalization import BatchNorm1d as _BatchNorm1d
import torch.nn.functional as F
class Xvector(torch.nn.Module):
    """This model extracts X-vectors for speaker recognition
    Arguments
    ---------
    activation : torch class
        A class for constructing the activation layers.
    tdnn_blocks : int
        Number of time-delay neural (TDNN) layers.
    tdnn_channels : list of ints
        Output channels for TDNN layer.
    tdnn_kernel_sizes : list of ints
        List of kernel sizes for each TDNN layer.
    tdnn_dilations : list of ints
        List of dilations for kernels in each TDNN layer.
    lin_neurons : int
        Number of neurons in linear layers.
    Example
    -------
    >>> compute_xvect = Xvector()
    >>> input_feats = torch.rand([5, 10, 40])
    >>> outputs = compute_xvect(input_feats)
    >>> outputs.shape
    torch.Size([5, 1, 512])
    """

    def __init__(
        self,
        device="cpu",
        activation=torch.nn.LeakyReLU,
        tdnn_blocks=5,
        tdnn_channels=[512, 512, 512, 512, 1500],
        tdnn_kernel_sizes=[5, 3, 3, 1, 1],
        tdnn_dilations=[1, 2, 3, 1, 1],
        lin_neurons=512,
        in_channels=40,
    ):

        super().__init__()
        self.blocks = nn.ModuleList()

        # TDNN has convolutional layers with the given dilation factors
        # and kernel sizes. We here loop over all the convolutional layers
        # that we wanna add. Note that batch normalization is used after
        # the activations function in this case. This improves the
        # speaker-id performance a bit.
        for block_index in range(tdnn_blocks):
            out_channels = tdnn_channels[block_index]
            self.blocks.extend(
                [
                    Conv1d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=tdnn_kernel_sizes[block_index],
                        dilation=tdnn_dilations[block_index],
                    ),
                    activation(),
                    BatchNorm1d(input_size=out_channels),
                ]
            )
            in_channels = tdnn_channels[block_index]

        # Statistical pooling. It converts a tensor of variable length
        # into a fixed-length tensor. The statistical pooling returns the
        # mean and the standard deviation.
        self.blocks.append(StatisticsPooling())

        # Final linear transformation.
        # self.blocks.append(
        #     Linear(
        #         input_size=out_channels * 2,  # mean + std,
        #         n_neurons=lin_neurons,
        #         bias=True,
        #         combine_dims=False,
        #     )
        # )

    def forward(self, x, lens=None):
        """Returns the x-vectors.
        Arguments
        ---------
        x : torch.Tensor
        """

        for layer in self.blocks:
            try:
                x = layer(x, lengths=lens)
            except TypeError:
                x = layer(x)
        return x.transpose(1,2)
