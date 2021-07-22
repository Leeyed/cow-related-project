import torch
import torch.nn as nn
import math
import torch.nn.utils.weight_norm as weight_norm
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0],-1)

def get_output_shape(model, shape):
    batch_size = 1
    input_data = torch.rand(batch_size, *shape, requires_grad=False)
    output_data = model(input_data)
    output_data = output_data.view(1, -1)
    return output_data.size()[1]

def initialize(model, std=0.1):
    for p in model.parameters():
        p.data.normal_(0,std)

    # init last linear layer of the transformer at 0
    model.transformer.net[-1].weight.data.zero_()
    model.transformer.net[-1].bias.data.copy_(torch.eye(3).flatten()[:model.transformer.net[-1].out_features])
    # NOTE: this initialization the last layer of the transformer layer to identity here means the apply_tranform function should not
    #       add an identity matrix when converting coordinates



class STN(nn.Module):
    def __init__(self, input_shape, out_dim=6):
        super(STN, self).__init__()

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(input_shape[0], 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            Flatten()
        )

        output_size = get_output_shape(self.localization, input_shape)

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(output_size, 32),
            nn.ReLU(True),
            nn.Linear(32, out_dim)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)
        return x
