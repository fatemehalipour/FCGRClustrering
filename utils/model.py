from torch import nn
from torch.nn.functional import normalize


class BackBoneModel(nn.Module):
    def __init__(self,
                 input_shape: int,
                 output_shape: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            # Create the conv layer
            nn.Conv2d(in_channels=input_shape,
                      out_channels=128,
                      kernel_size=7,
                      stride=2,
                      padding=1),  # values we can set our NN's are called hyperparameters.
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=7,
                      stride=2,
                      padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2)
        )
        self.linear_layer = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.0, inplace=True),
            nn.Linear(in_features=4608,  # there is a trick to calculating this
                      out_features=output_shape),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        # x = self.conv_block_2(x)
        x = self.linear_layer(x)
        return x


class Network(nn.Module):
    def __init__(self, backbone, rep_dim, feature_dim, class_num):
        super(Network, self).__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.cluster_num = class_num
        self.instance_projector = nn.Sequential(
            nn.Linear(rep_dim, rep_dim),
            nn.ReLU(),
            nn.Linear(rep_dim, self.feature_dim),
        )
        self.cluster_projector = nn.Sequential(
            nn.Linear(rep_dim, rep_dim),
            nn.ReLU(),
            nn.Linear(rep_dim, self.cluster_num),
            nn.Softmax(dim=1)
        )

    def forward(self, x_i, x_j):
        h_i = self.backbone(x_i)
        h_j = self.backbone(x_j)

        z_i = normalize(self.instance_projector(h_i), dim=1)
        z_j = normalize(self.instance_projector(h_j), dim=1)

        c_i = self.cluster_projector(h_i)
        c_j = self.cluster_projector(h_j)

        return z_i, z_j, c_i, c_j

    def forward_cluster(self, x):
        h = self.backbone(x)
        c = self.cluster_projector(h)
        # c = torch.argmax(c, dim=1)
        return c
