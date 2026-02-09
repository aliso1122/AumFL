import torch.nn as nn
import torch


class MLP(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        hidden_size2=36,
        init="n",
        name="global;MLP",
    ) -> None:
        super(MLP, self).__init__()
        self.name = name
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)

        # self.flatten = nn.Flatten()
        self.activation = nn.ELU(0.2)

        # self.activation = nn.ReLU(0.2)
        # self.activation = nn.Sigmoid()
        # self.bn = nn.BatchNorm1d(input_size, affine=False)
        if init == "y":
            self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, mean=0, std=0.02)

    def set_name(self, name):
        self.name = name

    def forward(self, x):
        # print(x)
        # x = self.bn(x)
        # print(x)
        # assert False
        # x = self.flatten(x)
        
        x = self.fc1(x)

        x = self.activation(x)
        # print(x)

        x = self.fc2(x)

        x = self.activation(x)

        x = self.fc3(x)

        # x = torch.softmax(x, -1)

        # x = self.fc4(x)
        x = self.activation(x)

        return x


class MLP2(nn.Module):

    def __init__(
        self, input_size, hidden_size, output_size, init="n", name="global;MLP"
    ) -> None:
        super(MLP2, self).__init__()
        self.name = name
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        # self.flatten = nn.Flatten()
        self.activation = nn.ELU(0.2)
        # self.activation = nn.Sigmoid()
        # self.bn = nn.BatchNorm1d(input_size, affine=False)
        if init == "y":
            self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0.02)

    def set_name(self, name):
        self.name = name

    def forward(self, x):
        # print(x)
        # x = self.bn(x)
        # print(x)
        # assert False
        # x = self.flatten(x)

        x = self.fc1(x)

        x = self.activation(x)

        x = self.fc2(x)

        x = self.activation(x)

        # x = self.fc4(x)
        # x = self.activation(x)

        return x


class MLP1(nn.Module):

    def __init__(self, input_size, output_size, init="n", name="global;MLP") -> None:
        super(MLP1, self).__init__()
        self.name = name
        self.fc1 = nn.Linear(input_size, output_size)

        if init == "y":
            self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0.02)

    def set_name(self, name):
        self.name = name

    def forward(self, x):
        # print(x)
        # x = self.bn(x)
        # print(x)
        # assert False
        # x = self.flatten(x)
        x = self.fc1(x)
        # x = self.fc4(x)
        # x = self.activation(x)

        return x
