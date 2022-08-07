import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CIFARNet(nn.Module):

    def __init__(self, num_classes=10):

        super(CIFARNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2, bias=False)  # 3 * 32 * 5 * 5 =
        self.conv2 = nn.Conv2d(32, 32, 5, 1, 2, bias=False)
        self.conv3 = nn.Conv2d(32, 64, 5, 1, 2, bias=False)

        self.fc1 = nn.Linear(576, 64)
        self.fc2 = nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                #n = m.weight.size(1)
                #m.weight.data.normal_(0, 0.01)
                #m.bias.data.zero_()
                pass
    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(F.max_pool2d(x, 3, 2))

        x = self.conv2(x)
        x = F.avg_pool2d(F.relu(x), 3, 2)

        x = self.conv3(x)
        x = F.avg_pool2d(F.relu(x), 3, 2)

        x = x.view(-1, 576)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

if __name__ == '__main__':

    net = CIFARNet()
    inputs = torch.rand([1, 3, 32, 32])
    outputs = net(inputs)
