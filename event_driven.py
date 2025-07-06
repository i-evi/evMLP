import torch
import torch.nn as nn
from thop import profile

class IndexGenerator(nn.Module):
    def __init__(self, image_size, image_channels, config, event_threshold, device):
        super(IndexGenerator, self).__init__()
        self.f = []
        self.w = []
        self.rearrange = []
        self.numw = image_size
        self.event_threshold = event_threshold
        for i in config:
            self.f.append(nn.AvgPool2d(i[1]))
            self.numw = self.numw // i[1]
            self.w.append(torch.arange(1, self.numw ** 2 + 1,dtype=torch.float32).to(device))

    def forward(self, x, y):
        batch_size = x.shape[0]
        diff = ((x - y).abs() > self.event_threshold) * 1.
        index = []
        for i, f in enumerate(self.f):
            diff = f(diff)
            index.append(
                (diff.mean(1).reshape([batch_size, -1]) != 0 ) * self.w[i]
            )
        return index

def avoid_dropout(module, depth=0):
    if str(module).startswith("Dropout"):
        module.p = 0.
    for child in module.children():
        avoid_dropout(child, depth + 1)

class EventDrivenEvMLP():
    def __init__(self, net, event_threshold, device="cuda:0"):
        super(EventDrivenEvMLP, self).__init__()
        self.net = net
        self.net.eval()
        avoid_dropout(self.net)
        self.index_generator = IndexGenerator(
            image_size = net.image_size,
            image_channels = net.image_channels,
            config = net.config,
            event_threshold = event_threshold,
            device = device
        )
        # Create buffer for features
        self.features = [torch.zeros(0, dtype=torch.float32).to(device)]
        feature_size = net.image_size
        for i in net.config:
            feature_size = feature_size // i[1]
            self.features.append(torch.zeros([feature_size ** 2, i[0]], dtype=torch.float32).to(device))
        self.prev_image = torch.ones(
            [1, net.image_channels, net.image_size, net.image_size], dtype=torch.float32).to(device) * -1.
        self.prev_oup = torch.ones(
            [1, net.classes], dtype=torch.float32).to(device) * -1.

    def eval(self, image):
        index_set = self.index_generator(self.prev_image, image)
        self.features[0] = image.permute(0, 2, 3, 1).reshape([-1, self.net.image_channels])

        macs_acc = 0.
        param_acc = 0.

        for i, _ in enumerate(self.net.blks):
            index = (index_set[i][index_set[i] != 0.]).to(torch.int64).to(image.device) - 1
            inp = self.net.blks[i][0](self.features[i])[index]
            macs, _ = profile(self.net.blks[i][1:], inputs=(inp,), verbose=False)
            macs_acc = macs_acc + macs
            self.features[i + 1][index] = self.net.blks[i][1:](inp)

        if (macs_acc > 0.):
            macs, _ = profile(self.net.linear, inputs=(self.features[-1],), verbose=False)
            macs_acc = macs_acc + macs
            oup = self.net.linear(self.features[-1])
            self.prev_oup = oup
        else:
            oup = self.prev_oup

        self.prev_image = image
        return oup, macs_acc