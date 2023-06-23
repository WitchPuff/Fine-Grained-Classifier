
import torch
from efficientnet_pytorch import EfficientNet
from torchvision.models import mobilenet_v3_small
from torchvision.models import shufflenet_v2_x0_5
import torch.nn as nn
import numpy as np
import torchvision.models as models

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Constants
batch_size = 50
epoch = 70
num_classes = 121
# model = 'MobileNet_V3_small'
model = 'ResNet101'

optim = 'SGD'

cleaned_trainlist = 'data/lists/cleaned_train_list.mat'
cleaned_testlist = 'data/lists/cleaned_test_list.mat'
train_mini_samples = 'data/lists/train_mini_samples.mat'
test_mini_samples = 'data/lists/test_mini_samples.mat'
net = models.resnet50(pretrained=True)
net.fc = nn.Linear(net.fc.in_features, num_classes)

# net = EfficientNet.from_pretrained('efficientnet-b0')
# net._fc = nn.Linear(net._fc.in_features, num_classes)
# net = mobilenet_v3_small(pretrained=True)
# net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, num_classes)
net.to(device)



class Cutout(object):
    """
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
        	# (x,y)表示方形补丁的中心位置
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img