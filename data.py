import torch
import numpy as np
from torch.utils.data import SubsetRandomSampler
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import scipy.io
import os
import utils
device = 'cuda' if torch.cuda.is_available() else 'cpu'
imgs = os.path.join('data', 'images', 'Images')
lists = os.path.join('data', 'lists')

# 数据清洗，限定通道为RGB三个
def clean(mat, path):
    data = list(zip([i[0] for i in mat['file_list'].squeeze()], mat['labels'].squeeze()))
    ret = {'file_list': [],
            'labels': []}
    for _, (x, y) in enumerate(data):
        img = Image.open(os.path.join(imgs, x))
        if transforms.ToTensor()(img).shape[0] == 3:
            ret['file_list'].append(x)
            ret['labels'].append(y)
        else:
            print(x, transforms.ToTensor()(img).shape[0])
    scipy.io.savemat(path, ret)

# few-shot training
def mini_samples(mat, path, bound):
    data = list(zip([i.strip() for i in mat['file_list']], mat['labels'].squeeze()))
    ret = {'file_list': [],
            'labels': []}
    cnt = 0
    pre = 1
    skip = False
    for _, (x, y) in enumerate(data):
        # print(_)
        if cnt == bound:
            skip = True
        if pre == y and skip:
            pre = y
            continue
        if pre != y and skip:
            cnt = 0
            skip = False
        pre = y
        cnt += 1
        ret['file_list'].append(x)
        ret['labels'].append(y)
    scipy.io.savemat(path, ret)

# mat = scipy.io.loadmat(os.path.join(lists, 'cleaned_train_list.mat'))
# mini_samples(mat, os.path.join(lists, 'train_mini_samples.mat'), 25)
# mat = scipy.io.loadmat(os.path.join(lists, 'cleaned_test_list.mat'))
# mini_samples(mat, os.path.join(lists, 'test_mini_samples.mat'), 10)
# mat = scipy.io.loadmat(os.path.join(lists, 'test_mini_samples.mat'))

# print(len(mat['file_list']))


class StandfordDogs(Dataset):
    def __init__(self, file_list, transform=transforms.ToTensor()):
        mat = scipy.io.loadmat(file_list)
        self.file_list = list(zip([i.strip() for i in mat['file_list']], mat['labels'].squeeze()))
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        image_path, label = self.file_list[index]
        img = Image.open(os.path.join(imgs, image_path))
        image = self.transform(img)
        return image, int(label)




transform_train_default = transforms.Compose([
    transforms.RandomCrop(224, padding=4, pad_if_needed=True),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_train_augm = transforms.Compose([
    # transforms.Pad((0, 0, 224, 224)),  # 使用0填充剩余画布
    # transforms.Resize((224, 224)),  # 将图片调整为目标大小
    transforms.RandomCrop(224, padding=4, pad_if_needed=True),
    # 创建旋转变换
    # transforms.RandomRotation((-30, 30)),
    # 随机灰度变换
    # transforms.RandomGrayscale(p=0.5),
    # 随机扭曲
    # transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    # 随机抖动
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    # 随机水平翻转
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # utils.Cutout(n_holes=1, length=16),
])

transform_train = {'default': transform_train_default,
                    'augm': transform_train_augm}

transform_test = transforms.Compose([
    transforms.RandomCrop(224, padding=4, pad_if_needed=True),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])



def load_data(batch=utils.batch_size, valid=0.9, mini_sample=False, data_augm=False): # 读取数据，返回迭代器
    if mini_sample:
        train = utils.train_mini_samples
        test = utils.test_mini_samples
    else:
        train = utils.cleaned_trainlist
        test = utils.cleaned_testlist
    if data_augm:
        train_trans = transform_train['augm']
    else:
        train_trans = transform_train['default']
    trainset = StandfordDogs(train, train_trans)
    testset = StandfordDogs(test, transform_test)
    train_len = len(trainset)
    indices = list(range(train_len))
    split = int(np.floor(valid * train_len))
    np.random.shuffle(indices)
    train_indices, valid_indices = indices[:split], indices[split:]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch,
                                                sampler=train_sampler)#,num_workers=1,pin_memory=True)
    validloader = torch.utils.data.DataLoader(trainset, batch_size=batch,
                                                sampler=valid_sampler)#,num_workers=1,pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch,
                                                shuffle=False)#,num_workers=1,pin_memory=True)
    
    return trainloader, validloader, testloader

