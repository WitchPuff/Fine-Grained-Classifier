import scipy.io
from PIL import Image
import os
from torchvision.transforms import ToTensor
import xml.etree.ElementTree as ET
# 读取 .mat 文件
trainmat = scipy.io.loadmat('data/lists/train_list.mat')
testmat = scipy.io.loadmat('data/lists/test_list.mat')

def clean(mat):
    # 查看文件中的变量
    data = list(zip([i[0] for i in mat['file_list'].squeeze()], mat['labels'].squeeze()))
    ret = {'file_list': [],
            'labels': []}
    for _, (x, y) in enumerate(data):
        img = Image.open(os.path.join('data', 'images', 'Images', x))
        if ToTensor()(img).shape[0] == 3:
            ret['file_list'].append(x)
            ret['labels'].append(y)
        else:
            print(x, ToTensor()(img).shape[0])
    scipy.io.savemat("data/lists/cleaned_train_list.mat",ret)

# # clean(trainmat)
# mat = scipy.io.loadmat('data/lists/cleaned_train_list.mat')
# # print(testmat.items())
# data = list(zip([i.strip() for i in mat['file_list']], mat['labels'].squeeze()))
# print(data)
annotation_dir = "data/annotation/Annotation/"
images_dir = "data/images/Images/"
data_dir = "data/"
categories = os.listdir(images_dir)
num_of_categories = len(categories)

# os.mkdir("data_cropped")
# for i in categories:
#     os.mkdir("data_cropped/" + i)
print(len(os.listdir("data_cropped")))
for i in os.listdir("data_cropped"):
    for file in os.listdir(annotation_dir + i):
        img = Image.open(images_dir + i + "/" + file + ".jpg")
        tree = ET.parse(annotation_dir + i + "/" + file)
        xmin = int(tree.getroot().findall('object')[0].find('bndbox').find('xmin').text)
        xmax = int(tree.getroot().findall('object')[0].find('bndbox').find('xmax').text)
        ymin = int(tree.getroot().findall('object')[0].find('bndbox').find('ymin').text)
        ymax = int(tree.getroot().findall('object')[0].find('bndbox').find('ymax').text)
        
        img_cropped = img.crop((xmin,ymin,xmax,ymax))
        img_cropped = img.convert("RGB")
        img_cropped.save("data_cropped/" + i + "/" + file + ".jpg")