## 预期

- [ ] Various training tricks to improve model performance（高亮的为需要做对比实验的，其余就作为基本的tricks）

  - [ ] ==[**数据增广**](https://blog.csdn.net/q6115759/article/details/130758769?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522168708032416800188570135%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=168708032416800188570135&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-130758769-null-null.142^v88^control_2,239^v2^insert_chatgpt&utm_term=%E6%95%B0%E6%8D%AE%E5%A2%9E%E5%B9%BF%E6%98%AF%E4%BB%80%E4%B9%88&spm=1018.2226.3001.4187)==
    - [x] 裁剪
    - [x] 旋转
    - [x] 灰度
    - [x] 图像扭曲
    - [x] 颜色抖动
    - [ ] 合成图片
  - [x] 学习率动态衰减：若一段时间内验证集损失没有下降，则动态调整学习率
  - [ ] **==改变网络结构==**：在ResNet中加入自注意力模块（与ViT对比可以放在一起做实验）
  - [x] 用relu激活函数代替sigmoid
  - [x] 正则化（L1正则、L2正则、dropout）
  - [x] **==注意类别不均衡数据==**（用带权重的损失函数、batch-size balanced sampling）：训练集中每个类别的图片数量可能不均衡
  - [x] **==resize图片==**：center-crop（原图短边等于网络输入size，长边等比例缩放），防止丢失信息
  - [ ] 多任务学习multi-task：多个数据集
  - [x] 早停策略（patience=5，lr_minimum=1e-5）
  - [x] 基础的参数设置统一为：![img](https://img-blog.csdnimg.cn/20190918135449555.png)

- [ ] Transfer learning: fine-tune pretrained model

  - [x] 在预训练模型上微调，直接调用Pretrained的MobileNetV3

- [ ] Attend to local regions: object localization or segmentation

  - [ ] 用SSD/Yolo进行目标检测与裁剪
  - [x] 在数据预处理时统一裁剪为相同的尺寸（resize图片：center-crop（原图短边等于网络输入size，长边等比例缩放），防止丢失信息）

- [ ] Synthetic image generation as part of data augmentation

  - [ ] 使用huggingface的stable diffusion合成图片

- [ ] ViT model backbone vs. CNN backbone: explore how to effectively use ViT

  - [ ] ViT backbone的对比

- [ ] Interpretation of the model: visualization of model predictions

  - [x] train、test、valid的loss和acc曲线
  - [ ] 可视化：
    - [x] 原始图像
    - [ ] 目标检测后裁剪图像
    - [x] 预处理后图像
    - [ ] 特征图

- [ ] Robustness of the model: adversarial examples as input, (optional) improve robustness

  - [x] 设置标签0为其他（分类时多一个类别）
  - [ ] 鲁棒性测试

- [ ] Self-supervised learning: e.g., generate a pre-trained model, and/or used as an auxiliary task 

  - [ ] 在无监督学习中，网络参数的优化是通过最大化或最小化某个定义的目标函数来实现的，而无需使用标签或人工提供的指导信号。

    一种常见的无监督学习方法是自编码器（autoencoder）。自编码器由编码器和解码器组成，其目标是通过最小化重构误差来学习数据的紧凑表示。具体来说，编码器将输入数据映射到低维潜在空间，然后解码器将潜在表示重构为输入数据。优化过程中，目标函数是输入数据与解码器输出之间的重构误差，通常使用均方差（MSE）或交叉熵作为损失函数。

    另一种无监督学习方法是生成对抗网络（GAN）。GAN由生成器和判别器组成，生成器试图生成逼真的样本，而判别器则试图区分生成的样本和真实样本。生成器和判别器通过对抗训练相互竞争，优化过程中目标是使生成器生成的样本尽可能逼真，同时使判别器难以区分真实样本和生成样本。

    除了自编码器和GAN，还有其他无监督学习方法，如聚类、降维、生成模型等。每种方法都有其特定的目标函数和优化策略，以促进模型参数的学习和调整，使其能够从未标注数据中提取有用的特征。

    在无监督学习中，网络参数的优化通常使用梯度下降或其变种方法，如随机梯度下降（SGD）或自适应优化器（如Adam）来更新参数。优化的目标是最大化或最小化定义的目标函数，以使模型能够从未标注数据中学习到有用的表示。

- [x] (Optional) few-shot learning: reduce training data to 10-20 images/class 



## 步骤

### 1、（Baseline）MobileV3

直接调用Pretrained做迁移学习，应该比较简单

参数：

1. 优化器：SGD，lr=1e-2，momentum=0.9，*weight_decay*=5e-3
2. 动态学习率：
   1. scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, *milestones*=[5, 15, 20], *gamma*=0.6)
   2. 每5个epoch，如验证集loss没有下降，则学习率减半
3. 数据预处理：随机裁剪为224x224，归一化

![Result_MobileNet_V3_small](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/202306182318018.png)

![Loss_MobileNet_V3_small](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/202306182318525.png)

![Accuracy_MobileNet_V3_small](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/202306182318546.png)



### 2、合成图片

对比：

1. 原数据集
2. 加入合成图片的数据集

### 3、目标检测&局部裁剪

对比：

1. 原数据集
2. 裁剪后的数据集

### 3、Training Tricks

- [ ] 数据增广
  - [ ] 随机裁剪和归一化
  - [ ] 旋转、灰度、扭曲、抖动、水平翻转、保持原比例拉伸裁剪、随机遮掩图片

| Original                                                     | Augmented                                                    |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![image-20230618231539111](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/202306182315182.png) | ![image-20230618231531622](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/202306182315652.png) |
| ![image-20230618231616344](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/202306182316391.png) | ![image-20230618231620194](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/202306182316227.png) |

（由于归一化后的色彩信息人眼无法识别，此时显示的图片未经过归一化）

- [ ] 均衡类别&小样本训练

  - [ ] 未均衡处理、原数据集（baseline）
  - [ ] 均衡处理、小样本训练

  参数（与baseline不同的）：

  1. few-shot learning（小样本训练）：训练集与验证集供50张/类，测试集20张/类
  2. 均衡类别数据，每一类的图片数相同

  结果：

  ![image-20230618214131079](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/202306182141192.png)

  ![image-20230618214124223](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/202306182141330.png)

  ![image-20230618213651125](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/202306182136231.png)

  对比结果：收敛速度大幅加快，训练集学习能力更强，但泛化能力不足原数据集。

- [ ] 网络结构
  - [ ] 加入自注意力模块，没效果就直接编
  - [ ] baseline的ResNet

### 3、ViT

直接调用Pretrained模型做迁移学习，采用以上实验里最优的参数和ResNet+Attention、ResNet baseline做对比

## 分工

- [x] baseline训练
- [ ] 合成图片、新数据集读取、训练
- [ ] 目标检测&局部裁剪、新数据集读取、训练
- [ ] 改变training tricks、对比训练
- [ ] ViT调库训练
- [ ] ResNet+Attention编写和训练



只上传了lists，data结构如下：

![image-20230624015912944](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/202306240159022.png)



### 运行方式

```
python train.py
```

在utils.py修改参数
