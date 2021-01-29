import torch
from torch import nn
from torch.autograd import Function
import torch.nn.functional as F
import math
import numpy as np
import random

class EcgClassifier(nn.Module):
    """Feature classifier class for MNIST -> MNIST-M experiment in ATDA."""

    def __init__(self, dropout_keep=None, num_classes=5):
        """Init classifier."""
        super(EcgClassifier, self).__init__()

        self.dropout_keep = dropout_keep

        self.classifier = nn.Sequential(
            nn.Linear(256 * 10, 256 * 5),
            # nn.Linear(256 *5, 256 * 1),
            # nn.BatchNorm1d(256 * 5),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_keep),  # 0.5
            nn.Linear(256 * 5, num_classes),

        )


    def forward(self, x):
        """Forward classifier."""
        out = self.classifier(x)
        return out

class AlexNetforEcg_Single_Model(nn.Module):
    '''input tensor size:(None,1,3,128)'''
    def __init__(self):
        super(AlexNetforEcg_Single_Model, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 5), padding=(0, 0)),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),#（N,64,1,62)

            nn.Conv2d(64, 192, kernel_size=(1, 5), padding=(0, 2)),
            # nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),#(N,192,1,30)

            nn.Conv2d(192, 256, kernel_size=(1, 5), padding=(0, 2)),#(N,_,1,30)
            # nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

        )

        self.fc = nn.Sequential(
            nn.Linear(256 * 15, 256 * 10),
            # nn.Linear(256 * 14, 256 * 10),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),#0.3
            nn.BatchNorm1d(256 * 10),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 10, 256 * 5),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # 0.5
            nn.Linear(256 * 5, 4)

        )

    def forward(self, x):
        x = self.features(x)
        # print("feature size:", x.size())
        # x = x.view(x.size(0), self.num_flat_features(x))
        # print(x.size())
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        y = self.classifier(x)
        return x, y


class ResClassifier(nn.Module):
    def __init__(self, num_classes=5, dropout_keep=0.5):
        super(ResClassifier, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(256 * 10, 256 * 5),
            # nn.BatchNorm1d(1000, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_keep)
            )
        self.fc2 = nn.Linear(256 * 5, num_classes)
        # self.extract = extract
        self.dropout_p = dropout_keep

    def forward(self, x, extract=False):
        fc1_emb = self.fc1(x)
        if self.training:
            fc1_emb.mul_(math.sqrt(1 - self.dropout_p))
        logit = self.fc2(fc1_emb)

        if extract:
            return fc1_emb, logit
        return logit

class AlexNetforEcg_DS1_to_DS2(nn.Module):
    '''input tensor size:(None,1,3,128)'''
    def __init__(self):
        super(AlexNetforEcg_DS1_to_DS2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 5), padding=(0, 0)),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(96),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),#（N,64,1,62)


            nn.Conv2d(64, 192, kernel_size=(1, 5), padding=(0, 2)),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),#(N,192,1,30)



            nn.Conv2d(192, 256, kernel_size=(1, 5), padding=(0, 2)),#(N,_,1,30)
            # nn.Conv2d(192, 256, kernel_size=(1, 3), padding=(0, 1)),
            # # nn.BatchNorm2d(384),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(inplace=True),
            # nn.Conv2d(384, 384, kernel_size=(1, 1), padding=(0, 0)),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(384, 256, kernel_size=(1, 1), padding=(0, 0)),
            # nn.ReLU(inplace=True),

            # nn.Conv2d(384, 256, kernel_size=(1, 1), padding=(0, 0)),#(N,256,1,32)

            # nn.BatchNorm2d(256),

            # nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),


        )

        self.fc = nn.Sequential(
            nn.Linear(256 * 15, 256 * 10),
            # nn.Linear(256 * 15, 256 * 5),
            # nn.Linear(256 * 14, 256 * 10),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),#0.3
            nn.BatchNorm1d(256 * 10),
        )

    def forward(self, x):
        x = self.features(x)
        # print("feature size:", x.size())
        # x = x.view(x.size(0), self.num_flat_features(x))
        # print(x.size())
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class AlexNetforEcg_DS1_to_DS2_each_patient(nn.Module):
    '''input tensor size:(None,1,3,128)'''
    def __init__(self):
        super(AlexNetforEcg_DS1_to_DS2_each_patient, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 5), padding=(0, 0)),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),#（N,64,1,62)

            nn.Conv2d(64, 192, kernel_size=(1, 5), padding=(0, 2)),
            # nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),#(N,192,1,30)

            nn.Conv2d(192, 256, kernel_size=(1, 5), padding=(0, 2)),#(N,_,1,30)
            # nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

        )

        self.fc = nn.Sequential(
            nn.Linear(256 * 15, 256 * 10),
            # nn.Linear(256 * 14, 256 * 10),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),#0.3
            nn.BatchNorm1d(256 * 10),
        )

    def forward(self, x):
        x = self.features(x)
        # print("feature size:", x.size())
        # x = x.view(x.size(0), self.num_flat_features(x))
        # print(x.size())
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class AlexNetforEcg_mitdb_to_incart(nn.Module):
    '''input tensor size:(None,1,3,128)'''
    def __init__(self):
        super(AlexNetforEcg_mitdb_to_incart, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 5), padding=(0, 0)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),#（N,64,1,62)


            nn.Conv2d(64, 192, kernel_size=(1, 5), padding=(0, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),#(N,192,1,30)

            nn.Conv2d(192, 256, kernel_size=(1, 5), padding=(0, 2)),#(N,_,1,30)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

            ####DS1->SVDB、DS1->INCARTDB实验多加了一个卷积层
            nn.Conv2d(256, 256, kernel_size=(1, 5), padding=(0, 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(1, 5), padding=(0, 2)),
            nn.ReLU(inplace=True),


        )

        self.fc = nn.Sequential(
            nn.Linear(256 * 15, 256 * 10),
            # nn.Linear(256 * 14, 256 * 10),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),#0.3
            nn.BatchNorm1d(256 * 10),
        )

    def forward(self, x):
        x = self.features(x)
        # print("feature size:", x.size())
        # x = x.view(x.size(0), self.num_flat_features(x))
        # print(x.size())
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class AlexNetforEcg_mitdb_to_svdb(nn.Module):
    '''input tensor size:(None,1,3,128)'''

    def __init__(self):
        super(AlexNetforEcg_mitdb_to_svdb, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 5), padding=(0, 0)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),  # （N,64,1,62)

            nn.Conv2d(64, 192, kernel_size=(1, 5), padding=(0, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),  # (N,192,1,30)

            nn.Conv2d(192, 256, kernel_size=(1, 5), padding=(0, 2)),  # (N,_,1,30)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

            ####DS1->SVDB、DS1->INCARTDB实验多加了一个卷积层
            nn.Conv2d(256, 256, kernel_size=(1, 5), padding=(0, 2)),
            nn.ReLU(inplace=True),


        )

        self.fc = nn.Sequential(
            nn.Linear(256 * 15, 256 * 10),
            # nn.Linear(256 * 14, 256 * 10),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),  # 0.3
            nn.BatchNorm1d(256 * 10),
        )

    def forward(self, x):
        x = self.features(x)
        # print("feature size:", x.size())
        # x = x.view(x.size(0), self.num_flat_features(x))
        # print(x.size())
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd
    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.lambd)

def grad_reverse(x, lambd=1.0):
    return GradReverse(lambd)(x)


class AdversarialLayer(torch.autograd.Function):
    def __init__(self, high_value=1.0, max_iter=10000):
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = high_value
        self.max_iter = 10000.0

    def forward(self, input):
        self.iter_num += 1
        output = input * 1.0
        return output

    def backward(self, gradOutput):

        self.coeff = np.float(
            2.0 * (self.high - self.low) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iter)) - (
                        self.high - self.low) + self.low)
        # self.coeff = 1
        return -self.coeff * gradOutput

class AdversarialNetwork(nn.Module):
  def __init__(self):
    super(AdversarialNetwork, self).__init__()
    self.ad_layer1 = nn.Linear(2560, 1024)
    self.ad_layer2 = nn.Linear(1024,1024)
    self.ad_layer3 = nn.Linear(1024, 1)
    # self.ad_layer1.weight.data.normal_(0, 0.01)
    # self.ad_layer2.weight.data.normal_(0, 0.01)
    # self.ad_layer3.weight.data.normal_(0, 0.3)
    # self.ad_layer1.bias.data.fill_(0.0)
    # self.ad_layer2.bias.data.fill_(0.0)
    # self.ad_layer3.bias.data.fill_(0.0)
    self.relu1 = nn.ReLU()
    self.relu2 = nn.ReLU()
    self.dropout1 = nn.Dropout(0.5)
    self.dropout2 = nn.Dropout(0.5)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    x = self.ad_layer1(x)
    x = self.relu1(x)
    x = self.dropout1(x)
    x = self.ad_layer2(x)
    x = self.relu2(x)
    x = self.dropout2(x)
    x = self.ad_layer3(x)
    x = self.sigmoid(x)
    return x

  def output_num(self):
    return 1

class DomainClassifier(nn.Module):
    """Feature classifier class for MNIST -> MNIST-M experiment in ATDA."""
    def __init__(self, num_classes=5):
        """Init classifier."""
        super(DomainClassifier, self).__init__()

        self.classifier_d = nn.Sequential(
            nn.Linear(256 * 10, 256 * 2),
            nn.ReLU(inplace=True),
            nn.Linear(256 * 2, 256 * 2),
            nn.ReLU(inplace=True),
            nn.Linear(256 * 2, num_classes),
        )

    def forward(self, x):
        """Forward classifier."""
        # x = grad_reverse(x)
        out = self.classifier_d(x)
        return out


class Generator(nn.Module):

    '''Dense Convolutional Networks With Focal Loss and Image Generation for
       Electrocardiogram Classification
       Convert beat to image'''

    def __init__(self):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(128, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1568),

        )

        self.upsample = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, 64, kernel_size=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 64, kernel_size=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(64, 3, kernel_size=(1, 1), padding=(0, 0)),
        )


    def forward(self, x):
        x = self.fc(x)
        # print("feature size:", x.size())
        x = x.view(-1, 32, 7, 7)
        # print(x.size())
        # x = x.view(x.size(0), -1)
        x = self.upsample(x)
        return x


class Bottleneck(nn.Module):
    '''
        the above mentioned bottleneck, including two conv layer, one's kernel size is 1×1, another's is 3×3
        in_planes可以理解成channel
        after non-linear operation, concatenate the input to the output
        DenseNet的非线性变换H采用了Bottleneck结构BN-ReLU-Conv(1×1)-BN-ReLU-Conv(3×3)，1×1的卷积用于降低维度，将channels数降
        低至4 * Growth_rate
        Bottleneck是这样一种网络，其输入输出channel差距较大，就像一个瓶颈一样，上窄下宽亦或上宽下窄，
        特征图的大小会因为最后一步的cat从N×in_planes×H×W变成N×(in_planes+growth_rate)×H×W
    '''

    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        # self.conv2 = nn.Conv2d(in_planes, growth_rate, kernel_size=3, padding=1, bias=False)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        # out = self.conv2(F.relu(self.bn1(out)))
        out = self.dropout(out)
        # input and output are concatenated here
        out = torch.cat([out, x], 1)
        return out


class Transition(nn.Module):
    '''
        transition layer is used for down sampling the feature

        when compress rate is 0.5, out_planes is a half of in_planes
    '''

    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        # use average pooling change the size of feature map here

        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=8, reduction=1, num_classes=5):
        super(DenseNet, self).__init__()
        '''
        Args:
            block: bottleneck
            nblock: a list, the elements is number of bottleneck in each denseblock
            growth_rate: channel size of bottleneck's output
            reduction: 
        '''
        self.generator = Generator()
        self.growth_rate = growth_rate

        # num_planes = 2 * growth_rate
        num_planes = 16
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=32, padding=15, bias=False)

        # a DenseBlock and a transition layer
        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0] * growth_rate
        # the channel size is superposed, mutiply by reduction to cut it down here, the reduction is also known as compress rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        # a DenseBlock and a transition layer
        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1] * growth_rate
        # the channel size is superposed, mutiply by reduction to cut it down here, the reduction is also known as compress rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        # a DenseBlock and a transition layer
        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2] * growth_rate
        # the channel size is superposed, mutiply by reduction to cut it down here, the reduction is also known as compress rate
        # out_planes = int(math.floor(num_planes * reduction))
        # self.trans3 = Transition(num_planes, out_planes)
        # num_planes = out_planes

        # only one DenseBlock
        # self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        # num_planes += nblocks[3] * growth_rate

        # the last part is a linear layer as a classifier
        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []

        # number of non-linear transformations in one DenseBlock depends on the parameter you set
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        img = self.generator(x)
        out = self.conv1(img)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        out = nn.AdaptiveAvgPool2d((1,1))(F.relu(self.bn(out)))
        out = out.view(out.size(0), -1)
        # out = self.linear(out)
        return out


def densenet():
    return DenseNet(Bottleneck, [2, 2, 2])

class AlexNet(nn.Module):
    def __init__(self,num_classes=5):
        super(AlexNet,self).__init__()

        self.generator = Generator()

        self.layer1 = nn.Sequential(  # 输入1*28*28
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 32*28*28
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32*14*14
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 64*14*14
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64*7*7
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 128*7*7
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 256*7*7
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 256*7*7
            nn.MaxPool2d(kernel_size=3, stride=2),  # 256*3*3
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 9, 256 * 4),
            # nn.Linear(256 * 14, 256 * 10),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),  # 0.3
            nn.BatchNorm1d(256 * 4),
        )
        # self.fc1 = nn.Linear(256 * 3 * 3, 1024)
        # self.fc2 = nn.Linear(1024, 512)
        # self.fc3 = nn.Linear(512, 10)


    def forward(self, x):
        x = self.generator(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(-1, 256 * 3 * 3)
        x = self.classifier(x)
        return x



