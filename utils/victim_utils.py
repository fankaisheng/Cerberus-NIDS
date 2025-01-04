from utils.utils import *


class DNN3(nn.Module):
    def __init__(self, num_classes=7, input_size=1024):
        super().__init__()
        self.num_classes = num_classes
        self.fc1 = nn.Linear(input_size, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, num_classes)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        
    
    def forward(self, x):
        x = x.reshape(-1, 1024)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        if self.num_classes == 1:
            x = self.sigmoid(x)
        return x



class LeNet5(nn.Module):
    # target model  &  clone model
    def __init__(self,num_classes=7):
        super().__init__()
        self.num_classes = num_classes
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(128, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, num_classes),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x = self.model(x)
        # out = nn.functional.softmax(out, dim=1)
        if self.num_classes == 1:
            x = self.sigmoid(x)
        return x


class VGG16(nn.Module):
    def __init__(self, num_classes=7):
        super(VGG16, self).__init__()
        self.num_classes = num_classes
        self.vgg16 = torchvision.models.vgg16(pretrained=True)
        self.vgg16.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        num_features = self.vgg16.classifier[6].in_features
        self.vgg16.classifier[6] = nn.Linear(num_features, 7)
        self.sigmoid = nn.Sigmoid()

        
    def forward(self, x):
        x = self.vgg16(x)
        if self.num_classes == 1:
            x = self.sigmoid(x)
        return x
    


class ResNet18(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.num_classes = num_classes
        self.resnet18 = torchvision.models.resnet18(pretrained=True)
        # 修改第一层卷积层的输入通道数
        self.resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        # 修改全连接层的输出类别数
        in_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(in_features, num_classes)
        self.sigmoid = nn.Sigmoid()

        # self.resnet18 = self.replace_batch_norm(self.resnet18)


    def replace_batch_norm(self, model):
        for child_name, child in model.named_children():
            if isinstance(child, nn.BatchNorm2d):
                setattr(model, child_name, nn.Identity())
            elif isinstance(child, nn.Sequential) or isinstance(child, torchvision.models.resnet.BasicBlock):
                setattr(model, child_name, self.replace_batch_norm(child))
        return model
    

    def forward(self, x):
        x = self.resnet18(x)
        if self.num_classes == 1:
            x = self.sigmoid(x)
        return x




class MobileNetV2(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.num_classes = num_classes
        self.mobilenet_v2 = torchvision.models.mobilenet_v2(pretrained=True)
        # for module in self.mobilenet_v2.modules():
        #     if isinstance(module, nn.BatchNorm2d):
        #         module.track_running_stats = False
        # 修改首层卷积的输入通道数
        self.mobilenet_v2.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        # 修改全连接层的输出类别数
        in_features = self.mobilenet_v2.classifier[1].in_features
        self.mobilenet_v2.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes)
        )
        self.sigmoid = nn.Sigmoid()
        # self.replace_batch_norm(self.mobilenet_v2)


    def forward(self, x):
        x = self.mobilenet_v2(x)
        if self.num_classes == 1:
            x = self.sigmoid(x)
        return x


    def replace_batch_norm(self, model):
        for name, child in model.named_children():
            if isinstance(child, nn.BatchNorm2d):
                setattr(model, name, nn.Identity())
            else:
                self.replace_batch_norm(child)
    


class ShuffleNetV2(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.num_classes = num_classes
        self.shufflenet_v2 = torchvision.models.shufflenet_v2_x1_0(pretrained=True)
        # for module in self.shufflenet_v2.modules():
        #     if isinstance(module, nn.BatchNorm2d):
        #         module.track_running_stats = False
        # 修改首层卷积的输入通道数
        self.shufflenet_v2.conv1[0] = nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        # 修改全连接层的输出类别数
        in_features = self.shufflenet_v2.fc.in_features
        self.shufflenet_v2.fc = nn.Linear(in_features, num_classes)
        self.sigmoid = nn.Sigmoid()

        # self.replace_batch_norm(self.shufflenet_v2)

    def forward(self, x):
        x = self.shufflenet_v2(x)
        if self.num_classes == 1:
            x = self.sigmoid(x)
        return x


    def replace_batch_norm(self, model):
        for name, child in model.named_children():
            if isinstance(child, nn.BatchNorm2d):
                setattr(model, name, nn.Identity())
            else:
                self.replace_batch_norm(child)
