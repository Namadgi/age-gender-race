import torch.nn as nn
# from torchsummary import summary

class MobileNetV1(nn.Module):
    def __init__(self, ch_in=3):
        super(MobileNetV1, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(inp, oup, 3, stride, 0, bias=False),
                nn.BatchNorm2d(oup, eps=1e-3, momentum=0.01),
                nn.ReLU6(inplace=True)
                )

        def conv_dw(inp, oup, stride, pad = False):
            p = stride % 2
            return nn.Sequential(
                # dw
                nn.ZeroPad2d((p, 1, p, 1)),
                nn.Conv2d(inp, inp, 3, stride, 0, groups=inp, bias=False),
                nn.BatchNorm2d(inp, eps=1e-3, momentum=0.01),
                nn.ReLU6(inplace=True),

                # pw
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup, eps=1e-3, momentum=0.01),
                nn.ReLU6(inplace=True),
                )

        self.model = nn.Sequential(
            conv_bn(ch_in, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2, True),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2, True),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2, True),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2, True),
            conv_dw(1024, 1024, 1),
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(1024, 256, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
        )

        self.age = nn.Sequential(
            nn.Linear(256, 100, bias=True),
            nn.Softmax(dim=1),
        )

        self.gender = nn.Sequential(
            nn.Linear(256, 1, bias=True),
            nn.Sigmoid(),
        )

        self.race = nn.Sequential(
            nn.Linear(1024, 5, bias=True),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        age_gender_x = self.fc(x)
        age = self.age(age_gender_x)
        gender = self.gender(age_gender_x)
        race = self.race(x)
        return age, gender, race

if __name__=='__main__':
    # model check
    model = MobileNetV1(ch_in=3)
    # summary(model, input_size=(3, 224, 224), device='cpu')