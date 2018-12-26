import torch.nn as nn
import torch
import torch.utils.data as torchdata

class SimpleNet(nn.Module):
   def __init__(self, num_classes=17,start=8,cn=10,fc_number=2048,model_name='resnet50-101'):
       super(SimpleNet, self).__init__()
       self.start = start
       self.cn = cn
       self.model_name = model_name
       self.model_number = len(model_name.split('-'))
       self.fc_number = fc_number

       basemodel1 = resnet50()
       basemodel1.conv1 = nn.Conv2d(self.cn, 64, kernel_size=7, stride=2, padding=3, bias=False)
       basemodel1.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=1)
       self.basemodel1 = nn.Sequential(*list(basemodel1.children())[:-1])

       basemodel2 = resnet101()
       basemodel2.conv1 = nn.Conv2d(self.cn, 64, kernel_size=7, stride=2, padding=3, bias=False)
       basemodel2.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=1)
       self.basemodel2 = nn.Sequential(*list(basemodel2.children())[:-1])

       self.fc = nn.Linear(in_features=fc_number * self.model_number, out_features=num_classes)

   def forward(self, input):
       out1 = self.basemodel1(input).view(-1, self.fc_number)
       out2 = self.basemodel2(input).view(-1, self.fc_number)
       outputs = torch.cat((out1,out2),1)
       output = self.fc(outputs)
       return output


