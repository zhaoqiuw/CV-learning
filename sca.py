import toch
import torch.nn as nn
class PAM_Module(nn.Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self,in_dim = 512, kernel_size=3):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # x.size() 30,40,50,30
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 30,1,50,30
        x = torch.cat([avg_out, max_out], dim=1)
   
        x = self.conv1(x)  # 30,1,50,30
        return self.sigmoid(x)  # 30,1,50,30



class CAM_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim= 512):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.Softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.Softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out
    
class  SCA_Module(nn.Module):
    def __init__(self, in_dim = 512):
        super(SCA_Module, self).__init__()
        self.chanel_in = in_dim
        self.PAM = PAM_Module(in_dim=in_dim)
        self.CAM = CAM_Module(in_dim=in_dim)
        self.conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.bn  =  nn.BatchNorm2d(in_dim)
    def forward(self,x):
        x1 = self.PAM(x)
        x2 = self.CAM(x)
      
        x = torch.mul(x1,x2)
        x = self.conv(x)
        x = self.bn(x)
        return x
