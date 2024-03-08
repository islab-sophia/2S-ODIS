import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def circle_padding(img,padding=1):
    img = F.pad(img,(padding,padding,0,0,),mode="circular")
    img = F.pad(img,(0,0,padding,padding))
    return img

class CircleConv(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding=0,groups=groups)
        self.padding = padding
    def forward(self,x):
        x = circle_padding(x,self.padding)
        x = self.conv(x)
        return x


class MBConvBlock(nn.Module):
    def __init__(self, in_channels, expand_ratio=4, stride=1, alpha=1.0):
        super(MBConvBlock, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.expand_ratio = expand_ratio
        self.alpha = alpha

        self.expand_features = int(self.in_channels * self.expand_ratio)
        self.expand_conv = nn.Conv2d(self.in_channels, self.expand_features, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm = nn.GroupNorm(1,self.expand_features)

        # Depthwise convolution phase
        self.depthwise_conv = CircleConv(self.expand_features, self.expand_features, kernel_size=3,
                                         stride=self.stride, padding=1, groups=self.expand_features)

        # Squeeze and Excitation phase
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                nn.Conv2d(self.expand_features,int(self.expand_features*alpha),1,1,0),
                                nn.GELU(),
                                nn.Conv2d(int(self.expand_features*alpha),self.expand_features,1,1,0),
                                nn.Sigmoid())

        # Pointwise convolution phase
        self.pointwise_conv = nn.Conv2d(self.expand_features, self.in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        # Expansion phase
        x = self.norm(self.expand_conv(x))
        # Depthwise convolution phase
        x = F.gelu(self.depthwise_conv(x))
        # Squeeze and Excitation phase
        x = x * self.se(x)
        # Pointwise convolution phase
        x = self.pointwise_conv(x)
        # Skip connection and addition
        x = x + identity
        return x

class DeepConditionNetwork(nn.Module):
    def __init__(self,d_model):
        super().__init__()
        self.conv1 = CircleConv(4,d_model//8,5,2,2)
        self.conv2 = CircleConv(d_model//8,d_model//4,3,2,1)
        self.conv3 = CircleConv(d_model//4,d_model//2,3,2,1)
        self.conv4 = CircleConv(d_model//2,d_model,3,2,1)

        self.layer1 = nn.Sequential(*[MBConvBlock(d_model//4) for i in range(4)])
        self.layer2 = nn.Sequential(*[MBConvBlock(d_model//2) for i in range(4)])

        self.norm1 = nn.GroupNorm(1,d_model//8)
        self.norm2 = nn.GroupNorm(1,d_model//4)
        self.norm3 = nn.GroupNorm(1,d_model//2)

    def forward(self,x):
        x1 = F.gelu(self.norm1(self.conv1(x)))
        x2 = self.norm2(self.conv2(x1))
        x2 = self.layer1(x2)
        x3 = self.norm3(self.conv3(x2))
        x3 = self.layer2(x3)
        x4 = self.conv4(x3)
        return x4

class MultiAxisTransformerLayer(nn.Module):
    def __init__(self,d_model,n_head,drop_path_rate=0.1,patch_size=4):
        super().__init__()
        self.mbconv = MBConvBlock(d_model)
        self.layer1 = nn.TransformerEncoderLayer(d_model, n_head,dim_feedforward=d_model*4, batch_first=True,norm_first=True)
        self.layer2 = nn.TransformerEncoderLayer(d_model, n_head,dim_feedforward=d_model*4, batch_first=True,norm_first=True)
        self.drop_path_rate = drop_path_rate
        self.patch_size = patch_size

    def forward(self,x):
        x = rearrange(x, "b (h w) c->b c h w",h=16)
        b,c,h,w = x.shape
        x = self.mbconv(x)
        x = rearrange(x,"b c (h1 h2) (w1 w2)->(b h1 w1) (h2 w2) c",
                      h1=h//self.patch_size,h2=self.patch_size,w1=w//self.patch_size,w2=self.patch_size)
        x = self.layer1(x)
        x = rearrange(x,"(b h1 w1) (h2 w2) c->(b h2 w2) (h1 w1) c",
                      h1=h//self.patch_size,h2=self.patch_size,w1=w//self.patch_size,w2=self.patch_size)
        x = self.layer2(x)
        x = rearrange(x,"(b h2 w2) (h1 w1) c->b (h1 h2 w1 w2) c",
                      h1=h//self.patch_size,h2=self.patch_size,w1=w//self.patch_size,w2=self.patch_size)
        return x

class FirstStageModel(nn.Module):
    def __init__(self, vocab_size, d_model,layer=8,seq_len=512,deep_conv=False):
        super(FirstStageModel, self).__init__()
        self.pos_enc = nn.Parameter(torch.randn(1,seq_len,d_model))
        self.embedding = nn.Embedding(vocab_size+1, d_model)
        self.trans = nn.ModuleList([MultiAxisTransformerLayer(d_model,8) for i in range(layer)])
        if deep_conv:
            self.cond_conv =DeepConditionNetwork(d_model)
        else:
            self.cond_conv = nn.Conv2d(4, d_model, 16,16)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x,cond,*args):
        x = self.embedding(x)
        cond = rearrange(self.cond_conv(cond),"b c h w->b (h w) c")
        x = x + self.pos_enc + cond
        for t in self.trans:
            x = t(x)
        output = self.fc(x)
        return output

if __name__=="__main__":
    mod = FirstStageModel(1024,128,1,deep_conv=True).cuda()
    inp = torch.randint(0, 1024, (1,512)).cuda()
    inp_c = torch.randn((1,4,256,512)).cuda()
    out = mod(inp,inp_c)
    print(out.shape)