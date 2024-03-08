import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops import rearrange
from einops.layers.torch import Rearrange
import math
import numpy as np
from utils.image_extract import ImageExtraction,get_default_angles,ExtractImageConvertor

class GlobalInstanceNorm(nn.Module):
    def __init__(self,in_channel,affine=False):
        super(GlobalInstanceNorm, self).__init__()
        self.norm = nn.InstanceNorm3d(in_channel,affine=True)
    def forward(self,x):
        x = einops.rearrange(x,f"(b t) c h w -> b c t h w",t=26)
        x = self.norm(x)
        x = einops.rearrange(x,f"b c t h w -> (b t) c h w",t=26)
        return x

def circle_padding(img,padding=1):
    img = F.pad(img,(padding,padding,0,0,),mode="circular")
    img = F.pad(img,(0,0,padding,padding))
    return img

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, expand_ratio=4, stride=1, alpha=1):
        super(MBConvBlock, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.expand_ratio = expand_ratio
        self.expand_features = int(self.in_channels * self.expand_ratio)
        self.alpha = alpha
        self.act = nn.GELU()
        self.norm1 = GlobalInstanceNorm(self.expand_features,affine=True)
        self.norm2 = GlobalInstanceNorm(self.expand_features,affine=True)
        self.expand_conv = nn.Conv2d(self.in_channels, self.expand_features, kernel_size=1, stride=1, padding=0, bias=False)
        self.depthwise_conv = nn.Conv2d(self.expand_features, self.expand_features, kernel_size=3, stride=1,
                                        padding=1, groups=self.expand_features)
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                nn.Conv2d(self.expand_features,int(self.expand_features*alpha),1,1,0),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(int(self.expand_features*alpha),self.expand_features,1,1,0),
                                nn.Sigmoid())
        self.pointwise_conv = nn.Conv2d(self.expand_features, self.in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x,*args):
        x = self.act(self.norm1(self.expand_conv(x)))
        x = self.act(self.norm2(self.depthwise_conv(x)))
        x = x * self.se(x)
        x = self.pointwise_conv(x)
        return x


class GaussianBlur(nn.Module):
    def __init__(self):
        super().__init__()
        self.ext = ImageExtraction(image_shape=16,erp_image_shape=(64,32))
        self.reverse = ExtractImageConvertor(image_shape=16,erp_image_shape=(64,32))
        self.update_rate = nn.Parameter(torch.tensor([0],dtype=torch.float32,requires_grad=True))
    def forward(self,x):
        with torch.no_grad():
            x = self.reverse(x)
            x = self.ext(x)
        x = torch.sigmoid(self.update_rate)*x
        return x

class CrossMultiHeadAttention(nn.Module):
    def __init__(self,in_channel,n_head):
        super().__init__()
        self.init_linear = nn.Conv2d(in_channel,in_channel*3,1,1,0)
        self.out_linear = nn.Conv2d(in_channel,in_channel,1,1,0)
        angles = np.array(get_default_angles())
        angle_delta = np.abs(angles.reshape(26, 1, 2) - angles.reshape(1, 26, 2))
        angle_delta = np.sum(np.minimum(angle_delta, 360 - angle_delta), axis=2)
        angle_bool = angle_delta == 45
        angle_bool[0,-8::2] = True
        angle_bool[-8:,0] = True
        angle_bool[1,2:10:2] = True
        angle_bool[2:10,1] = True
        self.access_array = np.array([np.argwhere(angle_bool[i]) for i in range(26)]).reshape(26,4)
        self.n_head = n_head

    def forward(self,x,*args):
        b,c,h,w = x.shape
        x = self.init_linear(x)
        q,k,v = torch.chunk(einops.rearrange(x,"(b t) c h w->b t c h w",t=26),chunks=3,dim=2)
        q = einops.rearrange(q,"b t (c head) h w->(b t head) (h w) c",head=self.n_head)
        k = einops.rearrange(k[:,self.access_array],"b t1 t2 (c head) h w->(b t1 head) c (t2 h w)",head=self.n_head)
        v = einops.rearrange(v[:,self.access_array],"b t1 t2 (c head) h w->(b t1 head) (t2 h w) c",head=self.n_head)
        qk = torch.softmax(torch.bmm(q,k)/math.sqrt(q.shape[-1]),dim=-1)
        qkv = torch.bmm(qk,v)
        x = self.out_linear(einops.rearrange(qkv,"(b  head) (h w) c->b (c head) h w",h=h,w=w,head=self.n_head))
        return x

class CrossAttention(nn.Module):
    def __init__(self,in_channel,n_head):
        super(CrossAttention, self).__init__()
        self.mha = nn.MultiheadAttention(in_channel,n_head,batch_first=True)
        angles = [(0, 90), (0, -90)]
        for i in range(1, 4):
            angles += [(0 + 45 * j, -90 + 45 * i) for j in range(8)]
        angles = np.array(angles)
        angle_delta = np.abs(angles.reshape(26, 1, 2) - angles.reshape(1, 26, 2))
        angle_delta = np.sum(np.minimum(angle_delta,360-angle_delta),axis=2)
        angle_bool = angle_delta==45
        angle_bool[0,-8::2] = True
        angle_bool[-8::2,0] = True
        angle_bool[1,2:10:2] = True
        angle_bool[2:10:2,1] = True
        self.angle_select = [list(np.argwhere(angle_bool[i]).reshape(-1)) for i in range(26)]

    def forward(self,x,*args):
        b,c,h,w = x.shape
        x = einops.rearrange(x,f"(b t) c h w -> b t (h w) c",t=26,h=h,w=w)
        outs = []
        for i in range(26):
            query = x[:,i]
            kv = x[:,self.angle_select[i]].reshape(b//26,-1,c)
            out = self.mha(query,kv,kv)[0]
            outs.append(out)
        x = torch.stack(outs,dim=1)
        x = einops.rearrange(x,f"b t (h w) c -> (b t) c h w",t=26,h=h,w=w)
        return x

class CrossAttentionHeightSplit(nn.Module):
    def __init__(self, in_channel, n_head):
        super(CrossAttentionHeightSplit, self).__init__()
        self.mha = nn.ModuleList([nn.MultiheadAttention(in_channel, n_head, batch_first=True) for i in range(5)])
        # angles = [(0, 90), (0, -90)]
        # for i in range(1, 4):
        #     angles += [(0 + 45 * j, -90 + 45 * i) for j in range(8)]
        angles = np.array(get_default_angles())
        angle_delta = np.abs(angles.reshape(26, 1, 2) - angles.reshape(1, 26, 2))
        angle_delta = np.sum(np.minimum(angle_delta, 360 - angle_delta), axis=2)
        angle_bool = angle_delta == 45
        angle_bool[0, -8::2] = True
        angle_bool[-8::2, 0] = True
        angle_bool[1, 2:10:2] = True
        angle_bool[2:10:2, 1] = True
        self.angle_select = [list(np.argwhere(angle_bool[i]).reshape(-1)) for i in range(26)]

    def get_mha(self,i):
        if i<2:
            return self.mha[i]
        else:
            index = (i-2)//8 + 2
            return self.mha[index]

    def forward(self, x, *args):
        b, c, h, w = x.shape
        x = einops.rearrange(x, f"(b t) c h w -> b t (h w) c", t=26, h=h, w=w)
        outs = []
        for i in range(26):
            query = x[:, i]
            kv = x[:, self.angle_select[i]].reshape(b // 26, -1, c)
            out = self.get_mha(i)(query, kv, kv)[0]
            outs.append(out)
        x = torch.stack(outs, dim=1)
        x = einops.rearrange(x, f"b t (h w) c -> (b t) c h w", t=26, h=h, w=w)
        return x

class CrossAttention2(nn.Module):
    def __init__(self,in_channel,n_head):
        super(CrossAttention2, self).__init__()
        self.mha = nn.MultiheadAttention(in_channel,n_head,batch_first=True)
        angles = [(0, 90), (0, -90)]
        for i in range(1, 4):
            angles += [(0 + 45 * j, -90 + 45 * i) for j in range(8)]
        angles = np.array(angles)
        angle_delta = np.abs(angles.reshape(26, 1, 2) - angles.reshape(1, 26, 2))
        angle_delta = np.sum(np.minimum(angle_delta,360-angle_delta),axis=2)
        angle_bool = angle_delta<=45
        angle_bool[0,-8::2] = True
        angle_bool[-8::2,0] = True
        angle_bool[1,2:10:2] = True
        angle_bool[2:10:2,1] = True
        self.angle_select = [list(np.argwhere(angle_bool[i]).reshape(-1)) for i in range(26)]

    def forward(self,x,*args):
        b,c,h,w = x.shape
        x = einops.rearrange(x,f"(b t) c h w -> b t (h w) c",t=26,h=h,w=w)
        outs = []
        for i in range(26):
            query = x[:,i]
            kv = x[:,self.angle_select[i]].reshape(b//26,-1,c)
            out = self.mha(query,kv,kv)[0]
            outs.append(out)
        x = torch.stack(outs,dim=1)
        x = einops.rearrange(x,f"b t (h w) c -> (b t) c h w",t=26,h=h,w=w)
        return x

class SquareAttention(nn.Module):
    def __init__(self,in_channel,n_head):
        super(SquareAttention, self).__init__()
        self.mha = nn.MultiheadAttention(in_channel,n_head,batch_first=True)
        angles = [(0, 90), (0, -90)]
        for i in range(1, 4):
            angles += [(0 + 45 * j, -90 + 45 * i) for j in range(8)]
        angles = np.array(angles)
        angle_delta = np.abs(angles.reshape(26, 1, 2) - angles.reshape(1, 26, 2))
        angle_delta = np.minimum(angle_delta,360-angle_delta)
        angle_bool = angle_delta.max(axis=-1)==45
        angle_bool[0,-8:] = True
        angle_bool[-8:,0] = True
        angle_bool[1,2:10] = True
        angle_bool[2:10,1] = True
        self.angle_select = [list(np.argwhere(angle_bool[i]).reshape(-1)) for i in range(26)]

    def forward(self,x,*args):
        b,c,h,w = x.shape
        x = einops.rearrange(x,f"(b t) c h w -> b t (h w) c",t=26,h=h,w=w)
        outs = []
        for i in range(26):
            query = x[:,i]
            kv = x[:,self.angle_select[i]].reshape(b//26,-1,c)
            out = self.mha(query,kv,kv)[0]
            outs.append(out)
        x = torch.stack(outs,dim=1)
        x = einops.rearrange(x,f"b t (h w) c -> (b t) c h w",t=26,h=h,w=w)
        return x

class AllAttention(nn.Module):
    def __init__(self,in_channel,n_head):
        super(AllAttention, self).__init__()
        self.mha = nn.MultiheadAttention(in_channel,n_head,batch_first=True)

    def forward(self,x,*args):
        b,c,h,w = x.shape
        x = einops.rearrange(x,f"(b t) c h w -> b (t h w) c",t=26,h=h,w=w)
        x = self.mha(x,x,x)[0]
        x = einops.rearrange(x,f"b (t h w) c -> (b t) c h w",t=26,h=h,w=w)
        return x

class BlockAttention(nn.Module):
    def __init__(self,in_channel,n_head):
        super().__init__()
        self.mha = nn.MultiheadAttention(in_channel,n_head,batch_first=True)

    def forward(self, x, *args):
        b, c, h, w = x.shape
        x = einops.rearrange(x, f"b c h w -> b (h w) c", h=h, w=w)
        x = self.mha(x, x, x)[0]
        x = einops.rearrange(x, f"b (h w) c -> b c h w", h=h, w=w)
        return x

class GridAttention(nn.Module):
    def __init__(self,in_channel,n_head):
        super().__init__()
        self.mha = nn.MultiheadAttention(in_channel,n_head,batch_first=True)

    def forward(self, x, *args):
        b, c, h, w = x.shape
        x = einops.rearrange(x, f"(b t) c h w -> (b h w) t c", t=26, h=h, w=w)
        x = self.mha(x, x, x)[0]
        x = einops.rearrange(x, f"(b h w) t c -> (b t) c h w", t=26, h=h, w=w)
        return x


class MLP(nn.Module):
    def __init__(self,in_channel,hidden_channel,act=nn.ReLU(inplace=True)):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channel,hidden_channel,1,1,0),
            act,
            nn.Conv2d(hidden_channel,in_channel,1,1,0)
        )
    def forward(self,x,*args):
        return self.mlp(x)


class BlockModule(nn.Module):
    def __init__(self,in_channel,modules,):
        super().__init__()
        self.norms = nn.ModuleList([GlobalInstanceNorm(in_channel) for i in range(len(modules))])
        self.module_list = nn.ModuleList(modules) if type(modules)==list else modules

    def forward(self,x,*args):
        for i in range(len(self.norms)):
            if type(self.module_list[i]) == GaussianBlur:
                x = x + self.module_list[i](x, *args)
            else:
                x = x + self.module_list[i](self.norms[i](x),*args)
        return x

class MultiAxisAttention(nn.Module):
    def __init__(self,d_model,n_head,patch_size=8):
        super().__init__()
        self.linear1 = nn.Linear(d_model,d_model//2*3)
        self.linear2 = nn.Linear(d_model,d_model//2*3)
        self.n_head = n_head
        self.patch_size = patch_size
        self.out_linear = nn.Conv2d(d_model,d_model,1,1,0)

    def forward(self,x):
        x = rearrange(x,"(b t) c h w->b t c h w",t=26)
        b,_,c,h,w = x.shape
        x1 = rearrange(x,"b t c h w->(b t) (h w) c")
        x2 = rearrange(x,"b t c h w->(b h w) (t) c")

        x1 = self.linear1(x1)
        x2 = self.linear2(x2)


        x1 = torch.chunk(rearrange(x1,"b t (h c)->(b h) t c",h=self.n_head//2),3,dim=2)
        x2 = torch.chunk(rearrange(x2,"b t (h c)->(b h) t c",h=self.n_head//2),3,dim=2)

        x1 = F.scaled_dot_product_attention(x1[0],x1[1],x1[2])
        x2 = F.scaled_dot_product_attention(x2[0],x2[1],x2[2])


        x1 = rearrange(x1,"(b h) t c->b t (h c)",h=self.n_head//2)
        x2 = rearrange(x2,"(b h) t c->b t (h c)",h=self.n_head//2)

        x1 = rearrange(x1,"(b t) (h w) c->(b t) c h w",b=b,t=26,h=h)
        x2 = rearrange(x2,"(b h w) (t) c->(b t) c h w",b=b,t=26,h=h)

        x = torch.cat([x1,x2],dim=1)
        x = self.out_linear(x)

        return x

def create_layer(module_settings,in_channel,num_heads=8):
    modules = []
    for i in module_settings:
        if i=="MBConv":
            modules.append(MBConvBlock(in_channel))
        elif i=="CrossAttention":
            modules.append(CrossAttention(in_channel,num_heads))
        elif i=="CrossAttentionHeightSplit":
            modules.append(CrossAttentionHeightSplit(in_channel,num_heads))
        elif i=="CrossMultiHeadAttention":
            modules.append(CrossMultiHeadAttention(in_channel,num_heads))
        elif i=="CrossAttention2":
            modules.append(CrossAttention2(in_channel,num_heads))
        elif i=="SquareAttention":
            modules.append(SquareAttention(in_channel,num_heads))
        elif i=="AllAttention":
            modules.append(AllAttention(in_channel,num_heads))
        elif i=="BlockAttention":
            modules.append(BlockAttention(in_channel,num_heads))
        elif i=="GridAttention":
            modules.append(GridAttention(in_channel,num_heads))
        elif i=="MultiAxisAttention":
            modules.append(MultiAxisAttention(in_channel,num_heads))
        elif i=="MLP":
            modules.append(MLP(in_channel,in_channel*4))
        else:
            raise NameError(f"module {i} does not implemented")
    return BlockModule(in_channel,modules)

class LayerModule(nn.Module):
    def __init__(self,module_settings,in_channel,block_repeat=2,num_heads=8):
        super().__init__()
        blocks = []
        for i in range(block_repeat):
            blocks.append(create_layer(module_settings,in_channel,num_heads))
        self.blocks = nn.ModuleList(blocks)
    def forward(self,x,*args):
        for block in self.blocks:
            x = block(x,*args)
        return x

def get_model_construction_settings(const):
    if const=="cross":
        return ["MBConv","CrossAttention","MLP"]
    elif const=="maxvit":
        return ["MBConv","BlockAttention","MLP","GridAttention","MLP"]
    elif const=="block_cross":
        return ["MBConv","BlockAttention","MLP","CrossAttention","MLP"]
    elif const=="maxvit_cross":
        return ["MBConv","BlockAttention","MLP","CrossAttention","MLP","GridAttention","MLP"]
    elif const=="multiaxis":
        return ["MultiAxisAttention","MLP"]
    else:
        return const.split("_")

class SecondStageModel(nn.Module):
    def __init__(self,in_channel,num_heads=8,
                 layer_const=["cross"],block_nums=[2,],
                 output_channel=1024,view_angle=60):
        super(SecondStageModel, self).__init__()
        self.emb = nn.Sequential(nn.Embedding(1025,in_channel),Rearrange("b h w c->b c h w"))
        assert len(layer_const) == len(block_nums)
        assert len(layer_const)%2==1

        self.half_layer_num = len(layer_const)//2
        layers = []
        for layer,block_nums in zip(layer_const,block_nums):
            layers.append(LayerModule(get_model_construction_settings(layer),in_channel,
                                      block_repeat=block_nums,num_heads=num_heads))
        self.layers = nn.ModuleList(layers)
        self.last_conv = nn.Conv2d(in_channel,output_channel,1,1,0)
        self.pos_enc = PositionalEncoding(in_channel)
        self.cond_conv = nn.Sequential(
            nn.Conv2d(7,in_channel//2,5,2,2),
            nn.GroupNorm(1,in_channel//2),
            nn.GELU(),
            nn.Conv2d(in_channel//2, in_channel, 3, 2, 1)
        )
        self.extractor = ImageExtraction(image_shape=64,view_angle=view_angle)

    def forward(self,x,cond,mask,*args):
        if len(x.shape)==2:
            x = x.reshape(-1,16,16)
        with torch.no_grad():
            cond = torch.cat([cond,mask],axis=1)
            cond = self.extractor(cond)
        cond = self.cond_conv(cond)
        x = self.emb(x) + cond + self.pos_enc(x)
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        x = self.last_conv(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, embed_size):
        super(PositionalEncoding, self).__init__()
        self.embed_size = embed_size
        self.pos_encoding = self.generate_positional_encoding()
        self.linear = nn.Conv2d(16,embed_size,1,1,0)

    def generate_positional_encoding(self):
        y,x = torch.meshgrid(torch.arange(256) / 256, torch.arange(128) / 128)
        ls = []
        for i in range(1,5):
            ls.append(torch.sin((2**i)*torch.pi*x))
            ls.append(torch.cos((2**i)*torch.pi*x))
        for i in range(4):
            ls.append(torch.sin((2**i)*torch.pi*y))
            ls.append(torch.cos((2**i)*torch.pi*y))
        stacks = torch.stack(ls,dim=0).unsqueeze(0)
        return ImageExtraction(image_shape=16,erp_image_shape=(256,128))(stacks)


    def forward(self, x):
        if x.device != self.pos_encoding.device:
            self.pos_encoding = self.pos_encoding.to(x.device)
        posenc = self.linear(self.pos_encoding)
        return posenc.repeat(x.shape[0]//26,1,1,1)


if __name__=="__main__":
    # mod =UNet(128,mask_data=True,layer_const=["crossfast"],block_nums=[2,]).cuda()
    mod = SecondStageModel(128,mask_data=True,layer_const=["multiaxis"],block_nums=[2,]).cuda()
    inp = torch.randint(0, 1024, (1*26,16,16)).cuda()
    inp_cond = torch.randn((1,3,256,512)).cuda()
    inp_cond2 = torch.randn((1,4,256,512)).cuda()
    # inp_cond = torch.randint(0,1023,(1,256,512)).cuda()
    from torchinfo import  summary
    summary(mod,input_data=[inp,inp_cond,inp_cond2],depth=7)
