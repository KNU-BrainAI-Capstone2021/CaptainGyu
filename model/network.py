import torch
import torch.nn as nn

#Code Operation Block

class ZeroPaddingConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(ZeroPaddingConv, self).__init__()
        padding = ((kernel_size - 1) * dilation + 1) // 2
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
                              padding_mode=padding_mode)
        
    def forward(self, x):
        y = self.conv(x)
        
        return y
    
class DownScale(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=2, dilation=1):
        super(DownScale, self).__init__()
        self.out_ch = out_ch
        self.conv = ZeroPaddingConv(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, 
                                    stride=stride, dilation=dilation)
        self.leakyRelu = nn.LeakyReLU(0.1)
        
    def get_out_ch(self):
        return self.out_ch
        
    def forward(self, x):
        x = self.conv(x)
        y = self.leakyRelu(x)
        return y

class DownScale_Block(nn.Module):
    def __init__(self, in_ch, base_ch, n_downscales=4, kernel_size=5, stride=2):
        super(DownScale_Block, self).__init__()
        self.downs = []
        
        last_ch = in_ch
        for i in range(n_downscales):
            cur_ch = base_ch*(min(2**i, 8))
            self.downs.append(DownScale(last_ch, cur_ch, kernel_size=kernel_size, stride=stride))
            last_ch = self.downs[-1].get_out_ch()
            
    def forward(self, x):
        for down in self.downs:
            x = down(x)
            
        return x    
    
class UpScale_Block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super(UpScale_Block, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        
        self.conv = ZeroPaddingConv(in_ch, out_ch*4, kernel_size=kernel_size)
        self.leakyRelu = nn.LeakyReLU(0.1)
        self.depth2space = nn.PixelShuffle(upscale_factor=2) #(N, C*r*r, H, W) => (N, C, H*r, W*r)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.leakyRelu(x)
        y = self.depth2space(x)
        
        return y 
    
class Residual_Block(nn.Module):
    def __init__(self, ch, kernel_size=3):
        super(Residual_Block, self).__init__()
        self.ch = ch
        self.conv1 = ZeroPaddingConv(ch, ch, kernel_size=kernel_size)
        self.conv2 = ZeroPaddingConv(ch, ch, kernel_size=kernel_size)
        self.leakyRelu = nn.LeakyReLU(0.2)
    
    def forward(self, x):
        x_f = self.conv1(x)
        x_f = self.leakyRelu(x_f)
        x_f = self.conv2(x_f)
        y = self.leakyRelu(x + x_f)
        
        return y
    
#Encoder Inter Decoder

class Encoder(nn.Module):
    def __init__(self, in_ch, base_ch):
        super(Encoder, self).__init__()
        self.in_ch = in_ch
        self.base_ch = base_ch
        
        self.downsScale = DownScale_Block(in_ch, base_ch)
        self.flatten = nn.Flatten()
        
    def get_out_res(self, res):
        return res // (2**4)
    
    def get_out_ch(self):
        return self.base_ch * 8
    
    def forward(self, x):
        x = self.downsScale(x)
        y = self.flatten(x)
        
        return y
    
class Inter(nn.Module):
    def __init__(self, in_ch, middle_ch, out_ch, res):
        super(Inter, self).__init__()
        self.dense_res = res // 16
        self.out_ch = out_ch
        self.fc1 = nn.Linear(in_ch, middle_ch, bias=True)
        self.fc2 = nn.Linear(middle_ch, self.dense_res * self.dense_res * out_ch, bias=True)
        self.upscale = UpScale_Block(out_ch, out_ch)
    
    def get_out_res(self): 
        return self.dense_res*2
    
    def get_out_ch(self):
        return self.out_ch
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = torch.reshape(x, (-1, self.out_ch, self.dense_res, self.dense_res))
        y = self.upscale(x)
        
        return y    
    
class Decoder(nn.Module):
    def __init__(self, in_ch, d_ch, d_mask_ch):
        super(Decoder, self).__init__()
        self.up_d1 = UpScale_Block(in_ch, d_ch*8)
        self.up_d2 = UpScale_Block(d_ch*8, d_ch*4)
        self.up_d3 = UpScale_Block(d_ch*4, d_ch*2)
        self.res1 = Residual_Block(d_ch*8)
        self.res2 = Residual_Block(d_ch*4)
        self.res3 = Residual_Block(d_ch*2)
        self.conv = ZeroPaddingConv(d_ch*2, 3, kernel_size=1)
        
        self.up1 = UpScale_Block(in_ch, d_mask_ch*8)
        self.up2 = UpScale_Block(d_mask_ch*8, d_mask_ch*4)
        self.up3 = UpScale_Block(d_mask_ch*4, d_mask_ch*2)
        self.conv_up = ZeroPaddingConv(d_mask_ch*2, 1, kernel_size=1)
        
        self.sig = nn.Sigmoid()
        
    def forward(self, x):
        x_im = self.up_d1(x)
        x_im = self.res1(x_im)
        x_im = self.up_d2(x_im)
        x_im = self.res2(x_im)
        x_im = self.up_d3(x_im)
        x_im = self.res3(x_im)
        x_im = self.conv(x_im)
        x_im = self.sig(x_im)
        
        
        x_seg = self.up1(x)
        x_seg = self.up2(x_seg)
        x_seg = self.up3(x_seg)
        x_seg = self.conv_up(x_seg)
        x_seg = self.sig(x_seg)
        
        
        return x_im, x_seg
    
# Network & Partition for training
    
class DF(nn.Module):
    def __init__(self, input_ch, enc_base_ch, inter_middle, inter_out_ch, res, d_ch, d_mask_ch):
        super(DF, self).__init__()
        self.enc = Encoder(input_ch, enc_base_ch)
        enc_out_ch = self.enc.get_out_ch() * (res // 16) * (res // 16)
        
        self.inter = Inter(enc_out_ch, inter_middle, inter_out_ch, res)  
        
        self.src_dec = Decoder(inter_out_ch, d_ch, d_mask_ch)
        self.dst_dec = Decoder(inter_out_ch, d_ch, d_mask_ch)
        
    def conversion(self, x):
        enc_out = self.enc(x)
        inter_out = self.inter(enc_out)
        
        output_im, output_seg = self.dst_dec(inter_out)
        
        return output_im, output_seg
    
    def forward_src(self, x):
        enc_out = self.enc(x)
        inter_out = self.inter(enc_out)
        
        src_dec_out_im, src_dec_out_seg = self.src_dec(inter_out)
        
        return src_dec_out_im, src_dec_out_seg
    
    def forward_dst(self, x):
        enc_out = self.enc(x)
        inter_out = self.inter(enc_out)
        
        dst_dec_out_im, dst_dec_out_seg = self.dst_dec(inter_out)
        
        return dst_dec_out_im, dst_dec_out_seg    
    
class DF_Src(nn.Module):
    def __init__(self, DF_network):
        super(DF_Src, self).__init__()
        self.enc = DF_network.enc
        self.inter = DF_network.inter
        self.src_dec = DF_network.src_dec
        
    def forward(self, x):
        enc_out = self.enc(x)
        inter_out = self.inter(enc_out)
        
        src_dec_out_im, src_dec_out_seg = self.src_dec(inter_out)
        
        return src_dec_out_im, src_dec_out_seg
        
class DF_Dst(nn.Module):
    def __init__(self, DF_network):
        super(DF_Dst, self).__init__()
        self.enc = DF_network.enc
        self.inter = DF_network.inter
        self.dst_dec = DF_network.dst_dec
        
    def forward(self, x):
        enc_out = self.enc(x)
        inter_out = self.inter(enc_out)
        
        dst_dec_out_im, dst_dec_out_seg = self.dst_dec(inter_out)
        
        return dst_dec_out_im, dst_dec_out_seg