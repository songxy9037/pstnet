import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.optim import lr_scheduler

import functools
from einops import rearrange

import models
from networks import ResNet
from models.help_funcs import Transformer, TransformerDecoder, TwoLayerConv2d
from models.ps_vit import ps_vit
from cc_attention.functions import CrissCrossAttention

class PSTNet(nn.Module):
    """
    psvit + TR + bitemporal feature Differencing + CNN
    """
    def __init__(self, input_nc, output_nc, with_pos, resnet_stages_num=5,
                 token_len=4, token_trans=True,
                 enc_depth=1, dec_depth=1,
                 dim_head=64, decoder_dim_head=64,
                 tokenizer=True, if_upsample_2x=True,
                 pool_mode='max', pool_size=2,
                 backbone='resnet18',
                 decoder_softmax=True, with_decoder_pos=None,
                 with_decoder=True,
                 x_w=64,
                 x_h=64,
                 num_ps=4,
                 num_ma=1
                 ):
        super(PSTNet, self).__init__(input_nc, output_nc,backbone=backbone,
                                             resnet_stages_num=resnet_stages_num,
                                               if_upsample_2x=if_upsample_2x,
                                               )
        self.resnet = ResNet(input_nc, output_nc,
                 resnet_stages_num=5, backbone='resnet18',
                 output_sigmoid=False, if_upsample_2x=True)
        self.token_len = token_len
        self.conv_a = nn.Conv2d(32, self.token_len, kernel_size=1,
                                padding=0, bias=False)
        self.tokenizer = tokenizer
        if not self.tokenizer:
            #  if not use tokenzier，then downsample the feature map into a certain size
            self.pooling_size = pool_size
            self.pool_mode = pool_mode
            self.token_len = self.pooling_size * self.pooling_size

        self.token_trans = token_trans
        self.with_decoder = with_decoder
        dim = 32
        mlp_dim = 2*dim

        self.with_pos = with_pos
        if with_pos is 'learned':
            self.pos_embedding = nn.Parameter(torch.randn(1, self.token_len*2, 32))
        decoder_pos_size = 256//4
        self.with_decoder_pos = with_decoder_pos
        if self.with_decoder_pos == 'learned':
            self.pos_embedding_decoder =nn.Parameter(torch.randn(1, 32,
                                                                 decoder_pos_size,
                                                                 decoder_pos_size))
        self.enc_depth = enc_depth
        self.dec_depth = dec_depth
        self.dim_head = dim_head
        self.decoder_dim_head = decoder_dim_head
        self.transformer = Transformer(dim=dim, depth=self.enc_depth, heads=8,
                                       dim_head=self.dim_head,
                                       mlp_dim=mlp_dim, dropout=0)
        self.transformer_decoder = TransformerDecoder(dim=dim, depth=self.dec_depth,
                            heads=16, dim_head=self.decoder_dim_head, mlp_dim=mlp_dim, dropout=0,
                                                      softmax=decoder_softmax)

        self.psvit = ps_vit(num_iters=num_ps)
        self.x_w = x_w
        self.x_h = x_h
        self.cca = CrissCrossAttention(32)


    def ps_layer(self, x):
        b = x.shape[0]
        c = 32
        p = self.psvit(x)
        p = rearrange(p, 'b s c -> b c s')
        p = p.view([b, c, self.x_w, self.x_h])
        return p

    def semantic_tokens(self, x):
        # print('x',x.shape)
        b, c, w, h = x.shape
        spatial_attention = self.cca(x)
        spatial_attention = self.conv_a(spatial_attention)
        # # print('spa1',spatial_attention.shape)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        # # print('spa2', spatial_attention.shape)spa2 torch.Size([1, 4, 4096])
        spatial_attention = torch.softmax(spatial_attention, dim=-1)    # softmax函数对HW维进行空间注意映射
        # print('spa3', spatial_attention.shape)spa3 torch.Size([1, 4, 4096])
        x = x.view([b, c, -1]).contiguous()
        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x)  # 箭头左边表示输入张量，以逗号分割每个输入张量，箭头右边则表示输出张量

        return tokens

    def transformer_encoder(self, x):
        if self.with_pos:
            x += self.pos_embedding
        x = self.transformer(x)
        return x

    def transformer_decoder(self, x, m):
        b, c, h, w = x.shape
        if self.with_decoder_pos == 'fix':
            x = x + self.pos_embedding_decoder
        elif self.with_decoder_pos == 'learned':
            x = x + self.pos_embedding_decoder
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer_decoder(x, m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x

    def forward(self, x1, x2):

        p1 = self.ps_layer(x1)
        p2 = self.ps_layer(x2)
        # forward backbone resnet
        x1 = self.resnet(x1)    # 32,64,64
        x2 = self.resnet(x2)

         # tokenzier
        token1 = self.semantic_tokens(p1)  # b (h w) c 8,4,32
        token2 = self.semantic_tokens(p2)

        # transformer encoder
        self.tokens = torch.cat([token1, token2], dim=1)   # 8,8,32
        self.tokens = self.transformer_encoder(self.tokens)
        token1, token2 = self.tokens.chunk(2, dim=1)

        # transformer decoder
        x1 = self.transformer_decoder(x1, token1)
        x2 = self.transformer_decoder(x2, token2)

        # feature differencing
        x = torch.abs(x1 - x2)
        if not self.if_upsample_2x:
            x = self.upsamplex2(x)
        x = self.upsamplex4(x)
        # forward small cnn
        x = self.classifier(x)
        if self.output_sigmoid:
            x = self.sigmoid(x)
        return x