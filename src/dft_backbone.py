from functools import partial
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
import torch.fft
import torch.nn.functional as F


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class GlobalFilter(nn.Module):
    def __init__(self, embed_dim, N, num_filters=1):
        super().__init__()
        self.num_filters = num_filters
        self.complex_weight = nn.Parameter(torch.randn(num_filters, N//2+1, embed_dim, 2, dtype=torch.float32) * 0.02)


    def forward(self, x):

        x = x.to(torch.float32)
        x = torch.fft.rfft(x, dim=1, norm='ortho')
        power_spectrum = x**2
        all_values = []
        for filter_idx in range(self.num_filters):
            weight = torch.view_as_complex(self.complex_weight[filter_idx])
            y = x * weight * torch.cos(torch.tensor(((2*filter_idx + 1) * torch.pi) / (2 * self.num_filters)))* power_spectrum
            all_values.append(y)
        x = sum(all_values)
        x = torch.fft.irfft(x, dim=1, norm='ortho')

        return x

class Block(nn.Module):

    def __init__(self, input_size=300, embed_dim=128, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, number=844, num_filters = 1):
        super().__init__()
        self.norm1 = norm_layer(embed_dim)
        self.filter = GlobalFilter(embed_dim, number, num_filters)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(embed_dim)
        mlp_hidden_embed_dim = int(embed_dim * mlp_ratio)
        self.mlp = Mlp(in_features=embed_dim, hidden_features=mlp_hidden_embed_dim, act_layer=act_layer, drop=drop)
        self.norm3 = norm_layer(embed_dim)
        self.fremlp = FreMLP(num_tokens = number, embed_dim = embed_dim)

    def forward(self, x):
        filtered_x = self.drop_path(self.mlp(self.norm2(self.filter(self.norm1(x)))))
        filtered_x = filtered_x.unsqueeze(1)
        filtered_x = self.drop_path(self.fremlp(self.norm3(filtered_x)))
        filtered_x = filtered_x.squeeze()
        x = x + filtered_x
        return x


class PatchEmbed(nn.Module):
    """ fMRI to Patch Embedding
    """
    def __init__(self, num_tokens=768, kernel_size=10, stride=7):
        super().__init__()
        self.proj = nn.Conv1d(in_channels=1, out_channels=num_tokens, kernel_size=kernel_size, stride=stride, padding=0)
        
    def forward(self, x):
        B, L = x.shape
        x = x.view(B, 1, L)
        x = self.proj(x).flatten(2).transpose(1,2)
        return x

class FreMLP(nn.Module):
    def __init__(self, num_tokens = 257, embed_dim = 768):
        super().__init__()
        self.embed_dim = embed_dim #embed_dim
        self.hidden_size = 256 #hidden_size
        self.num_tokens = num_tokens
        self.sparsity_threshold = 0.01
        self.scale = 0.02
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_dim))
        self.r = nn.Parameter(self.scale * torch.randn(self.embed_dim, self.embed_dim))
        self.i = nn.Parameter(self.scale * torch.randn(self.embed_dim, self.embed_dim))
        self.rb = nn.Parameter(self.scale * torch.randn(self.embed_dim))
        self.ib = nn.Parameter(self.scale * torch.randn(self.embed_dim))

    def forward(self, x):

        B, nd, embed_dimension, _ = x.shape
        x = torch.fft.rfft(x, dim=2, norm='ortho') # FFT on N dimension

        o1_real = torch.zeros([B, nd, embed_dimension // 2 + 1, self.embed_dim],
                              device=x.device)
        o1_imag = torch.zeros([B, nd, embed_dimension // 2 + 1, self.embed_dim],
                              device=x.device)

        o1_real = F.relu(
            torch.einsum('bijd,dd->bijd', x.real, self.r) - \
            torch.einsum('bijd,dd->bijd', x.imag, self.i) + \
            self.rb
        )

        o1_imag = F.relu(
            torch.einsum('bijd,dd->bijd', x.imag, self.r) + \
            torch.einsum('bijd,dd->bijd', x.real, self.i) + \
            self.ib
        )

        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        y = torch.view_as_complex(y)
        x = torch.fft.irfft(y, n=self.num_tokens, dim=2, norm="ortho")

        return x


class DFTBackbone(nn.Module):
    
    def __init__(self, input_size=5917, patch_size=450, embed_dim =512, num_tokens=[512, 256, 128, 50], depth=[2,10,2,4],
                 mlp_ratio=4., drop_rate=0., drop_path_rate=0., norm_layer=None, cls_only = False, num_filters = 1):
                 

        super().__init__()
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens  # num_features for consistency with other models
        self.num_filters = num_filters
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.patch_embed = PatchEmbed(num_tokens=num_tokens[0], kernel_size=patch_size, stride=patch_size)
        self.number=((input_size-patch_size)//patch_size) + 1
        self.pos_embed = nn.Parameter(torch.zeros(1, self.number, num_tokens[0]))
        self.pos_drop = nn.Dropout(p=drop_rate)


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]  # stochastic depth decay rule
        self.tokens_down = nn.ModuleList()
        
        for i in range(len(num_tokens)-1):
            tokens_down = nn.Linear(num_tokens[i], num_tokens[i+1])
            self.tokens_down.append(tokens_down)


        self.pos_drop = nn.Dropout(p=drop_rate)
        self.blocks = nn.ModuleList()

        cur = 0
        for i in range(len(num_tokens)):

            print('using standard block')
            blk = nn.Sequential(*[
                Block(
                input_size=input_size, embed_dim=num_tokens[i], mlp_ratio=mlp_ratio[i],
                drop=drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer, number=self.number, num_filters = self.num_filters)
            for j in range(depth[i])
            ])

            self.blocks.append(blk)
            cur += depth[i]
    
        self.norm = norm_layer(num_tokens[-1])
        self.head = nn.Linear(self.number, self.embed_dim)

        self.final_dropout = nn.Identity()
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        for i in range(len(self.num_tokens)):
            x = self.blocks[i](x)
            if i != len(self.num_tokens)-1:
                x = self.tokens_down[i](x)

        x = self.norm(x)
        x = self.final_dropout(x)
        x = x.transpose(1, 2)
        x = self.head(x)
        x = x.flatten(1)

        return x
