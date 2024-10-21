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
    
    def get_cosine_factor(m, M):
        return torch.cos(
            torch.tensor(((2*m - 1) * torch.pi) / (2*M))
        ).item()

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
        filtered_x = filtered_x.squeeze(1)
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
        self.cls_only = cls_only
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
        if cls_only:
            self.head = nn.Linear(num_tokens[-1], self.embed_dim)
        else:
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
        if self.cls_only:
            x = x.mean(1)
            x = self.final_dropout(x)
        else:
            x = self.final_dropout(x)
            x = x.transpose(1, 2)
        x = self.head(x)
        x = x.flatten(1)

        return x

from dalle2_pytorch import DiffusionPrior
from dalle2_pytorch.dalle2_pytorch import l2norm, default, exists
from dalle2_pytorch.dalle2_pytorch import RotaryEmbedding, SinusoidalPosEmb, MLP, Rearrange, repeat, rearrange, prob_mask_like, LayerNorm, RelPosBias, Attention, FeedForward
class BrainDiffusionPrior(DiffusionPrior):
    """ 
    Original Paper from Reconstructing the Mind's Eye: fMRI-to-Image with Contrastive Learning and Diffusion Priors(https://arxiv.org/abs/2305.18274)
    """
    def __init__(self, *args, **kwargs):
        voxel2clip = kwargs.pop('voxel2clip', None)
        super().__init__(*args, **kwargs)
        self.voxel2clip = voxel2clip

    @torch.no_grad()
    def p_sample(self, x, t, text_cond = None, self_cond = None, clip_denoised = True, cond_scale = 1.,
                generator=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = t, text_cond = text_cond, self_cond = self_cond, clip_denoised = clip_denoised, cond_scale = cond_scale)
        if generator is None:
            noise = torch.randn_like(x)
        else:
            noise = torch.randn_like(x)
            # noise = torch.randn(x.size(), device=x.device, dtype=x.dtype, generator=generator)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        pred = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred, x_start

    @torch.no_grad()
    def p_sample_loop(self, *args, timesteps = None, **kwargs):
        timesteps = default(timesteps, self.noise_scheduler.num_timesteps)
        assert timesteps <= self.noise_scheduler.num_timesteps
        is_ddim = timesteps < self.noise_scheduler.num_timesteps

        if not is_ddim:
            normalized_image_embed = self.p_sample_loop_ddpm(*args, **kwargs)
        else:
            normalized_image_embed = self.p_sample_loop_ddim(*args, **kwargs, timesteps = timesteps)

        # print("PS removed all image_embed_scale instances!")
        image_embed = normalized_image_embed #/ self.image_embed_scale
        return image_embed

    @torch.no_grad()
    def p_sample_loop_ddpm(self, shape, text_cond, cond_scale = 1., generator=None):
        batch, device = shape[0], self.device

        if generator is None:
            image_embed = torch.randn(shape, device = device)
        else:
            image_embed = torch.randn(shape, device = device, generator=generator)
        x_start = None # for self-conditioning

        if self.init_image_embed_l2norm:
            image_embed = l2norm(image_embed) * self.image_embed_scale

        for i in tqdm(reversed(range(0, self.noise_scheduler.num_timesteps)), desc='sampling loop time step', total=self.noise_scheduler.num_timesteps, disable=True):
            times = torch.full((batch,), i, device = device, dtype = torch.long)

            self_cond = x_start if self.net.self_cond else None
            image_embed, x_start = self.p_sample(image_embed, times, text_cond = text_cond, self_cond = self_cond, cond_scale = cond_scale, 
                                                 generator=generator)

        if self.sampling_final_clamp_l2norm and self.predict_x_start:
            image_embed = self.l2norm_clamp_embed(image_embed)

        return image_embed

    def p_losses(self, image_embed, times, text_cond, noise = None):
        noise = default(noise, lambda: torch.randn_like(image_embed))

        image_embed_noisy = self.noise_scheduler.q_sample(x_start = image_embed, t = times, noise = noise)

        self_cond = None
        if self.net.self_cond and random.random() < 0.5:
            with torch.no_grad():
                self_cond = self.net(image_embed_noisy, times, **text_cond).detach()

        pred = self.net(
            image_embed_noisy,
            times,
            self_cond = self_cond,
            text_cond_drop_prob = self.text_cond_drop_prob,
            image_cond_drop_prob = self.image_cond_drop_prob,
            **text_cond
        )

        if self.predict_x_start and self.training_clamp_l2norm:
            pred = self.l2norm_clamp_embed(pred)

        if self.predict_v:
            target = self.noise_scheduler.calculate_v(image_embed, times, noise)
        elif self.predict_x_start:
            target = image_embed
        else:
            target = noise

        loss = nn.functional.mse_loss(pred, target) # mse
        # print("1", loss)
        # loss += (1 - nn.functional.cosine_similarity(pred, target).mean())
        # print("2", (1 - nn.functional.cosine_similarity(pred, target).mean()))
        return loss, pred

    def forward(
        self,
        text = None,
        image = None,
        voxel = None,
        text_embed = None,      # allow for training on preprocessed CLIP text and image embeddings
        image_embed = None,
        text_encodings = None,  # as well as CLIP text encodings
        *args,
        **kwargs
    ):
        assert exists(text) ^ exists(text_embed) ^ exists(voxel), 'either text, text embedding, or voxel must be supplied'
        assert exists(image) ^ exists(image_embed), 'either image or image embedding must be supplied'
        assert not (self.condition_on_text_encodings and (not exists(text_encodings) and not exists(text))), 'text encodings must be present if you specified you wish to condition on it on initialization'

        if exists(voxel):
            assert exists(self.voxel2clip), 'voxel2clip must be trained if you wish to pass in voxels'
            assert not exists(text_embed), 'cannot pass in both text and voxels'
            if self.voxel2clip.use_projector:
                clip_voxels_mse, clip_voxels = self.voxel2clip(voxel)
                text_embed = clip_voxels_mse
            else:
                clip_voxels = self.voxel2clip(voxel)
                text_embed = clip_voxels_mse = clip_voxels
            # text_embed = self.voxel2clip(voxel)

        if exists(image):
            image_embed, _ = self.clip.embed_image(image)

        # calculate text conditionings, based on what is passed in

        if exists(text):
            text_embed, text_encodings = self.clip.embed_text(text)

        text_cond = dict(text_embed = text_embed)

        if self.condition_on_text_encodings:
            assert exists(text_encodings), 'text encodings must be present for diffusion prior if specified'
            text_cond = {**text_cond, 'text_encodings': text_encodings}

        # timestep conditioning from ddpm

        batch, device = image_embed.shape[0], image_embed.device
        times = self.noise_scheduler.sample_random_times(batch)
        # calculate forward loss

        loss, pred = self.p_losses(image_embed, times, text_cond = text_cond, *args, **kwargs)
        
        # undo the scaling so we can directly use it for real mse loss and reconstruction
        return loss, pred

class PriorNetwork(nn.Module):
    def __init__(
        self,
        dim,
        num_timesteps = None,
        num_time_embeds = 1,
        # num_image_embeds = 1,
        # num_brain_embeds = 1,
        num_tokens = 257,
        causal = True,
        learned_query_mode = 'none',
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.num_time_embeds = num_time_embeds
        self.continuous_embedded_time = not exists(num_timesteps)
        self.learned_query_mode = learned_query_mode

        self.to_time_embeds = nn.Sequential(
            nn.Embedding(num_timesteps, dim * num_time_embeds) if exists(num_timesteps) else nn.Sequential(SinusoidalPosEmb(dim), MLP(dim, dim * num_time_embeds)), # also offer a continuous version of timestep embeddings, with a 2 layer MLP
            Rearrange('b (n d) -> b n d', n = num_time_embeds)
        )

        if self.learned_query_mode == 'token':
            self.learned_query = nn.Parameter(torch.randn(num_tokens, dim))
        if self.learned_query_mode == 'pos_emb':
            scale = dim ** -0.5
            self.learned_query = nn.Parameter(torch.randn(num_tokens, dim) * scale)
        if self.learned_query_mode == 'all_pos_emb':
            scale = dim ** -0.5
            self.learned_query = nn.Parameter(torch.randn(num_tokens*2+1, dim) * scale)
        self.causal_transformer = FlaggedCausalTransformer(dim = dim, causal=causal, **kwargs)

        self.null_brain_embeds = nn.Parameter(torch.randn(num_tokens, dim))
        self.null_image_embed = nn.Parameter(torch.randn(num_tokens, dim))

        self.num_tokens = num_tokens
        self.self_cond = False

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 1.,
        **kwargs
    ):
        logits = self.forward(*args, **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, brain_cond_drop_prob = 1., image_cond_drop_prob = 1, **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        image_embed,
        diffusion_timesteps,
        *,
        self_cond=None,
        brain_embed=None,
        text_embed=None,
        brain_cond_drop_prob = 0.,
        text_cond_drop_prob = None,
        image_cond_drop_prob = 0.
    ):
        if text_embed is not None:
            brain_embed = text_embed
        if text_cond_drop_prob is not None:
            brain_cond_drop_prob = text_cond_drop_prob
        
        # image_embed = image_embed.view(len(image_embed),-1,16*16)
        # text_embed = text_embed.view(len(text_embed),-1,768)
        # brain_embed = brain_embed.view(len(brain_embed),-1,16*16)
        # print(*image_embed.shape)
        # print(*image_embed.shape, image_embed.device, image_embed.dtype)
        
        batch, _, dim, device, dtype = *image_embed.shape, image_embed.device, image_embed.dtype
        # num_time_embeds, num_image_embeds, num_brain_embeds = self.num_time_embeds, self.num_image_embeds, self.num_brain_embeds
        
        # classifier free guidance masks
        brain_keep_mask = prob_mask_like((batch,), 1 - brain_cond_drop_prob, device = device)
        brain_keep_mask = rearrange(brain_keep_mask, 'b -> b 1 1')

        image_keep_mask = prob_mask_like((batch,), 1 - image_cond_drop_prob, device = device)
        image_keep_mask = rearrange(image_keep_mask, 'b -> b 1 1')

        # mask out brain embeddings with null brain embeddings

        # import pdb; pdb.set_trace()
        null_brain_embeds = self.null_brain_embeds.to(brain_embed.dtype)
        brain_embed = torch.where(
            brain_keep_mask,
            brain_embed,
            null_brain_embeds[None]
        )

        # mask out image embeddings with null image embeddings
        null_image_embed = self.null_image_embed.to(image_embed.dtype)
        image_embed = torch.where(
            image_keep_mask,
            image_embed,
            null_image_embed[None]
        )

        # whether brain embedding is used for conditioning depends on whether brain encodings are available for attention (for classifier free guidance, even though it seems from the paper it was not used in the prior ddpm, as the objective is different)
        # but let's just do it right
        if self.continuous_embedded_time:
            # if continuous cast to flat, else keep int for indexing embeddings
            diffusion_timesteps = diffusion_timesteps.type(dtype)
        time_embed = self.to_time_embeds(diffusion_timesteps)

        if self.learned_query_mode == 'token':
            learned_queries = repeat(self.learned_query, 'n d -> b n d', b = batch)
        elif self.learned_query_mode == 'pos_emb':
            pos_embs = repeat(self.learned_query, 'n d -> b n d', b = batch)
            image_embed = image_embed + pos_embs
            learned_queries = torch.empty((batch, 0, dim), device=brain_embed.device)
        elif self.learned_query_mode == 'all_pos_emb':
            pos_embs = repeat(self.learned_query, 'n d -> b n d', b = batch)
            learned_queries = torch.empty((batch, 0, dim), device=brain_embed.device)
        else:
            learned_queries = torch.empty((batch, 0, dim), device=brain_embed.device)
        
        tokens = torch.cat((
            brain_embed,  # 257
            time_embed,  # 1
            image_embed,  # 257
            learned_queries  # 257
        ), dim = -2)
        if self.learned_query_mode == 'all_pos_emb':
            tokens = tokens + pos_embs

        # attend
        tokens = self.causal_transformer(tokens)

        # get learned query, which should predict the image embedding (per DDPM timestep)
        pred_image_embed = tokens[..., -self.num_tokens:, :]

        return pred_image_embed
    
class FlaggedCausalTransformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        norm_in = False,
        norm_out = True,
        attn_dropout = 0.,
        ff_dropout = 0.,
        final_proj = True,
        normformer = False,
        rotary_emb = True,
        causal=True
    ):
        super().__init__()
        self.init_norm = LayerNorm(dim) if norm_in else nn.Identity() # from latest BLOOM model and Yandex's YaLM

        self.rel_pos_bias = RelPosBias(heads = heads)

        rotary_emb = RotaryEmbedding(dim = min(32, dim_head)) if rotary_emb else None

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, causal = causal, dim_head = dim_head, heads = heads, dropout = attn_dropout, rotary_emb = rotary_emb),
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout, post_activation_norm = normformer)
            ]))

        self.norm = LayerNorm(dim, stable = True) if norm_out else nn.Identity()  # unclear in paper whether they projected after the classic layer norm for the final denoised image embedding, or just had the transformer output it directly: plan on offering both options
        self.project_out = nn.Linear(dim, dim, bias = False) if final_proj else nn.Identity()

    def forward(self, x):
        n, device = x.shape[1], x.device

        x = self.init_norm(x)

        attn_bias = self.rel_pos_bias(n, n + 1, device = device)

        for attn, ff in self.layers:
            x = attn(x, attn_bias = attn_bias) + x
            x = ff(x) + x

        out = self.norm(x)
        return self.project_out(out)