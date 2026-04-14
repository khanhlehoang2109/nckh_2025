# %%writefile /kaggle/working/mdm/exp_code/model/sam_stunet.py

from abc import abstractmethod
from sentence_transformers import SentenceTransformer, models
import math 
import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from diffusion.fp16_util import convert_module_to_f16, convert_module_to_f32
from diffusion.nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
from model.qna import FusedQnA1d


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        x = x.float()
        emb = emb.float()
        for layer in self:
            x = layer(x, emb)
        return x, None

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        assert self.head_dim * nhead == d_model, "d_model must be divisible by nhead"

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
# 
        # self.w_kv = SAM(dim=d_model, ca_num_heads=nhead*2)
        
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections
        Q = self.w_q(query)  # (batch_size, seq_len, d_model)
        # KV = self.w_kv(key)    # (batch_size, seq_len, d_model)
        # K, V = torch.chunk(KV, 2, dim=-1)
        K = self.w_k(key)  # (batch_size, seq_len, d_model)
        V = self.w_v(value)  # (batch_size, seq_len, d_model)

        # Split into multiple heads
        Q = Q.view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)  # (batch_size, nhead, seq_len, head_dim)
        K = K.view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        # print(f"K shape : {K.shape}")
        # print(f"V shape : {V.shape}")
        # print(f"Q shape : {Q.shape}")
        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))  # (batch_size, nhead, seq_len, seq_len)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        output = torch.matmul(attn_weights, V)  # (batch_size, nhead, seq_len, head_dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)  # (batch_size, seq_len, d_model)

        # Final linear projection
        output = self.w_o(output)
        return output


class PositionwiseFeedforward(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout=0.1):
        super(PositionwiseFeedforward, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, d_out_model, emb_dim, nhead, dim_feedforward, dropout=0.0):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.feedforward = PositionwiseFeedforward(d_model, dim_feedforward, dropout)
        self.out = nn.Linear(d_model, d_out_model) if d_model != d_out_model else nn.Identity()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.sequence_pos_encoder = PositionalEncoding(d_model, dropout)
        self.emb_proj = nn.Linear(emb_dim, d_model)

    def forward(self, src, emb):
        # print(f"emb : {emb.shape}")

        emb = self.emb_proj(emb)
        # print(f"emb : {emb.shape}")
        # print(f"src : {src.shape}")
        src = torch.cat((emb, src), axis=1)
        src = self.sequence_pos_encoder(src)
        
        # Multi-head self-attention
        attn_output = self.self_attn(src, src, src)
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)

        # Feedforward network
        ff_output = self.feedforward(src)
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)

        return self.out(src)[:,1:]




class SAM(nn.Module):
    def __init__(self, dim, ca_num_heads=4, qkv_bias=False, qk_scale=None, 
                       attn_drop=0., dropout=0., ca_attention=1, expand_ratio=2):
        super().__init__()

        self.ca_attention = ca_attention
        self.dim = dim
        self.ca_num_heads = ca_num_heads

        assert dim % ca_num_heads == 0, f"dim {dim} should be divided by num_heads {ca_num_heads}."
        print(f"Dropout in SAM : {dropout}")

        self.act = nn.GELU()
        self.proj_out = nn.Linear(dim, 2*dim, bias=qkv_bias)
        self.proj_drop = nn.Dropout(dropout)

        self.split_groups=self.dim//ca_num_heads
        
        # self.v = nn.Conv1d(dim, dim, kernel_size=3, padding=1, stride=1)
        # self.s = nn.Conv1d(dim, dim, kernel_size=3, padding=1, stride=1)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.s = nn.Linear(dim, dim, bias=qkv_bias)

        for i in range(self.ca_num_heads):
            # local_conv = nn.Conv1d(dim//self.ca_num_heads, dim//self.ca_num_heads, kernel_size=(3 + i*4), padding=2*i + 1, stride=1, groups=dim//self.ca_num_heads)
            local_conv = nn.Conv1d(dim//self.ca_num_heads, dim//self.ca_num_heads, kernel_size=(3+i*2), padding=i + 1, stride=1, groups=dim//self.ca_num_heads)
            setattr(self, f"local_conv_{i + 1}", local_conv)
        
        self.proj = nn.Sequential(
            nn.Conv1d(dim, dim*expand_ratio, kernel_size=1, padding=0, stride=1, groups=self.split_groups),
            nn.BatchNorm1d(dim*expand_ratio),
            nn.SiLU(),
            nn.Conv1d(dim*expand_ratio, dim, kernel_size=1, padding=0, stride=1)
        )

        # self.norm = normalization(dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)

        
        
    def forward(self, x):
        x = self.norm(x)
        B, N, C = x.shape
        v = self.v(x)
        s = self.s(x).reshape(B, N, self.ca_num_heads, C//self.ca_num_heads).permute(2, 0, 3, 1)
        
        for i in range(self.ca_num_heads):
            local_conv = getattr(self, f"local_conv_{i + 1}")
            s_i= s[i]
            s_i = local_conv(s_i).reshape(B, self.split_groups, -1, N) # + g_i.reshape(B, self.split_groups, -1, N)
            if i == 0:
                s_out = s_i
            else:
                s_out = torch.cat([s_out,s_i],2)
                
        s_out = s_out.reshape(B, C, N)
        s_out = self.proj(s_out)
        self.modulator = s_out
        s_out = s_out.reshape(B, C, N).permute(0, 2, 1)
        
        out = s_out * v
        
        out = self.proj_out(out)
        out = self.proj_drop(out)
        return out

class SAM_UNetModel(nn.Module):
    def __init__(
        self,
        cond_mask_prob,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        dropout=0.5,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=1,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        padding_mode='zeros',
        padding=1,
        cond_mode="text",
    ):
        super().__init__()
        
        self.cond_mode = cond_mode

        self.cond_mask_prob = cond_mask_prob
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.sub_sample_mult = np.power(2, len(self.channel_mult))
        self.dims = dims
        self.padding_mode = padding_mode
        self.padding = padding
        
        print(f"USE CHECKPOINT : {use_checkpoint}")


        self.text_dropout = nn.Dropout(0.0)
        time_embed_dim = model_channels * 4


        ch = int(channel_mult[0]) * model_channels
        

        self._feature_size = ch
        input_block_chans = [ch]
        
        out_ch = ch
        self.input_process = InputProcess(self.in_channels, ch)
        self.sequence_pos_encoder = PositionalEncoding(time_embed_dim, dropout)
        self.embed_timestep = TimestepEmbedder(time_embed_dim, self.sequence_pos_encoder)
        self.input_blocks = nn.ModuleList([])
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                # layers =  [TransformerEncoderLayer(d_model=ch,
                #                                  d_out_model=ch,
                #                                  emb_dim=time_embed_dim,
                #                                 nhead=4,              # Number of attention heads
                #                                 dim_feedforward=ch*2, # Feedforward network dimension
                #                                 dropout=dropout
                #                             ),]
                

                # layers.append(SAM(dim=out_ch, ca_num_heads=num_heads_upsample))
                
                ch = out_ch
                out_ch = ch

                self._feature_size += ch
                input_block_chans.append(ch)
                self.input_blocks.append(TransformerEncoderLayer(d_model=ch,
                                                 d_out_model=ch,
                                                 emb_dim=time_embed_dim,
                                                nhead=4,              # Number of attention heads
                                                dim_feedforward=ch*2, # Feedforward network dimension
                                                dropout=dropout
                                            ))


        self.output_blocks = nn.ModuleList([])
        
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks+1):
                ich = input_block_chans.pop()
                if level == len(channel_mult) - 1 and i == 0:
                    ich = 0
                # layers =  [TransformerEncoderLayer(d_model=ich+ch,
                #                                  d_out_model=ch,
                #                                  emb_dim=time_embed_dim,
                #                                 nhead=4,              # Number of attention heads
                #                                 dim_feedforward=ch*2, # Feedforward network dimension
                #                                 dropout=dropout
                #                             ),]
                # d_model, d_out_model, emb_dim, nhead, dim_feedforward, dropout=0.0
                
                # layers.append(SAM(dim=ch, ca_num_heads=num_heads_upsample))
                

                self.output_blocks.append(TransformerEncoderLayer(d_model=ich+ch,
                                                 d_out_model=ch,
                                                 emb_dim=time_embed_dim,
                                                nhead=4,              # Number of attention heads
                                                dim_feedforward=ch*2, # Feedforward network dimension
                                                dropout=dropout
                                            ))
                self._feature_size += ch


        self.output_process = OutputProcess(ch, out_channels)


        self.embed_text = nn.Linear(768, time_embed_dim)
        print('EMBED TEXT')
        print('Loading dangvantuan/vietnamese-embedding...')
        clip_version = 'dangvantuan/vietnamese-embedding'
        self.clip_model = self.load_and_freeze_clip(clip_version)
        

#         self.motion_args['cond_mask_prob'] = 0.2
        print(f"Probability for masking condition : {self.cond_mask_prob}")
        

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def load_and_freeze_clip(self, clip_version):
        word_embedding_model = models.Transformer(clip_version)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        clip_model = SentenceTransformer(modules=[word_embedding_model, pooling_model]) # Actually this line is unnecessary since clip by default already on float16

        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def mask_cond(self, cond, force_mask=False):
        # print("*******************",cond.device)
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    def encode_text(self, raw_text):
        device = next(self.parameters()).device
        embeddings = torch.tensor(self.clip_model.encode(raw_text)).float().to(device)
        return embeddings

    def forward(self, x, timesteps, y=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """ 

        x = x.squeeze(dim=2)
        
        bs, nfeats, nframes = x.shape

        emb = self.embed_timestep(timesteps)


        force_mask = y.get('uncond', False)
        
        if 'text_embed' in y.keys():  # caching option
            enc_text = y['text_embed']
        else:
            enc_text = self.encode_text(y['text'])
        emb += self.embed_text(self.mask_cond(enc_text, force_mask=force_mask))
        emb = emb.permute(1,0,2)  # [1, bs, d]

        x = x.reshape(bs, -1, nframes)


        x = self.input_process(x)
        # xseq = torch.cat((emb, x), axis=1)
        # xseq = self.sequence_pos_encoder(xseq)

        
        hs = [x]
        for module in self.input_blocks:
            x = module(x, emb)
            hs.append(x)
        for level, module in enumerate(self.output_blocks):
            if level == 0:
                x = hs.pop()
            else:
                x = th.cat([x, hs.pop()], dim=2)
            x = module(x, emb)


        _out = self.output_process(x)
        _out = _out.reshape(bs, nfeats, 1, nframes)
        return _out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x): # [bs, seqlen+1, d]
        x = x.permute(1,0,2)
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x).permute(1,0,2)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        timesteps = timesteps.to(self.sequence_pos_encoder.pe.device)
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, nfeats, nframes = x.shape
        x = x.permute((2, 0, 1)).reshape(nframes, bs, nfeats)
        x = self.poseEmbedding(x)  # [seqlen, bs, d]
        x = x.permute(1,0,2)
        return x



class OutputProcess(nn.Module):
    def __init__(self, input_feats, nfeats):
        super().__init__()
        self.input_feats = input_feats
        self.nfeats = nfeats
        self.poseFinal = nn.Linear(input_feats, nfeats)

    def forward(self, output):
        output = output.permute(1,0,2)
        nframes, bs, _ = output.shape
        output = self.poseFinal(output)  # [seqlen, bs, 150]
        output = output.reshape(nframes, bs, self.nfeats)
        output = output.permute(1, 2, 0)  # [bs, njoints, nfeats, nframes]
        return output
    
