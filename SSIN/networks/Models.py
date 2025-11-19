''' Define the Transformer model '''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from SSIN.networks.Layers import NewRelativeEncoderLayer


def gelu(x):
   return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class TwoLayerFCN(nn.Module):
    def __init__(self, feat_dim, n_hidden1, n_hidden2):
        super().__init__()
        self.feat_dim = feat_dim
        self.linear_1 = nn.Linear(feat_dim, n_hidden1)
        self.linear_2 = nn.Linear(n_hidden1, n_hidden2)

    def forward(self, in_vec, non_linear=False):
        """pos_vec: absolute position vector, n * feat_dim"""
        assert in_vec.shape[-1] == self.feat_dim, f"in_vec.shape: {in_vec.shape}, feat_dim:{self.feat_dim}"

        if non_linear:
            mid_emb = F.gelu(self.linear_1(in_vec))
        else:
            mid_emb = self.linear_1(in_vec)

        out_emb = self.linear_2(mid_emb)
        return out_emb


class SpaFormer(nn.Module):
    def __init__(self, d_feat, d_pos, n_layers, n_head, d_k, d_v,
                 d_model, d_inner, dropout=0.1, scale_emb=False, return_attns=False, temperature=None):
        super().__init__()

        self.d_model = d_model
        self.scale_emb = scale_emb
        self.return_attns = return_attns

        self.feature_enc = TwoLayerFCN(d_feat, d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)

        self.layer_stack = nn.ModuleList(
            [NewRelativeEncoderLayer(d_model, d_inner, n_head, d_k, d_v, d_pos, dropout=dropout, temperature=temperature)
             for _ in range(n_layers)])

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.temporal_conv = TemporalConvBlock(d_model, kernel_size=3, dropout=dropout) #加一層Conv

        self.linear = nn.Linear(d_model, d_model)
        self.activ2 = gelu
        self.decoder = TwoLayerFCN(d_model, d_model, 1)

    def forward(self, feat_seq, r_pos_mat, masked_pos, attn_mask=None):
        enc_slf_attn_list = []

        # -- Forward
        enc_output = self.feature_enc(feat_seq, non_linear=True)
        if self.scale_emb:
            enc_output *= self.d_model ** 0.5

        enc_output = self.dropout(enc_output)
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, r_pos_mat, attn_mask=attn_mask)
            enc_slf_attn_list += [enc_slf_attn] if self.return_attns else []

        
        enc_output = self.temporal_conv(enc_output) #Transformer layer加一層卷積

        masked_pos = masked_pos[:, :, None].expand(-1, -1, enc_output.size(-1)) 


        h_masked_1 = torch.gather(enc_output, 1, masked_pos)

        context = enc_output.mean(dim=1, keepdim=True)     
        h_masked_1 = h_masked_1 + context                   

        h_masked_2 = self.layer_norm(self.activ2(self.linear(h_masked_1)))
        dec_output = self.decoder(h_masked_2, non_linear=True) 

        if self.return_attns:
            return dec_output, h_masked_1, h_masked_2, enc_slf_attn_list
        return dec_output, h_masked_1, h_masked_2


class TemporalConvBlock(nn.Module):
    def __init__(self, d_model, kernel_size=3, dropout=0.1):
        super().__init__()
        padding = kernel_size // 2  # 長度不變
        self.conv = nn.Conv1d(d_model, d_model, kernel_size, padding=padding)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model]
        """
        residual = x

        #transpose
        x = x.transpose(1, 2) 
        x = self.conv(x)
        x = x.transpose(1, 2) 
        x = gelu(x)
        x = self.dropout(x)
        x = x + residual
        x = self.layer_norm(x)
        return x