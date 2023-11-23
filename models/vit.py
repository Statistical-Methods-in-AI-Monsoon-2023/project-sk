import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, Din, Dout):
        super().__init__()
        self.query = nn.Linear(Din, Dout)
        self.key = nn.Linear(Din, Dout)
        self.value = nn.Linear(Din, Dout)

    def forward(self, x):
        '''
        x is a tensor of shape (B, N, Din) representing N tensors of shape D in a batch size of B
        returns a tensor of shape (B, N, Dout) representing the vectors after attending to each other
        '''
        q = self.query(x) # (B, N, Dout)
        k = self.key(x) # (B, N, Dout)
        v = self.value(x) # (B, N, Dout)

        attention_logits = q @ torch.transpose(k, 1, 2)
        attention_mask = F.softmax(attention_logits, dim=0)
        output = attention_mask @ v

        return output

class Transformer(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(D)
        self.layer_norm2 = nn.LayerNorm(D)
        self.attention = Attention(D, D)
        self.linear = nn.Linear(D, D)

    def forward(self, x):
        '''
        forward pass through the transformer
        '''
        y = self.layer_norm1(x)
        y = self.attention(y) + x
        z = self.layer_norm2(y)
        z = self.linear(z) + y

        return z # (B, N, D)

class VIT(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_size = (32, 32)
        C = 3
        P = 4
        D = 64
        num_classes = 10
        num_transformer_blocks = 6
        self.patch_size = P
        self.num_patches = self.img_size[0] * self.img_size[1] // (P ** 2)
        self.pos_embeddings = nn.Parameter(torch.randn(1, self.num_patches, D))
        self.cls_token = nn.Parameter(torch.randn(D))
        self.patch_embedding = nn.Linear(C*P**2, D)
        self.encoder_layers = []
        for _ in range(num_transformer_blocks):
            self.encoder_layers.append(Transformer(D))
        self.encoder = nn.Sequential(*self.encoder_layers)
        self.final_proj = nn.Linear(D, num_classes)
    
    def forward(self, img):
        '''
        img is a torch tensor of shape (B, 3, 32, 32)
        forward pass through the VIT
        '''
        B = img.shape[0]
        patches = img.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size) # (B, C, Nx, Ny, P, P)
        patches = patches.permute(0, 2, 3, 1, 4, 5) # (B, Nx, Ny, C, P, P)
        patches = patches.reshape(B, self.num_patches, -1)
        patch_embed = self.patch_embedding(patches) + self.pos_embeddings # (B, N, D)
        patch_embed = torch.column_stack([patch_embed, self.cls_token.reshape(1, 1, -1)], dim = 1)
        encoded = self.encoder(patch_embed) # (B, N, D)
        encoded = encoded[:, -1] # (B, D)
        out_logits = self.final_proj(encoded) # (B, num_classes)
        
        return F.sigmoid(out_logits)