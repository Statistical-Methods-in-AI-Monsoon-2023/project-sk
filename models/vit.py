import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, Din, Dout):
        super.__init__()
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

        attention_logits = q @ k.T
        attention_mask = F.softmax(attention_logits, dim=0)
        output = attention_mask @ v

        return output

class Transformer(nn.Module):
    def __init__(self, num_attention, D):
        super().__init__()
        self.layers = []
        for _ in range(num_attention):
            self.layers.append(Attention(D, D))
    
    def forward(self, x):
        '''
        forward pass through the transformer
        '''
        for layer in self.layers:
            x = layer(x)
        return x # (B, N, D)

class VIT(nn.Module):
    def __init__(self):
        super.__init__()
        self.img_size = (32, 32)
        C = 3
        P = 4
        self.patch_size = P
        self.num_patches = self.img_size[0] * self.img_size[1] / (P ** 2)
        D = 128
        num_classes = 10
        num_attention_blocks = 5
        self.patch_embedding = nn.Linear(C*P**2, D)
        self.encoder = Transformer(num_attention_blocks, D)
        self.final_proj = nn.Linear(D, num_classes)
    
    def forward(self, img):
        '''
        img is a torch tensor of shape (B, 3, 32, 32)
        forward pass through the VIT
        '''
        B = img.shape[0]
        patches = img.unfold(1, self.patch_size, self.patch_size).unfold(2, self.patch_size, self.patch_size) # (B, C, Nx, Ny, P, P)
        patches = patches.permute(0, 2, 3, 1, 4, 5) # (B, Nx, Ny, C, P, P)
        patches = patches.reshape(B, self.num_patches, -1)
        patch_embed = self.patch_embedding(patches) # (B, N, D)
        encoded = self.encoder(patch_embed) # (B, N, D)
        encoded = encoded.mean(dim = 1) # (B, D)
        out_logits = self.final_proj(encoded) # (B, num_classes)
        
        return out_logits