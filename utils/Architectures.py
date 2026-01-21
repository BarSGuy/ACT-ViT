from einops.layers.torch import Rearrange
import torch.nn.functional as F
import torch.nn as nn
import torch
from utils.constants import FEATURE_DIMS
from einops import repeat
from vit_pytorch import ViT



def get_model(args, input_shape, input_dim):
    model_mapping = {
        'ACT-Vit-foundation': ACT_Vit_foundation,
        'ACT-MLP-foundation': ACT_MLP_foundation,
    }
    return model_mapping[args.probe_model](args=args, input_dim=input_dim, input_shape=input_shape)
    

######################## ACT_Vit_foundation ########################


class ModuleListPerLLMLinear(nn.Module):
    def __init__(self, in_features, out_features, num_llms):
        super(ModuleListPerLLMLinear, self).__init__()
        self.num_llms = num_llms
        self.adapters = nn.ModuleList([nn.Linear(in_features, out_features) for _ in range(num_llms)])
        
    def forward(self, input, indices):
        out = torch.zeros(*input.shape[:-1], self.adapters[0].out_features,
                        device=input.device, dtype=input.dtype)
        for i in torch.unique(indices):
            mask = (indices == i)
            adapter_out = self.adapters[i](input[mask].to(torch.float32)).to(input.dtype)
            out[mask] = adapter_out
        return out




class PatchEmbeddingWithPos(nn.Module):
    def __init__(self, patch_height, patch_width, patch_dim, hidden_dim):
        super().__init__()
        self.rearrange1 = Rearrange('b c (h p1) (w p2) -> b h w p1 p2 c', p1=patch_height, p2=patch_width)
        self.rearrange2 = Rearrange('b h w p1 p2 c -> b (h w) (p1 p2 c)')

        self.pos_embedding = nn.Parameter(torch.randn(1, 1, patch_height, patch_width, hidden_dim))

        self.norm1 = nn.LayerNorm(patch_dim)
        self.linear = nn.Linear(patch_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x.shape: [B, c, h=L, w=N]
        x = self.rearrange1(x)  # x.shape: [B, num_patches, patch_dim]
        x = x + self.pos_embedding
        x = self.rearrange2(x)

        x = self.norm1(x)
        x = self.linear(x)
        x = self.norm2(x)
        return x


        

class ACT_Vit_foundation(nn.Module):
    def __init__(self, args, input_dim, input_shape):
        super(ACT_Vit_foundation, self).__init__()
        
        self.input_dim = input_dim
        self.input_shape = input_shape #
        self.hidden_dim = args.hidden_dim
        self.heads = args.heads
        self.dropout = args.dropout
        self.num_layers = args.num_layers
        self.patch_size = eval(args.patch_size)
        
        
        # Unpack input shape
        L, N, _ = self.input_shape
        
        assert L % self.patch_size[0] == 0 and N % self.patch_size[1] == 0, "Patch size must be divisible by input shape"
        
        in_dim_per_llm = list(FEATURE_DIMS.values()) # should match the llm indices
        # NOTE: This creates the adapters for all the possible LLMs I am using -- not the actual ones appearing in this experiment
        self.linear_adapters = ModuleListPerLLMLinear(max(in_dim_per_llm), self.hidden_dim, num_llms=len(in_dim_per_llm))

        # Create the ViT module from vit_pytorch
        self.vit = ViT(
            image_size=(L, N),
            patch_size=self.patch_size,
            num_classes=1,
            dim=self.hidden_dim,
            depth=self.num_layers,
            heads=self.heads,
            mlp_dim=self.hidden_dim,
            dropout=self.dropout,
            emb_dropout=self.dropout,
            channels=self.hidden_dim,
            pool=args.pool
        )
        
        def pair(t):
            return t if isinstance(t, tuple) else (t, t)
        
        patch_height, patch_width = pair(self.patch_size)
        patch_dim = self.hidden_dim * patch_height * patch_width
        
        image_height, image_width = pair((L, N))
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.input_dim, patch_height * patch_width))
        # Override patch embedding method
        self.vit.to_patch_embedding = PatchEmbeddingWithPos(
            patch_height=patch_height,
            patch_width=patch_width,
            patch_dim=patch_dim,
            hidden_dim=self.hidden_dim,
        )

        
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, activations, indices):
        # fast version
        # x = self.adapters_fast(activations, indices)
        x = self.linear_adapters(activations, indices)
            
        # Input x is of shape (B, L, N, d)
        B, L, N, d = x.shape 

        # Permute the input to (B, d, L, N) which is (B, channels, height, width)
        x = x.permute(0, 3, 1, 2)
        
        # Forward pass through ViT
        preds = self.vit(x)  # Output shape: (B, num_classes)
        preds = self.sigmoid(preds).squeeze(-1)
        return preds


######################## ACT_MLP_foundation ########################

class ACT_MLP(nn.Module):

    def __init__(self, args, input_dim, input_shape):

        super(ACT_MLP, self).__init__()        
        self.input_dim = input_dim
        self.input_shape = input_shape #
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.dropout = args.dropout
        
        # Linear layers
        self.lin_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for i in range(self.num_layers):
            in_dim = self.hidden_dim if i > 0 else self.input_shape[-1] * self.input_shape[-2] * self.input_shape[-3]
            out_dim = self.hidden_dim if (i+1) < self.num_layers else 1
            self.lin_layers.append(nn.Linear(in_dim, out_dim))
            if (i+1) < self.num_layers:
                self.batch_norms.append(nn.BatchNorm1d(out_dim))

        # Output act
        self.sigmoid = nn.Sigmoid()

    def forward(self, activations):
        B, L, N, d = activations.shape 

        x = activations.reshape(B, L * N * d).to(torch.float32)
        
        for i in range(self.num_layers):
            x = self.lin_layers[i](x)
            if (i+1) < self.num_layers:
                x = self.batch_norms[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout)
        return self.sigmoid(x).squeeze(-1)  # Apply sigmoid for binary classification


class ACT_MLP_foundation(ACT_MLP):
    def forward(self, activations, indices):
        # Ignore indices and call parent's forward method
        return super().forward(activations)
