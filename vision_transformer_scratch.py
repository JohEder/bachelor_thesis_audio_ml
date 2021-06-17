import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    """Split image into patches an then embed them.

    Parameters
    -----------
    img_size : in
    Size of the image (is is a square)

    patch_size : int
    Size of the patch (it is a square)

    in_chans : int rgb -> 3
    number of input channels

    embed_dim : int
    the embedding dimension


    Attributes 
    ----------
    n_patches : int
    number of patches inside of our image

    proj : nn.Conv2d
    Convolutional layer that does both the splitting into patches and their embedding
    """


    def __init__(self, img_size, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        """Run forward pass.

        Parameters
        ----------
        x torch.Tensor shape (n_smaples, in_chans, img_size, img_size)

        Returns
        -----
        torch.Tensor (n_samples, n_patches, embed_dim)

        """

        x = self.proj(x)  #(n_samples, embed_dim, n_patches ** 0.5, n_patches**0.5)
        x = x.flatten(2) #(n_samples, embed_dim, n_patches) taking the last two dimensions and flattening them
        x = x.transpose(1, 2) #(n_samples, n_patches, embed_dim) swaping the last two dimensions

        return x


class Attention(nn.Module):
    """Attension mechanism
    
    params
    dim : in
        The input and output dimension of per token features

    n_heads : int
        Number of attention heads

    qkv_bias : bool
    If True then we include bias to the query, key and value projections

    attn_p : float
        Dropout probability applied to the query, key and value tensors

    proj_p : float
        Dropout probability applied to the output tensor.
    

    Attributes
    ---------
    scale : float
        normalizing constant for the dot product

    qkv : nn.Linear
        Linear projection for hte query, key and value

    proj : nn.Linear
        Linear mapping that takes in the concatenated output of all attention heads and maps it into a new space.
    
    attn_drop, proj_drop : nn.Dropout
    """

    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5 #do not feed too large values into the softmax layer

        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.atttn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)


    def forward(self, x):
        """Run forward pass
        params:
        x : torch.Tensor shape  (n_samples, n_patches + 1, dim)

        Returns
        torch.Tensor
        (n_samples, n_patches + 1, dim)
        """

        n_samples, n_tokens, dim = x.shape
        if dim != self.dim:
            raise ValueError
        
        qkv = self.qkv(x) #(n_samples, n_patches + 1, 3*dim)
        qkv = qkv.reshape(n_samples, n_tokens, 3, self.n_heads, self.head_dim) #(n_samples, n_patches + 1, 3, n_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4) #change order (3, n_samples, n_heads, n_patches + 1, head_dim)

        q, k, v = qkv[0], qkv[1], qkv[2]
        k_t = k.transpose(-2, -1) #(n_samples, n_heads, head_dim, n_patches + 1)

        dp = (q @ k_t) * self.scale #dot product (n_samples, n_heads, n_patches + 1, n_patches + 1)
        attn = dp.softmax(dim= -1) #apply softmax to hte last dimension (n_samples, n_heads, n_patches + 1, n_patches + 1)
        attn = self.atttn_drop(attn)

        weighted_avg = attn @ v #(n_samples, n_heads, n_patches + 1, head_dim)
        weighted_avg = weighted_avg.transpose(
            1, 2
        )       #(n_samples, n_patches + 1, n_heads, head_dim) 
        weighted_avg = weighted_avg.flatten(2) #(n_samples, n_patches + 1, dim)

        weighted_avg = weighted_avg.flatten(2) #(n_samples, n_patches + 1, dim)

        x = self.proj(weighted_avg) # (n_samples, n_patches + 1, dim)
        x = self.proj_drop(x) # (n_samples, n_patches + 1, dim)

        return x

    


class MLP(nn.Module):
    """Multilayer percepron. 
    

    
    
    """