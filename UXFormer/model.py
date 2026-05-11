import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderMiniBlock(nn.Module):
    def __init__(self, in_channels, n_filters=32):
        super(EncoderMiniBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, n_filters, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(n_filters)
        self.dropout = nn.Dropout(0.4)
        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(n_filters)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        return x

class DecoderMiniBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderMiniBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(channel)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        out = self.channel_att(x) * x
        out = self.spatial_att(out) * out
        return out

class PatchEmbed(nn.Module):
    """Image to Patch Embedding for ViT"""
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=32):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)  # B, N, C
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossAttention(nn.Module):
    """Cross Attention between ViT features (key, value) and CNN features (query)"""
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, vit_features, cnn_features):
        B, N, C = vit_features.shape
        
        q = self.q(cnn_features).reshape(B, cnn_features.shape[1], self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(vit_features).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, cnn_features.shape[1], C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
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

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., 
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MultiHeadAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                                     attn_drop=attn_drop, proj_drop=drop)
        
        self.drop_path = nn.Identity() if drop_path == 0. else nn.Dropout(drop_path)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=32, depth=1,
                 num_heads=2, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, ape=False):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.ape = ape

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        
        self.norm = norm_layer(embed_dim)
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x

class FinalPatchExpand_X4(nn.Module):
    """Final patch expansion to match original image size"""
    def __init__(self, input_resolution, dim_scale=4, dim=32):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16*dim, bias=False)
        self.output_dim = dim 
        self.norm = nn.LayerNorm(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        from einops import rearrange
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2))
        x = x.view(B,-1,self.output_dim)
        x= self.norm(x)
        return x

class UXFormer(nn.Module):
    def __init__(self, n_channels, n_channels_transformer, n_classes, n_filters=32, 
                 depths=[1, 1, 1], num_heads=[2, 4, 8], window_size=7,    # 4개 -> 3개로 줄임
                 mlp_ratio=4., qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0, 
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, ape=False, 
                 img_size=224, patch_size=4, deep_supervision=False):
        super(UXFormer, self).__init__()
        
        self.n_channels = n_channels 
        self.n_channels_transformer = n_channels_transformer 
        self.n_classes = n_classes
        self.ape = ape
        self.img_size = img_size
        self.patch_size = patch_size
        self.deep_supervision = deep_supervision
        
        self.vit_dims = [n_filters, n_filters*2, n_filters*4]  # [32, 64, 128] (256 제거)
        self.cross_heads = num_heads  # [2, 4, 8] 
        self.depths = depths  # [1, 1, 1] 
        
        # U-Net Encoder 
        self.cblock1 = EncoderMiniBlock(n_channels, n_filters)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.cblock2 = EncoderMiniBlock(n_filters, n_filters*2)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.cblock3 = EncoderMiniBlock(n_filters*2, n_filters*4)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        self.cblock4 = EncoderMiniBlock(n_filters*4, n_filters*8)

        # Vision Transformer
        self.vit = VisionTransformer(
            img_size=img_size, patch_size=patch_size, in_chans=n_channels_transformer,
            embed_dim=self.vit_dims[0], depth=depths[0], num_heads=num_heads[0],
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
            norm_layer=norm_layer, ape=ape
        )
        
        # CNN feature patch embedding layers 
        self.patch_embed_1 = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=n_filters, embed_dim=n_filters)
        self.patch_embed_2 = PatchEmbed(
            img_size=img_size//2, patch_size=patch_size, in_chans=n_filters*2, embed_dim=n_filters*2)
        self.patch_embed_3 = PatchEmbed(
            img_size=img_size//4, patch_size=patch_size, in_chans=n_filters*4, embed_dim=n_filters*4)

        self.vit_dim_adjust2 = nn.Linear(self.vit_dims[0], self.vit_dims[1])  # 32 → 64
        self.vit_dim_adjust3 = nn.Linear(self.vit_dims[1], self.vit_dims[2])  # 64 → 128

        self.cross_attn1 = CrossAttention(self.vit_dims[0], num_heads=self.cross_heads[0], qkv_bias=qkv_bias, 
                                        attn_drop=attn_drop_rate, proj_drop=drop_rate)
        self.cross_attn2 = CrossAttention(self.vit_dims[1], num_heads=self.cross_heads[1], qkv_bias=qkv_bias, 
                                        attn_drop=attn_drop_rate, proj_drop=drop_rate)
        self.cross_attn3 = CrossAttention(self.vit_dims[2], num_heads=self.cross_heads[2], qkv_bias=qkv_bias, 
                                        attn_drop=attn_drop_rate, proj_drop=drop_rate)

        self.cbam1 = CBAM(n_filters * 2)
        self.cbam2 = CBAM(n_filters * 2 * 2)
        self.cbam3 = CBAM(n_filters * 4 * 2)

        self.up3 = nn.ConvTranspose2d(n_filters*8, n_filters*4, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(n_filters*4, n_filters*2, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(n_filters*2, n_filters, kernel_size=2, stride=2)

        self.ublock7 = DecoderMiniBlock(n_filters*4*2, n_filters*4)
        self.ublock8 = DecoderMiniBlock(n_filters*2*2, n_filters*2)
        self.ublock9 = DecoderMiniBlock(n_filters*2, n_filters)
        
        # Final output layer
        self.conv10 = nn.Conv2d(n_filters, n_classes, kernel_size=1)

        if self.deep_supervision:
            self.deep_head3 = nn.Conv2d(n_filters*4, n_classes, kernel_size=1)   # 56x56  
            self.deep_head2 = nn.Conv2d(n_filters*2, n_classes, kernel_size=1)   # 112x112
            self.deep_head1 = nn.Conv2d(n_filters, n_classes, kernel_size=1)     # 224x224 (최종)

        self.scale_up1 = FinalPatchExpand_X4(input_resolution=(56, 56), dim_scale=4, dim=32)
        self.scale_up2 = FinalPatchExpand_X4(input_resolution=(28, 28), dim_scale=4, dim=64)
        self.scale_up3 = FinalPatchExpand_X4(input_resolution=(14, 14), dim_scale=4, dim=128)

    def forward(self, x_unet, x_transformer):
        # U-Net Encoder - temporal features
        c1 = self.cblock1(x_unet)  # 224x224x32
        p1 = self.pool1(c1)
        c2 = self.cblock2(p1)     # 112x112x64
        p2 = self.pool2(c2)
        c3 = self.cblock3(p2)     # 56x56x128
        p3 = self.pool3(c3)
        c4 = self.cblock4(p3)     # 28x28x256
        
        vit_features = self.vit(x_transformer)  # [B, num_patches+1, 32]
        
        c1_vit = self.patch_embed_1(c1)  # [B, 3136, 32] (56x56 patches)
        c2_vit = self.patch_embed_2(c2)  # [B, 784, 64] (28x28 patches)
        c3_vit = self.patch_embed_3(c3)  # [B, 196, 128] (14x14 patches)
       
        s1 = self.cross_attn1(vit_features, c1_vit)  # [B, 3136, 32]
        
        s1_adj = self.vit_dim_adjust2(s1)  # [B, 3136, 64]
        s1_downsampled = s1_adj[:, :c2_vit.shape[1], :]  # [B, 784, 64]
        s2 = self.cross_attn2(s1_downsampled, c2_vit)  # [B, 784, 64]
        
        s2_adj = self.vit_dim_adjust3(s2)  # [B, 784, 128]
        s2_downsampled = s2_adj[:, :c3_vit.shape[1], :]  # [B, 196, 128]
        s3 = self.cross_attn3(s2_downsampled, c3_vit)  # [B, 196, 128]
        
        s1_scale_up = self.scale_up1(s1)
        s1_scale_up = s1_scale_up.view(s1_scale_up.shape[0], 224, 224, 32)
        s1_scale_up = s1_scale_up.permute(0, 3, 1, 2)
        
        s2_scale_up = self.scale_up2(s2)
        s2_scale_up = s2_scale_up.view(s2_scale_up.shape[0], 112, 112, 64)
        s2_scale_up = s2_scale_up.permute(0, 3, 1, 2)
        
        s3_scale_up = self.scale_up3(s3)
        s3_scale_up = s3_scale_up.view(s3_scale_up.shape[0], 56, 56, 128)
        s3_scale_up = s3_scale_up.permute(0, 3, 1, 2)
        
        u3 = self.up3(c4)  
        a3 = self.cbam3(torch.cat([s3_scale_up, u3], dim=1)) 
        d3 = self.ublock7(a3) 
        
        u2 = self.up2(d3)
        a2 = self.cbam2(torch.cat([s2_scale_up, u2], dim=1))
        d2 = self.ublock8(a2)
        
        u1 = self.up1(d2)
        a1 = self.cbam1(torch.cat([s1_scale_up, u1], dim=1))
        d1 = self.ublock9(a1)
        
        final_output = self.conv10(d1)
       
        if self.deep_supervision:
           
            deep_out3 = self.deep_head3(d3)  # [B, C, 56, 56]
            deep_out2 = self.deep_head2(d2)  # [B, C, 112, 112]
            deep_out1 = self.deep_head1(d1)  # [B, C, 224, 224]
           
            deep_out3_up = F.interpolate(deep_out3, size=(224, 224), mode='bilinear', align_corners=False)
            deep_out2_up = F.interpolate(deep_out2, size=(224, 224), mode='bilinear', align_corners=False)
           
            deep_outputs = [deep_out3_up, deep_out2_up, deep_out1]
            
            return final_output, deep_outputs, [d3, d2, d1]
        
        else:
            return final_output, [d3, d2, d1]