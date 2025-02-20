import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        # Standard residual structure
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        # If the stride is not 1 or the number of input channels is not equal to the number of output channels, a convolution operation is required
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Attention_sample(nn.Module):
    
    def __init__(self, ):
        super(Attention_sample, self).__init__()

    def forward(self, x, w, b):
        # Enter the shape of x: (n*64, 512, 4, 4)
        x = x.flatten(start_dim=2)  # (n*64, 512, 16)
        x = x.permute(0, 2, 1)  # (n*64, 16, 512)
        y_beta = F.linear(x, w, b)  # y_beta: (n*64, 16, num_classes)
        gamma = x.shape[1]  # gamma = 16
        alpha = torch.sqrt((x ** 2).sum(dim=2))  # alpha: (n*64, 16)
        alpha = alpha / alpha.sum(dim=1, keepdim=True)  # alpha: (n*64, 16)
        alpha = F.relu(alpha - .1 / float(gamma))  # alpha: (n*64, 16)
        alpha = alpha / alpha.sum(dim=1, keepdim=True)  # alpha: (n*64, 16)
        out = alpha.unsqueeze(dim=2) * x  # out: (n*64, 16, 512)
        out = out.sum(dim=1)  # out: (n*64, 512)
        
        # Returns output, attention weight alpha, and y_beta
        return out, alpha, y_beta

class Attention_up(nn.Module):
    
    def __init__(self, keep_patch_threshold=True, top_patch_num=None):
        super(Attention_up, self).__init__()
        self.keep_patch_threshold = keep_patch_threshold
        self.top_patch_num = top_patch_num

    def forward(self, x, w, b, patch_alpha):
        # Process bag in one slide at a time
        # x = torch.squeeze(x, dim=0)

        if not self.keep_patch_threshold:
            x_ = torch.empty((x.shape[0], self.top_patch_num, x.shape[2]), device=x.device)
            for i in range(x.shape[0]):
                x_[i] = x[i, patch_alpha[i][:self.top_patch_num]]
            x = x_
        gamma = x.shape[1]
        out_alpha = F.linear(x, w, b)
        out = torch.sqrt((out_alpha ** 2).sum(dim=2))
        alpha = out / out.sum(dim=1, keepdim=True).expand_as(out)
        if self.keep_patch_threshold:
            # alpha = F.relu(alpha - .1 / float(gamma))
            alpha = F.relu(alpha - .1 / float(gamma))  #
        alpha = alpha / alpha.sum(dim=1, keepdim=True).expand_as(alpha)

        out = alpha.unsqueeze(dim=2).expand_as(x) * x
        out = out.sum(dim=1)
        embedding = out.detach()
        out = F.linear(out, w, b)

        # alpha = alpha.squeeze(dim=0).detach()
        alpha = alpha.detach()
        # out's shape: n * num_classes   embedding's shape: n * 512     out_alpha's shape: n * 64 * num_classes
        return out, embedding, alpha, out_alpha

class Attention(nn.Module):
    
    def __init__(self, keep_threshold=False):
        super(Attention, self).__init__()
        self.keep_threshold = keep_threshold

    def forward(self, x, w, b):
        # Process bag in one slide at a time
        # x = torch.squeeze(x, dim=0)
        gamma = x.shape[1]
        out_alpha = F.linear(x, w, b)
        
        out = torch.sqrt((out_alpha ** 2).sum(dim=2))
        alpha = out / out.sum(dim=1, keepdim=True).expand_as(out)
        if self.keep_threshold:
            # alpha = F.relu(alpha - .1 / float(gamma))
            alpha = F.relu(alpha - .1 / float(gamma))
        alpha = alpha / alpha.sum(dim=1, keepdim=True).expand_as(alpha)

        out = alpha.unsqueeze(dim=2).expand_as(x) * x
        out = out.sum(dim=1)
        out = F.linear(out, w, b) 

        alpha = alpha.squeeze(dim=0).detach()
        # out's shape: 1* num_classes   alpha's shape: gamma    out_alpha: 1 * gamma * num_classes
        return out, alpha, out_alpha.squeeze(dim=0)


class TransformerCellBlock(nn.Module):
    
    def __init__(self, in_features, num_heads, hidden_features=None, dropout=0.1):
        super(TransformerCellBlock, self).__init__()
        
        hidden_features = hidden_features or in_features
        
        self.norm1 = nn.LayerNorm(in_features)
        self.norm2 = nn.LayerNorm(in_features)
        
        self.multihead_attn = nn.MultiheadAttention(embed_dim=in_features, num_heads=num_heads, batch_first=True)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(in_features, 4 * hidden_features),  
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_features, in_features),  
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Input shape (N*64, 512, 4, 4)
        n, c, h, w = x.shape
        x_flat = x.view(n, h * w, c)  # (N*64, 16, 512)
        
        x_attn_input = self.norm1(x_flat)
        x_attn, _ = self.multihead_attn(x_attn_input, x_attn_input, x_attn_input)  # self-attention (N*64, 16, 512)
        x = x_flat + x_attn # Residual connection
        
        # LayerNorm + MLP
        x_mlp_input = self.norm2(x)
        out = self.feed_forward(x_mlp_input)  
        out = x + out

        out = out.view(n, c, h, w)  # (N*64, 512, 4, 4)
        
        return out

class TransformerPatchBlock(nn.Module):
    
    def __init__(self, dim, num_heads, hidden_dim, dropout=0.1):
        super(TransformerPatchBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        
        self.norm2 = nn.LayerNorm(dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        x_norm1 = self.norm1(x)
        attn_out, _ = self.attention(x_norm1, x_norm1, x_norm1)
        x = x + attn_out  # Residual connection + Dropout

        x_norm2 = self.norm2(x)
        ff_out = self.feed_forward(x_norm2)
        x = x + ff_out  

        return x  # Output shape: (N*64, 512)

class net_up_aa(nn.Module):
    
    def __init__(self, block, num_blocks=[2, 2, 2, 2], num_classes=2, keep_patch_threshold=True, top_patch_num=None):
        super(net_up_aa, self).__init__()
        self.in_planes = 64

        # Initial convolution layer: input shape (N, 3, 1024, 1024), output shape (N, 64, 512, 512)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)  
        self.bn1 = nn.BatchNorm2d(64)
        # Maximum pooling layer: input shape (N, 64, 512, 512), output shape (N, 64, 256, 256)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.TransformerCellBlock = TransformerCellBlock(512, 8, 512)
        self.TransformerPatchBlock = TransformerPatchBlock(512, 4, 512)

        self.linear = nn.Linear(512, num_classes) #(N*64, num_classes)
        
        self.num_classes = num_classes
        self.attention_sample = Attention_sample()
        self.attention = Attention_up(keep_patch_threshold=keep_patch_threshold, top_patch_num=top_patch_num)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, patch_alpha):
        n = x.shape[0]
        
        out = rearrange(x, 'N C (ph h) (pw w) -> (N ph pw) C h w', ph=8, pw=8) #output (N*64, 3, 128, 128)
        out = F.gelu(self.bn1(self.conv1(out)))

        out = self.maxpool(out)

        out = self.layer1(out) #(N*64, 64, 32, 32)
        out = self.layer2(out) #(N*64, 128, 16, 16)
        out = self.layer3(out) #(N*64, 256, 8, 8)
        out = self.layer4(out) #(N*64, 512, 4, 4)

        # Transformer_block weights the Attention of cell_representation (N*64, 512, 4, 4) to obtain patch_embedding (N*64, 512).
        cell_reperatation = cell_reperatation_Identity = out
        cell_reperatation = self.TransformerCellBlock(cell_reperatation)  
        cell_reperatation = self.TransformerCellBlock(cell_reperatation)  
        cell_reperatation = self.TransformerCellBlock(cell_reperatation)   
                  
        cell_reperatation = cell_reperatation + cell_reperatation_Identity    
        patch_embedding, _, _ = self.attention_sample(cell_reperatation, self.linear.weight, self.linear.bias)
        
        # Transformer_block weights the Attention of patch_embedding (N*64, 512) to bag_embedding (N*64, 512).
        patch_embedding = patch_embedding.reshape(n, -1, patch_embedding.shape[-1])
        patch_embedding_Identity = patch_embedding
        patch_embedding = self.TransformerPatchBlock(patch_embedding) 
        patch_embedding = self.TransformerPatchBlock(patch_embedding) 
                   
        patch_embedding = patch_embedding + patch_embedding_Identity        
        _, bag_embedding, _, _ = self.attention(patch_embedding, self.linear.weight, self.linear.bias, patch_alpha)
        
        # out shape: (N*64, 512), beta shape: (N*64, 512), y_beta shape: (N*64, 2)
        out, beta, y_beta = self.attention_sample(out, self.linear.weight, self.linear.bias)     
        out = out.reshape(n, -1, out.shape[-1])
        out, embedding, alpha, out_alpha = self.attention(out, self.linear.weight, self.linear.bias, patch_alpha)

        return out, bag_embedding, alpha, out_alpha, beta, y_beta

class MIL(nn.Module):
    
    def __init__(self, dim_fectures=512, num_classes=2, num_hidden_unit=32, keep_threshold=True):
        super(MIL, self).__init__()
        self.linear0 = nn.Linear(dim_fectures, num_hidden_unit)
        self.linear1 = nn.Linear(num_hidden_unit, num_classes)
        self.attention = Attention(keep_threshold=keep_threshold)

    def forward(self, x):
        x = F.linear(x, self.linear0.weight, self.linear0.bias)
        return self.attention(x, self.linear1.weight, self.linear1.bias)

class MRAN(nn.Module):
    
    def __init__(self, num_classes=2, dim_features=512, num_hidden_unit=32, keep_bag_threshold=True, keep_patch_threshold=True, top_patch_num=None):
        super(MRAN, self).__init__()
        self.upstream = net_up_aa(block=BasicBlock, num_classes=num_classes, keep_patch_threshold=keep_patch_threshold, top_patch_num=top_patch_num)
        self.downstream = MIL(dim_fectures=512, num_classes=2, num_hidden_unit=32, keep_threshold=True)

    def forward(self, x, tag, patch_alpha=-1):
        if tag == 0:
            return self.upstream(x, patch_alpha=patch_alpha)
        else:
            return self.downstream(x)