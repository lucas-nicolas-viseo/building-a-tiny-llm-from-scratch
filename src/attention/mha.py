import torch
import torch.nn as nn 

class MHA(nn.Module):
    def __init__(self,d_in,d_out,context_length,num_heads:int,dropout:float,qkv_bias=False):

        super().__init__()
        assert (d_out % num_heads == 0), "d_out must be divisible by number of heads" 
        self.d_out= d_out
        self.num_heads=num_heads
        self.head_dims = d_out // num_heads

        self.W_q = nn.Linear(d_in,d_out,qkv_bias)
        self.W_k = nn.Linear(d_in,d_out,qkv_bias)
        self.W_v = nn.Linear(d_in,d_out,qkv_bias)

        self.out_proj = nn.Linear(d_out,d_out)

        self.dropout = nn.Dropout(dropout)

        self.register_buffer("mask",torch.triu(torch.ones(context_length,context_length), diagonal=1))


        
        
    def forward(self, x):
        b,num_token,_ = x.shape
        
        #group q,k and v by head for mha computation
        # (b, num_heads, num_token, head_dims)
        q = self.W_q(x).view(b,num_token,self.num_heads,self.head_dims).transpose(1,2)
        k = self.W_k(x).view(b,num_token,self.num_heads,self.head_dims).transpose(1,2)
        v = self.W_v(x).view(b,num_token,self.num_heads,self.head_dims).transpose(1,2)
        
        att_mask = self.mask.bool()[:num_token,:num_token]
        
        att_scores = (q @ k.transpose(-2,-1)).masked_fill_(att_mask, -torch.inf)
        
        att_weights = self.dropout(torch.softmax(att_scores/(self.head_dims**0.5),dim =-1))
        
        return self.out_proj((att_weights @ v).transpose(2,1).contiguous().view(b,num_token,-1))
        

