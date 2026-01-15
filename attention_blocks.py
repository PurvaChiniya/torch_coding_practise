import torch 
import math 
import torch.nn as nn 
def scaled_dot_product_attention(Q, K, V, mask ):
    # the shape of QKV is batch_size, num_heads, seq_len, dk 
    dot_product = torch.matmul(Q, K.transpose(-1, -2)) 
    # dot product shape is batch_size, num_heads, seq_len, seq_len
    dk = Q.shape[-1]
    scaled_dot_product = dot_product/ math.sqrt(dk)
    
    if mask is not None:
        scaled_dot_product = torch.masked_fill(scaled_dot_product,mask==0, value=float("-inf"))
    scaled_dot_product = torch.softmax(scaled_dot_product, dim=-1)
    attention_scores = torch.matmul(scaled_dot_product, V)
    return attention_scores, scaled_dot_product

class FastMultiHeadAttention(torch.nn.Module):
    def __init__(self, num_heads, embed_size):
        super().__init__()
        self.num_heads = num_heads 
        self.embed_size = embed_size 
        self.dk = embed_size// num_heads
        # for parallel ocmputaion we create a. single layer of qkv matrices instead of 3 
        self.qkv = nn.Linear(embed_size, 3*embed_size)# we usually dont have a bias here 
        self.output_layer = nn.Linear(embed_size, embed_size)
    def forward(self,x, mask):
        # x shape is batch_size, seq_len, embed_size
        qkv = self.qkv(x) 
        batch_size = x.shape[0]
        # so this computation is getting parallely executed on gpu 
        # earlier we used to do q = nn.linear(x) v = nn.linear(x) whihc brings the series computations
        # qkv will be  batch_size, seq_len, 3*embed_size 
        Q, K , V = torch.chunk(qkv, chunks=3, dim = -1)
        # now we can proceed wiht normal scaled dot product attention 
        # make parallel across heads 
        Q = Q.view(batch_size, -1,self.num_heads , self.dk ).transpose(1,2)
        V = V.view(batch_size, -1,self.num_heads , self.dk ).transpose(1,2)
        K = K.view(batch_size, -1,self.num_heads , self.dk ).transpose(1,2)

        attention_output,_ = scaled_dot_product_attention(Q, K, V, mask)
        # shape is batch_size, num_heads, seq_len, dk 
        attention_output = attention_output.transpose(1,2)
        attention_output = attention_output.contiguous().view(batch_size,-1, self.embed_size )
        # we need conitguous because the transpose makes the memory non contigous??
        output = self.output_layer(attention_output)
        
        return output
    
class DecoderOnlyBlock(nn.Module):

    def __init__(self, embedding_dim, num_heads, dff, dropout):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.dff = dff 
        self.dropout = dropout 
        self.ffn = nn.Sequential(nn.Linear(embedding_dim, dff), nn.ReLU(), nn.Linear(dff,embedding_dim))
        self.attention = FastMultiHeadAttention(num_heads= 4, embed_size=embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.layernorm1 = nn.LayerNorm(embedding_dim)
        self.layernorm2 = nn.LayerNorm(embedding_dim)
    def forward(self , x, mask ):
        # assuming x is hte embedding here, batch_size, seq_len, embed_dim 
        # mask is 1, 1,, seq_len, seq_len
        residual = x 
        attention_output = self.attention(x, mask)
        
        x = residual+ self.layernorm1(attention_output)
        residual = x 
        FFN_output = self.ffn(x)
        FFN_output = self.layernorm2(FFN_output)
        x = residual + self.dropout(FFN_output)
        return x 