import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()

        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.ln1 = nn.LayerNorm(n_embd)
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):

        B, T, C = x.shape
        h = self.ln1(x)

        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        if attention_mask is not None:
            attn_mask = attention_mask[:, None, None, :].bool()
        else:
            attn_mask = None
            
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,is_causal=False)
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        
        x = x + self.dropout(self.proj(attn))
        x = x + self.mlp(self.ln2(x))
        
        return x

class Text_encoder(nn.Module):
    def __init__(self, n_head=8, n_layer=6, num_embeddings=4001, embedding_dim=512, max_length=64, dropout=0.2):
        super().__init__()
        self.token_embeddings = nn.Embedding(num_embeddings=num_embeddings,embedding_dim=embedding_dim)
        self.positional_embedding = nn.Embedding(num_embeddings=max_length,embedding_dim=embedding_dim)
        self.blocks = nn.ModuleList([TransformerBlock(embedding_dim,n_head,dropout)for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(embedding_dim)
        
    def forward(self, input_ids, attention_mask=None):
        B, T = input_ids.shape
        positions = torch.arange(T,device=input_ids.device).unsqueeze(0)
        x = (self.token_embeddings(input_ids)+self.positional_embedding(positions))
        for block in self.blocks:
            x = block(x, attention_mask)

        x = self.ln_f(x)
        
        return x

if __name__ == "__main__": 
    from Token.TTI_Tokenizer import TTITokenizer as TTIT
    
    tokenizer = TTIT("Token/bpe_tokenizer.json",max_length=64)
    vocab_size = tokenizer.vocab_size

    encoder = Text_encoder(n_head=8,n_layer=6,num_embeddings=vocab_size,embedding_dim=512,max_length=64,dropout=0.2)
    encoded = tokenizer.encode_ids("Red car infront of the house")

    input_ids = torch.tensor(encoded["input_ids"],dtype=torch.long).unsqueeze(0)
    attention_mask = torch.tensor(encoded["attention_mask"],dtype=torch.long).unsqueeze(0)

    attention_mask[:, 9:] = 0
    output = encoder(input_ids,attention_mask)

    print("Input:", input_ids.shape)
    print("Mask:", attention_mask.shape)
    print("Output:", output.shape)