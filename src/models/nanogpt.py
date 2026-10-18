import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head

        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=False)
        self.c_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(C, dim=2)

        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.c_fc = nn.Linear(d_model, 4 * d_model, bias=False)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model, bias=False)
        self.attn = CausalSelfAttention(d_model, n_head)
        self.ln_2 = nn.LayerNorm(d_model, bias=False)
        self.mlp = MLP(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MicroNanoGPT(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, n_layer: int = 2, n_head: int = 4, block_size: int = 64):
        super().__init__()
        self.block_size = block_size

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(vocab_size, d_model),
            wpe=nn.Embedding(block_size, d_model),
            h=nn.ModuleList([Block(d_model, n_head) for _ in range(n_layer)]),
            ln_f=nn.LayerNorm(d_model, bias=False)
        ))
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100
            )

        return logits, loss

    def load_state_dict_flexible(self, state_dict: dict):
        """
        Loads state dict allowing for vocabulary expansion (e.g. when tokens like '[' and ']' are added).
        """
        if "transformer.wte.weight" in state_dict:
            old_wte = state_dict["transformer.wte.weight"]
            old_v = old_wte.shape[0]
            curr_v = self.transformer.wte.weight.shape[0]
            if old_v != curr_v:
                with torch.no_grad():
                    n = min(old_v, curr_v)
                    self.transformer.wte.weight[:n] = old_wte[:n]
                    if "lm_head.weight" in state_dict:
                        self.lm_head.weight[:n] = state_dict["lm_head.weight"][:n]
                state_dict = {k: v for k, v in state_dict.items() if k not in ["transformer.wte.weight", "lm_head.weight"]}
        return self.load_state_dict(state_dict, strict=False)