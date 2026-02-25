



# class MultiHeadAttention(nn.Module):
#     def __init__(
#         self,
#         dim: int,
#         n_head: int,
#         dropout: float = 0.0,
#         rel_pos_bias: bool = False,
#     ):
#         super().__init__()
#         self.n_head = n_head
#         self.head_size = dim // n_head
#         self.scale = math.sqrt(self.head_size)

#         self.qkv = nn.Linear(dim, 3 * dim, False)
#         self.proj = nn.Linear(dim, dim, False)
#         self.attn_dropout = nn.Dropout(dropout)
#         self.resid_dropout = nn.Dropout(dropout)

#         self.rel_pos_bias = rel_pos_bias
#         # if rel_pos_bias:
#         #     self.rel_pos_forward = nn.Embedding(10, self.n_head, padding_idx=9)
#         #     self.rel_pos_backward = nn.Embedding(10, self.n_head, padding_idx=9)

#     def forward(self, x: Tensor, adj: Optional[Tensor] = None) -> Tensor:
#         B, L, C = x.shape

#         query, key, value = self.qkv(x).chunk(3, -1)
#         query = query.view(B, L, self.n_head, self.head_size).transpose(1, 2)
#         key = key.view(B, L, self.n_head, self.head_size).transpose(1, 2)
#         value = value.view(B, L, self.n_head, self.head_size).transpose(1, 2)
#         score = torch.matmul(query, key.mT) / self.scale

#         if self.rel_pos_bias:
#             adj = adj.masked_fill(torch.logical_and(adj > 1, adj < 9), 0)
#             adj = adj.masked_fill(adj != 0, 1)
#             adj = adj.float()
#             # pe = torch.stack([adj], dim=1).repeat(1, self.n_head // 1, 1, 1)
#             # pe = torch.stack([adj.mT], dim=1).repeat(1, self.n_head // 1, 1, 1)
#             # pe = torch.stack([adj, adj.mT], dim=1).repeat(1, self.n_head // 2, 1, 1)
#             # pe = torch.stack([adj, adj.mT, adj @ adj, adj.mT @ adj.mT], dim=1)
#             pe = torch.stack([adj, adj.mT, adj.mT @ adj, adj @ adj.mT], dim=1)
#             pe = pe + torch.eye(L, dtype=adj.dtype, device=adj.device)
#             pe = pe.int()

#             # pe = (
#             #     self.rel_pos_forward(rel_pos) + self.rel_pos_backward(rel_pos.mT)
#             # ).permute(0, 3, 1, 2)
#             # score = score * (1 + pe)
#             score = score.masked_fill(pe == 0, -torch.inf)

#         attn = F.softmax(score, dim=-1)
#         attn = self.attn_dropout(attn)  # (b, n_head, l_q, l_k)
#         x = torch.matmul(attn, value)
#         x = x.transpose(1, 2).reshape(B, L, C)
#         return self.resid_dropout(self.proj(x))

#     def extra_repr(self) -> str:
#         return f"n_head={self.n_head}"


# class SelfAttentionBlock(nn.Module):
#     def __init__(
#         self,
#         dim: int,
#         n_head: int,
#         dropout: float,
#         droppath: float,
#         rel_pos_bias: bool = False,
#     ):
#         super().__init__()
#         self.norm = nn.LayerNorm(dim)
#         # The larger the dataset, the better rel_pos_bias works
#         # probably due to the overfitting of rel_pos_bias
#         self.attn = MultiHeadAttention(dim, n_head, dropout, rel_pos_bias=rel_pos_bias)

#     def forward(self, x: Tensor, rel_pos: Optional[Tensor] = None) -> Tensor:
#         x_ = self.norm(x)
#         x_ = self.attn(x_, rel_pos)
#         return x_ + x



# class FeedForwardBlock(nn.Module):
#     def __init__(
#         self,
#         dim: int,
#         mlp_ratio: float,
#         act_layer: str,
#         dropout: float,
#         droppath: float,
#         gcn: bool = False,
#     ):
#         super().__init__()

#         self.norm = nn.LayerNorm(dim)
#         self.mlp = GINMlp(dim, mlp_ratio, act_layer=act_layer, drop=dropout)

#     def forward(self, x: Tensor, adj: Optional[Tensor] = None) -> Tensor:
#         x_ = self.norm(x)
#         x_ = self.mlp(x_, adj)
#         return x_ + x


# class GINMlp(nn.Module):
#     def __init__(
#         self,
#         dim: int,
#         mlp_ratio: float = 4.0,
#         out_features: Optional[int] = None,
#         act_layer: str = "relu",
#         drop: float = 0.0,
#     ):
#         super().__init__()
#         in_features = dim
#         out_features = out_features or in_features
#         hidden_features = int(mlp_ratio * in_features)
#         drop_probs = to_2tuple(drop)

#         self.fc1 = nn.Linear(in_features, hidden_features, False)
#         self.gcn = nn.Linear(in_features, hidden_features, False)
#         if act_layer.lower() == "relu":
#             self.act = nn.ReLU()
#         elif act_layer.lower() == "leaky_relu":
#             self.act = nn.LeakyReLU()
#         else:
#             raise ValueError(f"Unsupported activation: {act_layer}")
#         self.drop1 = nn.Dropout(drop_probs[0])
#         self.fc2 = nn.Linear(hidden_features, out_features, False)
#         self.drop2 = nn.Dropout(drop_probs[1])

#     def forward(self, x: Tensor, adj: Tensor) -> Tensor:
#         out = self.fc1(x)
#         gcn_x1, gcn_x2 = self.gcn(x).chunk(2, dim=-1)
#         out = out + torch.cat([adj @ gcn_x1, adj.mT @ gcn_x2], dim=-1)
#         out = self.act(out)
#         out = self.drop1(out)
#         out = self.fc2(out)
#         out = self.drop2(out)
#         return out


# class EncoderBlock(nn.Module):
#     def __init__(
#         self,
#         dim: int,
#         n_head: int,
#         mlp_ratio: float,
#         act_layer: str,
#         dropout: float,
#         droppath: float,
#     ):
#         super().__init__()
#         self.self_attn = SelfAttentionBlock(
#             dim, n_head, dropout, droppath, rel_pos_bias=True
#         )
#         self.feed_forward = FeedForwardBlock(
#             dim, mlp_ratio, act_layer, dropout, droppath, gcn=True
#         )

#     def forward(self, x: Tensor, adj: Tensor) -> Tensor:
#         x = self.self_attn(x, adj)
#         x = self.feed_forward(x, adj)
#         return x


# class nnEncoder(nn.Module):
#     def __init__(self, d_model, d_ff_ratio, gcn_layers):
#         super().__init__()
#         # Encoder stage
#         self.layers = nn.ModuleList()
#         for i in range(12):
#             self.layers.append(EncoderBlock(d_model, 4, 4, "relu", 0.1, 0))

#     def forward(self, x, adj):
#         for i, layer in enumerate(self.layers):
#             x = layer(x, adj)
#         return x
