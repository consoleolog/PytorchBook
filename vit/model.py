import torch
import torch.nn as nn
nn.MultiheadAttention()

class LinearProjection(nn.Module):
    def __init__(
        self,
        patch_vec_size, # P^2 * C
        num_patches,    # (H * W) / P^2 (N을 의미)
        latent_vec_dim, # D
        drop_rate,
    ):
        super().__init__()
        self.linear = nn.Linear(patch_vec_size, latent_vec_dim) # N x D
        self.cls_token = nn.Parameter(torch.randn(1, latent_vec_dim)) # 1 x D
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, latent_vec_dim)) # 1 x (N + 1) x D
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        batch_size = x.shape[0]

        x = torch.cat([
            # cls_token.shape (1, D)
            self.cls_token.repeat(batch_size, 1, 1),
            # linear (N, D)
            self.linear(x),
        ], dim=1) # 1 x D

        x += self.pos_embedding
        out = self.dropout(x)
        return out

class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        latent_vec_dim, # D
        num_heads,
        drop_rate,
    ):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_heads = num_heads
        self.latent_vec_dim = latent_vec_dim

        # 학습 가능한 임의의 파라미터
        self.head_dim = int(latent_vec_dim / num_heads)

        # Q, K, V 는 입력과 출력이 동일해야함
        self.Q = nn.Linear(latent_vec_dim, latent_vec_dim)
        self.K = nn.Linear(latent_vec_dim, latent_vec_dim)
        self.V = nn.Linear(latent_vec_dim, latent_vec_dim)

        # 분모가 0 으로 가는걸 방지하기 위한 일종의 트릭
        self.scale = torch.sqrt(self.head_dim * torch.ones(1).to(device))
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        # x.shape (B, N + 1, D)
        batch_size = x.shape[0]

        Q = self.Q(x) # shape (B, N + 1, D)
        K = self.K(x) # shape (B, N + 1, D)
        V = self.V(x) # shape (B, N + 1, D)

        Q = (Q
             .view(batch_size, -1, self.num_heads, self.head_dim)
             .permute(0, 2, 1, 3)
        ) # shape (B, num_head, N + 1, head_dim)

        K = (K
             .view(batch_size, -1, self.num_heads, self.head_dim)
             .permute(0, 2, 3, 1)
        ) # shape (B, num_head, N + 1, head_dim)

        V = (V
             .view(batch_size, -1, self.num_heads, self.head_dim)
             .permute(0, 2, 1, 3)
        ) # shape (B, num_head, N + 1, head_dim)

        # attention 연산
        attention = torch.softmax(Q @ K / self.scale, dim=-1) # shape (B, num_head, N+1, N+1)
        x = self.dropout(attention) @ V # shape (B, num_head, N+1, num_head)
        x = (x.permute(0, 2, 1, 3)
             .reshape(batch_size, -1, self.latent_vec_dim)
        ) # shape (B, N + 1, num_head * num_head)
        return x, attention

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        latent_vec_dim,
        num_heads,
        mlp_hidden_dim,
        drop_rate,
    ):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(latent_vec_dim)
        self.layer_norm_2 = nn.LayerNorm(latent_vec_dim)

        self.MSA = MultiHeadSelfAttention(
            latent_vec_dim=latent_vec_dim,
            num_heads=num_heads,
            drop_rate=drop_rate,
        )
        self.MLP = nn.Sequential(
            nn.Linear(latent_vec_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(mlp_hidden_dim, latent_vec_dim),
            nn.Dropout(drop_rate),
        )

        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        # x.shape (B, N+1, D)
        z = self.layer_norm_1(x)
        z, attention = self.MSA(z)

        z = self.dropout(z)

        x = x + z
        z = self.layer_norm_2(z)
        z = self.MLP(z)

        out = x + z
        return out, attention

class VisionTransformer(nn.Module):
    def __init__(
        self,
        patch_vec_size, # P^2 * C
        num_patches, # (H * W) / P^2 (N 을 의미)
        latent_vec_dim, # D
        num_heads,
        mlp_hidden_dim,
        drop_rate,
        num_layers,
        num_classes,
    ):
        super().__init__()

        self.PatchEmbedding = LinearProjection(
            patch_vec_size=patch_vec_size,
            num_patches=num_patches,
            latent_vec_dim=latent_vec_dim,
            drop_rate=drop_rate,
        )

        # 레이어 반복
        self.transformer = nn.ModuleList([
            TransformerEncoder(
                latent_vec_dim=latent_vec_dim,
                num_heads=num_heads,
                mlp_hidden_dim=mlp_hidden_dim,
                drop_rate=drop_rate,
            ) for _ in range(num_layers)
        ])

        self.MLPHead = nn.Sequential(
            nn.LayerNorm(latent_vec_dim),
            nn.Linear(latent_vec_dim, num_classes),
        )

    def forward(self, x):
        # x.shape (B, N, P^2 * C)
        att_list = []
        x = self.PatchEmbedding(x) # x.shape (B, N+1, D)
        for layer in self.transformer:
            x, att = layer(x)
            # x.shape (B, N+1, num_head )
            att_list.append(att)
        out = self.MLPHead(x[:, 0]) # CLS 토큰의 앞부분만 입력
        # shape (B, num_classes)
        return out, att_list