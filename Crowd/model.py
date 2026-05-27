from __future__ import annotations

import torch
from torch import nn


def aggregate_messages(messages: torch.Tensor, dst_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    out = torch.zeros(num_nodes, messages.size(-1), device=messages.device)
    out.index_add_(0, dst_index, messages)
    counts = torch.zeros(num_nodes, 1, device=messages.device)
    ones = torch.ones(dst_index.size(0), 1, device=messages.device)
    counts.index_add_(0, dst_index, ones)
    return out / counts.clamp(min=1.0)


def expand_edge_index(edge_index: torch.Tensor, batch_size: int, n_nodes: int) -> torch.Tensor:
    src, dst = edge_index
    offsets = (torch.arange(batch_size, device=edge_index.device) * n_nodes).view(-1, 1)
    src = src.view(1, -1) + offsets
    dst = dst.view(1, -1) + offsets
    return torch.stack([src.reshape(-1), dst.reshape(-1)], dim=0)


def expand_bipartite_edge_index(edge_index: torch.Tensor, batch_size: int, n_src: int, n_dst: int) -> torch.Tensor:
    src, dst = edge_index
    src_offsets = (torch.arange(batch_size, device=edge_index.device) * n_src).view(-1, 1)
    dst_offsets = (torch.arange(batch_size, device=edge_index.device) * n_dst).view(-1, 1)
    src = src.view(1, -1) + src_offsets
    dst = dst.view(1, -1) + dst_offsets
    return torch.stack([src.reshape(-1), dst.reshape(-1)], dim=0)


class InternalLayer(nn.Module):
    def __init__(self, hidden_dim: int, org_in_dim: int, dropout_rate: float = 0.3):
        super().__init__()
        self.msg = nn.Sequential(
            nn.Linear(hidden_dim * 2 + org_in_dim + 2, hidden_dim),
            nn.Tanh(),
            nn.Dropout(p=dropout_rate),  # 과적합 방지
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(p=dropout_rate),  # 과적합 방지
        )
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, h: torch.Tensor, u: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        msg_in = torch.cat([h[dst], h[src], u[dst] - u[src], pos[dst] - pos[src]], dim=-1)
        messages = self.msg(msg_in)
        agg = aggregate_messages(messages, dst, h.size(0))
        update = self.update(torch.cat([h, agg], dim=-1))
        return h + update


class ExternalLayer(nn.Module):
    def __init__(self, hidden_dim: int, ex_in_dim: int, dropout_rate: float = 0.3):
        super().__init__()
        self.ex_embed = nn.Sequential(
            nn.Linear(ex_in_dim + 2, hidden_dim),
            nn.Tanh(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(p=dropout_rate),
        )
        self.msg = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 2, hidden_dim),
            nn.Tanh(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(p=dropout_rate),
        )
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(p=dropout_rate),
        )

    def forward(
        self,
        h: torch.Tensor,
        ex_x: torch.Tensor,
        pos: torch.Tensor,
        ex_pos: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        src, dst = edge_index
        ex_emb = self.ex_embed(torch.cat([ex_x, ex_pos], dim=-1))
        msg_in = torch.cat([h[dst], ex_emb[src], pos[dst] - ex_pos[src]], dim=-1)
        messages = self.msg(msg_in)
        agg = aggregate_messages(messages, dst, h.size(0))
        update = self.update(torch.cat([h, agg], dim=-1))
        return h + update


class CrowdMPNN(nn.Module):
    def __init__(
        self,
        back_steps: int,
        hidden_dim: int = 128,
        n_passing: int = 4,
        dropout_rate: float = 0.3,  # 하이퍼파라미터 추가
    ):
        super().__init__()
        self.back_steps = int(back_steps)
        self.hidden_dim = int(hidden_dim)
        self.n_passing = int(n_passing)

        in_dim = self.back_steps
        self.embed = nn.Sequential(
            nn.Linear(in_dim + 2, hidden_dim),
            nn.Tanh(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(p=dropout_rate),
        )
        self.external1 = ExternalLayer(hidden_dim, ex_in_dim=in_dim, dropout_rate=dropout_rate)
        self.internal_layers = nn.ModuleList(
            [InternalLayer(hidden_dim, org_in_dim=in_dim, dropout_rate=dropout_rate) for _ in range(self.n_passing)]
        )
        self.external2 = ExternalLayer(hidden_dim, ex_in_dim=in_dim, dropout_rate=dropout_rate)
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        cctv_x: torch.Tensor,
        skt_x: torch.Tensor,
        cctv_pos: torch.Tensor,
        skt_pos: torch.Tensor,
        edge_index_cctv: torch.Tensor,
        edge_index_skt2cctv: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, n_cctv, hist_len, _ = cctv_x.shape
        _, n_skt, _, _ = skt_x.shape

        cctv_x = cctv_x.view(batch_size * n_cctv, hist_len)
        skt_x = skt_x.view(batch_size * n_skt, hist_len)

        cctv_pos = cctv_pos.view(1, n_cctv, 2).repeat(batch_size, 1, 1).view(batch_size * n_cctv, 2)
        skt_pos = skt_pos.view(1, n_skt, 2).repeat(batch_size, 1, 1).view(batch_size * n_skt, 2)

        edge_index_cctv = expand_edge_index(edge_index_cctv, batch_size, n_cctv)
        edge_index_skt2cctv = expand_bipartite_edge_index(edge_index_skt2cctv, batch_size, n_skt, n_cctv)

        h = self.embed(torch.cat([cctv_x, cctv_pos], dim=-1))
        h = self.external1(h, skt_x, cctv_pos, skt_pos, edge_index_skt2cctv)
        for layer in self.internal_layers:
            h = layer(h, cctv_x, cctv_pos, edge_index_cctv)
        h = self.external2(h, skt_x, cctv_pos, skt_pos, edge_index_skt2cctv)
        out = self.out(h).view(batch_size, n_cctv, 1)
        return out