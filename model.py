import math
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
from timm.models.layers import to_2tuple
import numpy as np
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange, repeat

# np.random.seed(0)

def index_reverse(index):
    index_r = torch.zeros_like(index)
    ind = torch.arange(0, index.shape[-1]).to(index.device)
    for i in range(index.shape[0]):
        index_r[i, index[i, :]] = ind
    return index_r


def semantic_neighbor(x, index):
    dim = index.dim()
    assert x.shape[:dim] == index.shape, "x ({:}) and index ({:}) shape incompatible".format(x.shape, index.shape)

    for _ in range(x.dim() - index.dim()):
        index = index.unsqueeze(-1)
    index = index.expand(x.shape)

    shuffled_x = torch.gather(x, dim=dim - 1, index=index)
    return shuffled_x


class ASSM(nn.Module):
    def __init__(self, dim, d_state, num_tokens=64, inner_rank=128, mlp_ratio=2.):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        inner_rank = inner_rank

        # Mamba params
        self.expand = mlp_ratio
        hidden = int(self.dim * self.expand)
        self.d_state = d_state
        self.selectiveScan = Selective_Scan(d_model=hidden, d_state=self.d_state, expand=1)
        self.out_norm = nn.LayerNorm(hidden)
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(hidden, dim, bias=True)

        self.in_proj = nn.Sequential(
            nn.Conv2d(self.dim, hidden, 1, 1, 0),
        )

        self.CPE = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden),
        )

        self.embeddingB = nn.Embedding(self.num_tokens, inner_rank)  # [64,32] [32, 48] = [64,48]
        self.embeddingB.weight.data.uniform_(-1 / self.num_tokens, 1 / self.num_tokens)

        self.route = nn.Sequential(
            nn.Linear(self.dim, self.dim // 3),
            nn.GELU(),
            nn.Linear(self.dim // 3, self.num_tokens),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x, x_size, token):
        B, n, C = x.shape
        H, W = x_size

        full_embedding = self.embeddingB.weight @ token.weight  # [128, C]

        pred_route = self.route(x)  # [B, HW, num_token]
        cls_policy = F.gumbel_softmax(pred_route, hard=True, dim=-1)  # [B, HW, num_token]

        prompt = torch.matmul(cls_policy, full_embedding).view(B, n, self.d_state)

        detached_index = torch.argmax(cls_policy.detach(), dim=-1, keepdim=False).view(B, n)  # [B, HW]
        x_sort_values, x_sort_indices = torch.sort(detached_index, dim=-1, stable=False)
        x_sort_indices_reverse = index_reverse(x_sort_indices)

        x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x = self.in_proj(x)
        x = x * torch.sigmoid(self.CPE(x))
        cc = x.shape[1]
        x = x.view(B, cc, -1).contiguous().permute(0, 2, 1)  # b,n,c

        semantic_x = semantic_neighbor(x, x_sort_indices) # SGN-unfold
        y = self.selectiveScan(semantic_x, prompt)
        y = self.out_proj(self.out_norm(y))
        x = semantic_neighbor(y, x_sort_indices_reverse) # SGN-fold

        return x


class Selective_Scan(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True)  # (K=4, D, N)
        self.selective_scan = selective_scan_fn

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor, prompt):
        B, L, C = x.shape
        K = 1  
        xs = x.permute(0, 2, 1).view(B, 1, C, L).contiguous()  # B, 1, C ,L

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        #  our ASE here ---
        Cs = Cs.float().view(B, K, -1, L) + prompt  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        return out_y[:, 0]

    def forward(self, x: torch.Tensor, prompt, **kwargs):
        b, l, c = prompt.shape
        prompt = prompt.permute(0, 2, 1).contiguous().view(b, 1, c, l)
        y = self.forward_core(x, prompt)  # [B, L, C]
        y = y.permute(0, 2, 1).contiguous()
        return y


class DWConv(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
        super().__init__()

        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=in_ch, bias=bias
        )

        self.pointwise = nn.Conv2d(
            in_ch, out_ch, 1, stride=1, padding=0, bias=bias
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


def sam_sim(x1: torch.Tensor, x2: torch.Tensor):

    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)

    sim = torch.matmul(x1, x2.transpose(-2, -1))
    cos = sim.clamp(-1,1)
    return 1 - cos.div(np.pi).acos() / np.pi


class Cluster(nn.Module):
    def __init__(self,
                 dim,
                 upscale,
                 windows,
                 heads=1,):
        super().__init__()

        self.head_dim = 32
        self.dim = dim
        self.heads = heads
        self.upscale = upscale
        self.out_dim = dim
        self.windows = windows

        self.f = nn.Conv2d(self.dim, self.heads * self.head_dim, kernel_size=1)  # for similarity
        self.v = nn.Conv2d(self.dim, self.heads * self.head_dim, kernel_size=1)  # for value
        self.out_proj = nn.Conv2d(self.heads * self.head_dim, self.out_dim, kernel_size=1)
        self.centers_proposal = nn.AvgPool2d(kernel_size=self.upscale, stride=self.upscale)
        self.sim_alpha = nn.Parameter(torch.ones(1))
        self.sim_beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        # Window quantity
        assert W % self.upscale == 0 and H % self.upscale == 0,\
                f"Ensure the feature map size ({W}*{H}) can be divided by {self.windows}*{self.windows}"
        H_windows = H // self.windows
        W_windows = W // self.windows

        value = self.v(x)
        x = self.f(x)

        # (8,32,64,64)
        x = rearrange(x, "b (e c) h w -> (b e) c h w", e=self.heads)
        value = rearrange(value, "b (e c) h w -> (b e) c h w", e=self.heads)

        # Splitting the whole (H,W) feature map evenly into H_windows×W_windows small windows
        if H_windows > 1 and W_windows > 1:
            x = rearrange(x, "b c (h_wins h) (w_wins w) -> (b h_wins w_wins) c h w",
                          h_wins=H_windows, w_wins=W_windows)
            value = rearrange(value, "b c (h_wins h) (w_wins w) -> (b h_wins w_wins) c h w",
                          h_wins=H_windows, w_wins=W_windows)

        # In each (split) small window, pool to a series of points
        centers = self.centers_proposal(x)
        # value_centers = rearrange(self.centers_proposal(value), 'b c h w -> b (h w) c')
        b, c, hh, ww = centers.shape

        # Calculate the similarity weight of each point in the window to each center
        sim = torch.sigmoid(
            self.sim_beta +
            self.sim_alpha * sam_sim(
                centers.reshape(b, c, -1).permute(0, 2, 1),
                x.reshape(b, c, -1).permute(0, 2, 1)
            )
        )  # [B,M,N]

        # sim_max_idx = sim.argmax(dim=1, keepdim=True)
        # mask = torch.zeros_like(sim)  # binary #[B,M,N]
        # mask.scatter_(1, sim_max_idx, 1.)
        # sim = sim * mask

        # Cluster points in each window to the most similar point
        value2 = rearrange(value, 'b c h w -> b (h w) c')  # [B,N,D]
        out = ((torch.einsum('bmn,bnd->bmd', sim, value2))
               / (sim.sum(dim=-1, keepdim=True) + 1e-6))  # Weighted average of pixel features belonging to the same center
        # Reconstruct the small windows together
        out = rearrange(out, "(b e h_wins w_wins) (h w) c -> b (c e) (h h_wins) (w w_wins)",
                        w=ww, h=hh, h_wins=H_windows, w_wins=W_windows, e=self.heads)
        # Multi-head channels return to original channel size
        out = self.out_proj(out) # b c h w

        return out


class ChannelAttn(nn.Module):
    def __init__(self, C, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),          # (B,C,1,1)
            nn.Flatten(1),                    # (B,C)
            nn.Linear(C, hidden),
            nn.GELU(),
            nn.Linear(hidden, C),
            nn.Sigmoid()                      # Ensure weights are 0~1
        )

    def forward(self, x):
        # x: (B,C,H,W)
        omega = self.mlp(x)                  # (B,C)  0~1 channel importance
        x_out = x * omega.view(-1, x.size(1), 1, 1)  # Channel weighting
        return omega, x_out                  # Return weighted feature + weights


class GibbsChannelDown(nn.Module):
    def __init__(self,
                 C,               # Input channel count
                 C_down=64,       # Dimensionally reduced channel count
                 n_step=20,       # Gibbs total steps
                 n_parallel=16,   # Number of channels sampled in parallel per step
                 lam=20.0):       # Importance penalty
        super().__init__()
        self.n_step = n_step
        self.n_parallel = n_parallel
        self.lam = lam

        self.score = ChannelAttn(C)           # Importance network
        self.c_down = nn.Conv2d(C, C_down, 1, bias=False)

    # ------------------------------------------------------
    # 2.1 Parallel Gibbs Sampling one update
    # ------------------------------------------------------
    def gibbs_step_parallel(self, omega, mask):
        """
        One parallel Gibbs update
        Parameters:
            omega: (N,C)  Importance score
            mask:  (N,C)  Current 0/1 state
        Returns:
            new_mask: (N,C)  Updated 0/1 state
        """
        N, C = mask.shape
        device = mask.device

        # 1) Randomly select n_parallel non-repeating channels (N, n_parallel)
        #    Each row is a random permutation of 0~C-1, taking the first n_parallel
        candidate = torch.stack([torch.randperm(C, device=device)[:self.n_parallel]
                                for _ in range(N)])

        # 2) Extract current state of these channels (N, n_parallel)
        curr = mask[torch.arange(N)[:, None], candidate]   # 0 or 1
        omega_ij = omega[torch.arange(N)[:, None], candidate]  # Corresponding importance

        # 3) Energy: High-information channels tend to remain/become 1
        #   curr=1: +omega  → hard to flip to 0
        #   curr=0: -omega  → easy to flip to 1
        energy_flip = self.lam * omega_ij * (2 * curr - 1)  # (N, n_parallel)

        # 4) Flip probability P(flip) = sigmoid(-energy_flip)
        # log_P_flip = -energy_flip
        P_flip = torch.sigmoid(energy_flip)  # (N, n_parallel)
        # print('P_flip:{}'.format(P_flip))

        # 6) Bernoulli sampling
        rand_u = torch.rand_like(P_flip)
        flip = (rand_u > P_flip).float()          # 1 means need to flip

        # 7) Write back to mask
        #    new_state = flip*(1-curr) + (1-flip)*curr
        new_state = flip * (1 - curr) + (1 - flip) * curr
        mask = mask.clone()
        mask[torch.arange(N)[:, None], candidate] = new_state
        return mask

    # ------------------------------------------------------
    # 2.2 forward
    # ------------------------------------------------------
    def forward(self, x):
        """
        x: (N,C,H,W)
        return: (N,C_down,H,W)
        """
        N, C, H, W = x.shape
        device = x.device

        # 1) Channel importance (N,C)
        omega, x_out = self.score(x)
        # print('omega:{}'.format(omega))

        # 2) Initial all open
        mask = torch.ones(N, C, device=device)

        # 3) Run n_step parallel Gibbs
        for _ in range(self.n_step):
            mask = self.gibbs_step_parallel(omega, mask)

        # 4) Apply mask to feature
        x = x * mask.view(N, C, 1, 1)

        # 5) 1v1 Dimensionality reduction
        return self.c_down(x), x_out, mask


class DownBranch(nn.Module):
    def __init__(self, dim, d_state, inner_rank, num_tokens, mlp_ratio):
        super().__init__()
        self.patch_embed = PatchEmbed()
        self.patch_unembed = PatchUnEmbed(embed_dim=dim)
        self.convd = nn.Sequential(nn.Conv2d(dim * 2, dim * 2, 1, 1, 0),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                   nn.Dropout2d(0.2),
                                   DWConv(dim * 2, dim * 2),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                   nn.Dropout2d(0.2),
                                   nn.Conv2d(dim * 2, dim, 1, 1, 0))


        self.attn = ASSM(
                        dim=dim,
                        d_state=d_state,
                        inner_rank=inner_rank,
                        num_tokens=num_tokens,
                        mlp_ratio=mlp_ratio,
                        )
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout2d(0.2)

        self.embeddingA = nn.Embedding(inner_rank, d_state)
        self.embeddingA.weight.data.uniform_(-1 / inner_rank, 1 / inner_rank)

    def forward(self, x, mask=None):
        shortcut = x
        if mask is not None:
            x = x * mask.view(mask.shape[0], mask.shape[1], 1, 1)
        x_size = (x.shape[2], x.shape[3])
        x_embed = self.patch_embed(x)
        x_embed = self.attn(self.norm(x_embed), x_size, self.embeddingA)
        x = self.drop(self.patch_unembed(x_embed, x_size))
        x = torch.cat((x, shortcut), dim=1)
        x = self.convd(x)
        x = x + shortcut
        return x


class UpBranch(nn.Module):
    def __init__(self, dim, d_state, inner_rank, num_tokens, mlp_ratio, n_sample, lamuda):
        super().__init__()
        self.patch_embed = PatchEmbed()
        self.dim_down = dim//8
        self.patch_unembed = PatchUnEmbed(embed_dim=self.dim_down)
        self.convu = nn.Sequential(nn.Conv2d(dim + self.dim_down, dim+ self.dim_down, 1, 1, 0),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                   nn.Dropout2d(0.2),
                                   DWConv(dim+ self.dim_down, dim+ self.dim_down),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                   nn.Dropout2d(0.2),
                                   nn.Conv2d(dim+ self.dim_down, dim, 1, 1, 0))

        self.norm = nn.LayerNorm(self.dim_down)

        self.attn = ASSM(
            dim=self.dim_down,
            d_state=d_state,
            inner_rank=inner_rank,
            num_tokens=num_tokens,
            mlp_ratio=mlp_ratio,
        )

        self.embeddingA = nn.Embedding(inner_rank, d_state)
        self.embeddingA.weight.data.uniform_(-1 / inner_rank, 1 / inner_rank)
        self.drop = nn.Dropout2d(0.2)

        self.MCdown = GibbsChannelDown(C=dim, C_down=self.dim_down,
                          n_step=5, n_parallel=n_sample, lam=lamuda)

    def forward(self, x):
        shortcut = x
        x_size = (x.shape[2], x.shape[3])

        x_down, x_channel, mask = self.MCdown(x)

        x_embed = self.patch_embed(x_down)
        x_embed = self.attn(self.norm(x_embed), x_size, self.embeddingA)
        x = self.drop(self.patch_unembed(x_embed, x_size))
        x = torch.cat((x, x_channel), dim=1)
        x = self.convu(x)
        x = x + shortcut
        return x, mask


class S2C(nn.Module):
    def __init__(self, dim, upscale, d_state, inner_rank, num_tokens, mlp_ratio, n_sample, lamuda):
        super().__init__()

        self.upsample = Upsample(scale=upscale, num_feat=dim)
        self.upbranch = UpBranch(dim, d_state, inner_rank, num_tokens, mlp_ratio, n_sample, lamuda)

    def forward(self, x, skip=None):
        xup = self.upsample(x)
        if skip is not None:
            xup = xup + skip
        x, mask = self.upbranch(xup)
        return x, mask


class S2M(nn.Module):
    def __init__(self, dim, upscale, d_state, inner_rank, num_tokens, mlp_ratio):
        super().__init__()

        self.cluster = Cluster(dim=dim, upscale=upscale, windows=8, heads=4)
        self.downbranch = DownBranch(dim, d_state, inner_rank, num_tokens, mlp_ratio)

    def forward(self, x, skip, mask=None):
        if mask is not None:
            x = x * mask.view(mask.shape[0], mask.shape[1], 1, 1)
        xdown = self.cluster(x) + skip
        x = self.downbranch(xdown)
        return x


class S2CDBlock(nn.Module):
    def __init__(self, dim, upscale, d_state, inner_rank, num_tokens, mlp_ratio, n_sample, lamuda):
        super(S2CDBlock, self).__init__()

        self.s2c = S2C(dim, upscale, d_state, inner_rank, num_tokens, mlp_ratio, n_sample, lamuda)
        self.s2m = S2M(dim, upscale, d_state, inner_rank, num_tokens, mlp_ratio)

    def forward(self, x):

        x1, mask1 = self.s2c(x)
        x2 = self.s2m(x1, x, mask=mask1)

        x3, mask2 = self.s2c(x2, skip=x1)
        x4 = self.s2m(x3, x2, mask=mask2)

        x5, _ = self.s2c(x4, skip=x3)
        return x5


class PatchEmbed(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])
        return x


class Upsample(nn.Sequential):
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class S2CD_Net(nn.Module):
    def __init__(self, inch, dim, upscale, d_state, inner_rank, num_tokens, mlp_ratio, n_sample, lamuda):
        super(S2CD_Net, self).__init__()
        self.conv_first = nn.Conv2d(inch, dim, 3, 1, 1)
        self.blockup = S2CDBlock(dim=dim,
                               upscale=upscale,
                               d_state=d_state,
                               inner_rank=inner_rank,
                               num_tokens=num_tokens,
                               mlp_ratio=mlp_ratio,
                               n_sample=n_sample,
                               lamuda=lamuda)
        self.conv_last = nn.Conv2d(dim, inch, 3, 1, 1)

    def forward(self, x):

        x = self.conv_first(x)
        x = self.blockup(x)
        x = self.conv_last(x)
        return x


Final_Model = S2CD_Net


if __name__ == '__main__':

    upscale = 4
    inch_dim = 128
    dddim = 256

    model = S2CD_Net(inch=inch_dim,
                        dim=dddim,
                        upscale=upscale,
                        d_state=16,
                        inner_rank=64,
                        num_tokens=128,
                        mlp_ratio=2.0,
                        n_sample=64,
                        lamuda=1.11).to('cuda')

    input_test = torch.randn((1, inch_dim, 16, 16)).cuda()
    qqq = 1

    if qqq == 1:

        print('Start testing model complexity')
        from thop import profile

        flops, params = profile(model, inputs=(input_test,))

        print("FLOPs={:.2f}G".format(flops / 1e9))
        print("Params={:.2f}M".format(params / 1e6))


        def cleanup_thop(model):
            for module in model.modules():
                # Remove attributes added by thop
                for attr in ['total_ops', 'total_params']:
                    if hasattr(module, attr):
                        delattr(module, attr)
                # Force clearing of all hooks
                if hasattr(module, '_forward_hooks'):
                    module._forward_hooks.clear()
                if hasattr(module, '_forward_pre_hooks'):
                    module._forward_pre_hooks.clear()
                if hasattr(module, '_backward_hooks'):
                    module._backward_hooks.clear()


        cleanup_thop(model)

        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        repetitions = 100
        timings = np.zeros((repetitions, 1))

        # MEASURE PERFORMANCE
        with torch.no_grad():
            for rep in range(repetitions):
                if (rep + 1) % 50 == 0:
                    print('Current round is {}'.format(rep + 1))
                starter.record()
                _ = model(input_test)
                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[rep] = curr_time
                
        mean_syn = np.sum(timings) / repetitions
        mean_fps = 1000. / mean_syn
        print('with_no_grad train FPS: {mean_fps:.2f}'.format(mean_fps=mean_fps))

