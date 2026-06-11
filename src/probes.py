import math
import torch
import torch.nn as nn


class Probe(nn.Module):
    def __init__(self, y_dim, out_dim, condition_mask=None, use_conditioned_query=True):
        super().__init__()
        self.use_conditioned_query = bool(use_conditioned_query)
        if condition_mask is None:
            # upper-triangular over y_dim, shifted one column right
            # mask[i, k] = 1 if i < k else 0
            mask = torch.zeros(y_dim, out_dim)
            for i in range(y_dim):
                for k in range(out_dim):
                    if i < k:
                        mask[i, k] = 1.0
            self.condition_mask = mask
        else:
            self.condition_mask = condition_mask  # (y_dim, out_dim)

    def apply_condition_mask(self, y_gt):
        """
        y_gt: (B, y_dim) in {0,1}
        returns:
          y: (B, y_dim, out_dim) where
             y[:, i, k] = (y_gt[:, i] - 0.5) if mask[i,k]=1 else 0
        """
        if self.condition_mask.device != y_gt.device:
            self.condition_mask = self.condition_mask.to(y_gt.device)
        # (B, y_dim, 1) and (1, y_dim, out_dim) -> (B, y_dim, out_dim)
        # y = self.condition_mask * y_gt.unsqueeze(-1) + (1 - self.condition_mask) * 0.5
        # Add batch dimension to condition_mask: (y_dim, out_dim) -> (1, y_dim, out_dim)
        mask = self.condition_mask.unsqueeze(0)
        y = mask * y_gt.unsqueeze(-1) + (1 - mask) * 0.5
        return y - 0.5

    def forward(self, z, y_gt):
        y = self.apply_condition_mask(y_gt)
        return z, y


class LinearProbe(Probe):
    def __init__(self, z_dim, y_dim, out_dim, condition_mask=None, use_conditioned_query=True):
        super().__init__(y_dim, out_dim, condition_mask, use_conditioned_query=use_conditioned_query)
        self.Wz = nn.Linear(z_dim, out_dim, bias=False)
        self.Wy = nn.Linear(y_dim, out_dim, bias=False)  # weights: (out_dim, y_dim)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, z, y_gt):
        # z: (B, z_dim)
        # y_gt: (B, y_dim)
        # out: (B, out_dim)
        z_term = self.Wz(z)  # (B, out_dim)
        if not self.use_conditioned_query:
            return z_term + self.bias

        y = self.apply_condition_mask(y_gt)  # (B, y_dim, out_dim)
        # y_term[b,k] = sum_i y[b,i,k] * Wy.weight[k,i]
        y_term = torch.einsum('bik,ki->bk', y, self.Wy.weight)

        return z_term + y_term + self.bias


class DNNConcatProbe(Probe):
    """
    Per-head MLP probe over concatenated [z, conditioned_y].

    For each head k, build:
      x_k = concat(z, y[:, :, k])  in R^(z_dim + y_dim)
    and apply a head-specific MLP to produce one logit.
    """
    def __init__(
        self,
        z_dim,
        y_dim,
        out_dim,
        hidden_dim=None,
        num_layers=3,
        dropout=0.0,
        condition_mask=None,
        use_conditioned_query=True,
    ):
        super().__init__(y_dim, out_dim, condition_mask, use_conditioned_query=use_conditioned_query)

        if num_layers is None or int(num_layers) <= 0:
            raise ValueError(f"num_layers must be a positive integer for dnn_concat (got {num_layers}).")
        if float(dropout) < 0.0 or float(dropout) >= 1.0:
            raise ValueError(f"dropout must be in [0, 1) for dnn_concat (got {dropout}).")

        self.z_dim = int(z_dim)
        self.y_dim = int(y_dim)
        self.out_dim = int(out_dim)
        self.in_dim = self.z_dim + (self.y_dim if self.use_conditioned_query else 0)
        self.num_layers = int(num_layers)
        self.dropout_p = float(dropout)

        if hidden_dim is None:
            hidden_dim = max(128, min(1024, self.in_dim))
        if self.num_layers > 1 and int(hidden_dim) <= 0:
            raise ValueError(
                f"hidden_dim must be a positive integer for multi-layer dnn_concat (got {hidden_dim})."
            )
        self.hidden_dim = int(hidden_dim)

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(self.dropout_p)

        if self.num_layers == 1:
            # Single linear readout per head.
            self.W_out = nn.Parameter(torch.randn(self.in_dim, self.out_dim) * 0.02)
            self.b_out = nn.Parameter(torch.zeros(self.out_dim))
            self.W_hidden = None
            self.b_hidden = None
        else:
            # First projection: in_dim -> hidden_dim (per head).
            self.W_in = nn.Parameter(torch.randn(self.in_dim, self.hidden_dim, self.out_dim) * 0.02)
            self.b_in = nn.Parameter(torch.zeros(self.hidden_dim, self.out_dim))

            # Optional middle hidden layers, each hidden_dim -> hidden_dim (per head).
            self.W_hidden = nn.ParameterList(
                [
                    nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim, self.out_dim) * 0.02)
                    for _ in range(self.num_layers - 2)
                ]
            )
            self.b_hidden = nn.ParameterList(
                [nn.Parameter(torch.zeros(self.hidden_dim, self.out_dim)) for _ in range(self.num_layers - 2)]
            )

            # Final projection: hidden_dim -> 1 (per head).
            self.W_out = nn.Parameter(torch.randn(self.hidden_dim, 1, self.out_dim) * 0.02)
            self.b_out = nn.Parameter(torch.zeros(1, self.out_dim))

    def _concat(self, z, y_gt):
        if z.ndim != 2:
            raise ValueError(f"Expected z to be rank-2 (B, z_dim), got shape {tuple(z.shape)}.")
        if z.shape[1] != self.z_dim:
            raise ValueError(f"Expected z.shape[1]=={self.z_dim}, got {z.shape[1]}.")

        z_rep = z.unsqueeze(-1).expand(-1, -1, self.out_dim) # (B, z_dim, out_dim)
        if not self.use_conditioned_query:
            return z_rep
        y = self.apply_condition_mask(y_gt)                  # (B, y_dim, out_dim)
        return torch.cat([z_rep, y], dim=1)                 # (B, in_dim, out_dim)

    def forward(self, z, y_gt):
        """
        z:    (B, z_dim)
        y_gt: (B, y_dim)
        out:  (B, out_dim)
        """
        x = self._concat(z, y_gt)  # (B, in_dim, out_dim)

        if self.num_layers == 1:
            out = torch.einsum("bik,ik->bk", x, self.W_out) + self.b_out
            return out

        h = torch.einsum("bik,ihk->bhk", x, self.W_in) + self.b_in
        h = self.activation(h)
        h = self.dropout(h)

        for W, b in zip(self.W_hidden, self.b_hidden):
            h = torch.einsum("bik,ihk->bhk", h, W) + b
            h = self.activation(h)
            h = self.dropout(h)

        out = torch.einsum("bik,iok->bok", h, self.W_out) + self.b_out
        return out.squeeze(1)


class AttentionQuadraticProbe(Probe):
    """
    Conditioned single-query attention probe with a quadratic readout.

    The readout is a per-head low-rank quadratic form over
    concat([CLS, attended], conditioned_y).
    """
    def __init__(
        self,
        z_dim,
        y_dim,
        out_dim,
        token_dim,
        rank,
        attend_cls=False,
        condition_mask=None,
        use_conditioned_query=True,
    ):
        super().__init__(y_dim, out_dim, condition_mask, use_conditioned_query=use_conditioned_query)

        if token_dim is None or int(token_dim) <= 0:
            raise ValueError(f"token_dim must be a positive integer (got {token_dim}).")
        if int(z_dim) % int(token_dim) != 0:
            raise ValueError(f"z_dim={z_dim} must be divisible by token_dim={token_dim}.")
        if rank is None or int(rank) <= 0:
            raise ValueError(f"rank must be a positive integer for attention_quadratic (got {rank}).")

        self.z_dim = int(z_dim)
        self.y_dim = int(y_dim)
        self.out_dim = int(out_dim)
        self.token_dim = int(token_dim)
        self.n_tokens = self.z_dim // self.token_dim
        self.attend_cls = bool(attend_cls)
        self.use_conditioned_query = bool(use_conditioned_query)
        self.rank = int(rank)

        self.z_attn_dim = 2 * self.token_dim  # concat([CLS, attended])
        self.in_dim = self.z_attn_dim + self.y_dim
        self.scale = 1.0 / math.sqrt(self.token_dim)
        self.quad_scale = 1.0 / math.sqrt(self.in_dim)

        if self.n_tokens < 2 and not self.attend_cls:
            raise ValueError("Need at least one spatial token when attend_cls=False.")

        # One learned query per output head, plus conditioning-dependent offset.
        self.query_base = nn.Parameter(torch.randn(self.out_dim, self.token_dim) * 0.02)
        self.query_from_y = nn.Parameter(torch.randn(self.y_dim, self.token_dim) * 0.02)
        self.Wq = nn.Linear(self.token_dim, self.token_dim, bias=False)

        # Per-head low-rank quadratic parameters: (in_dim, rank, out_dim).
        self.W1 = nn.Parameter(torch.randn(self.in_dim, self.rank, self.out_dim) * 0.02)
        self.W2 = nn.Parameter(torch.randn(self.in_dim, self.rank, self.out_dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(self.out_dim))

    def _unflatten_tokens(self, z):
        if z.ndim != 2:
            raise ValueError(f"Expected z to be rank-2 (B, z_dim), got shape {tuple(z.shape)}.")
        if z.shape[1] != self.z_dim:
            raise ValueError(f"Expected z.shape[1]=={self.z_dim}, got {z.shape[1]}.")
        return z.view(z.shape[0], self.n_tokens, self.token_dim)

    def forward(self, z, y_gt):
        """
        z:    (B, z_dim) flattened tokens from output_mode="all"
        y_gt: (B, y_dim)
        out:  (B, out_dim)
        """
        y = self.apply_condition_mask(y_gt)  # (B, y_dim, out_dim)
        tokens = self._unflatten_tokens(z)   # (B, n_tokens, token_dim)

        cls = tokens[:, 0]  # (B, token_dim)
        if self.attend_cls:
            attn_tokens = tokens
        else:
            attn_tokens = tokens[:, 1:]

        # q[b, k, d] = base[k, d] + sum_i y[b, i, k] * query_from_y[i, d]
        q = self.query_base.unsqueeze(0).expand(z.shape[0], -1, -1)
        if self.use_conditioned_query:
            q = q + torch.einsum("bik,id->bkd", y, self.query_from_y)
        q = self.Wq(q)

        # Single-query attention per head over tokens.
        attn_scores = torch.einsum("bkd,bnd->bkn", q, attn_tokens) * self.scale
        attn = torch.softmax(attn_scores, dim=-1)
        pooled = torch.einsum("bkn,bnd->bkd", attn, attn_tokens)  # (B, out_dim, token_dim)

        cls_rep = cls.unsqueeze(1).expand(-1, self.out_dim, -1)
        z_attn = torch.cat([cls_rep, pooled], dim=-1)  # (B, out_dim, 2*token_dim)

        # x[b, :, k] = concat(z_attn[b, k, :], y[b, :, k])
        x = torch.cat([z_attn, y.transpose(1, 2)], dim=-1).transpose(1, 2)  # (B, in_dim, out_dim)

        p1 = torch.einsum("bdk,drk->brk", x, self.W1)  # (B, rank, out_dim)
        p2 = torch.einsum("bdk,drk->brk", x, self.W2)  # (B, rank, out_dim)
        out = torch.einsum("brk,brk->bk", p1, p2) * self.quad_scale + self.bias
        return out


class QuadraticConcatProbe(Probe):
    """
    Symmetric multi-head fixed-rank quadratic probe (within-sample).

    For each sample, we form x = concat(z, y_masked), where y_masked contains
    centered ground-truth labels with a per-head conditioning mask (teacher forcing).
    Unconditioned label dimensions contribute zero.

    For each head k, the probe computes a symmetric low-rank quadratic score
        score_k = x^T W_sym^k x + b_k,
    where W_sym^k = 0.5 * (W1_k^T W2_k + W2_k^T W1_k).

    The probe operates independently on each sample (no cross-sample interactions)
    and outputs a tensor of shape (B, out_dim).
    """
    def __init__(self, z_dim, y_dim, rank, out_dim, condition_mask=None, use_conditioned_query=True):
        super().__init__(y_dim, out_dim, condition_mask, use_conditioned_query=use_conditioned_query)
        self.in_dim = z_dim + (y_dim if self.use_conditioned_query else 0)
        self.rank = rank
        self.out_dim = out_dim
        self.scale = 1.0 / math.sqrt(self.in_dim)

        # (in_dim, rank, out_dim)
        self.W1 = nn.Parameter(torch.randn(self.in_dim, rank, out_dim) * 0.02)
        self.W2 = nn.Parameter(torch.randn(self.in_dim, rank, out_dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def _concat(self, z, y_gt):
        z_rep = z.unsqueeze(-1).expand(-1, -1, self.out_dim) # (B, z_dim, out_dim)
        if not self.use_conditioned_query:
            return z_rep
        y = self.apply_condition_mask(y_gt)                  # (B, y_dim, out_dim)
        return torch.cat([z_rep, y], dim=1)                 # (B, in_dim, out_dim)

    def forward(self, z, y_gt):
        """
        z:    (B, z_dim)
        y_gt: (B, y_dim)
        out:  (B, out_dim)
        """
        x = self._concat(z, y_gt)  # (B, in_dim, out_dim)

        # projections under W1/W2 for each head
        p1 = torch.einsum('bdk,drk->brk', x, self.W1)  # (B, r, k)
        p2 = torch.einsum('bdk,drk->brk', x, self.W2)  # (B, r, k)

        # x^T W_sym x = <W1 x, W2 x>
        out = torch.einsum('brk,brk->bk', p1, p2) * self.scale + self.bias
        return out


        
class QuadraticConcatReuseProbe(Probe):
    """
    Reused quadratic probe with feature-wise parameter sharing.

    Parameters are learned per feature type and reused across conjunction heads.
    """
    def __init__(self, z_dim, y_dim, rank, out_dim, feature_dims, condition_mask=None, use_conditioned_query=True):
        super().__init__(y_dim, out_dim, condition_mask, use_conditioned_query=use_conditioned_query)

        N1, N2 = feature_dims
        assert N1 * N2 == out_dim, "feature_dims must multiply to out_dim"

        self.z_dim = z_dim
        self.y_dim = y_dim
        self.in_dim = z_dim + (y_dim if self.use_conditioned_query else 0)
        self.rank = rank
        self.out_dim = out_dim
        self.N1 = N1
        self.N2 = N2
        self.scale = 1.0 / math.sqrt(self.in_dim)

        # ---- learn ONLY these ----
        self.W1_z = nn.Parameter(torch.randn(z_dim, rank, N1) * 0.02)
        self.W2_z = nn.Parameter(torch.randn(z_dim, rank, N2) * 0.02)
        if self.use_conditioned_query:
            self.W1_y = nn.Parameter(torch.randn(y_dim, rank, out_dim) * 0.02)
            self.W2_y = nn.Parameter(torch.randn(y_dim, rank, out_dim) * 0.02)
        else:
            self.register_parameter("W1_y", None)
            self.register_parameter("W2_y", None)

        self.bias = nn.Parameter(torch.zeros(out_dim))

    def _build_W1_W2(self):
        """
        Construct reused W1, W2 on the fly.
        Returns:
            W1, W2: (in_dim, rank, out_dim)
        """
        # expand z-parameters across Cartesian product
        W1_z_full = self.W1_z.repeat_interleave(self.N2, dim=2)  # (z_dim, r, out_dim)
        W2_z_full = self.W2_z.repeat(1,1,self.N1)              # (z_dim, r, out_dim)

        if not self.use_conditioned_query:
            return W1_z_full, W2_z_full

        # concatenate z / y blocks
        W1 = torch.cat([W1_z_full, self.W1_y], dim=0)
        W2 = torch.cat([W2_z_full, self.W2_y], dim=0)
        return W1, W2

    def _concat(self, z, y_gt):
        z_rep = z.unsqueeze(-1).expand(-1, -1, self.out_dim) # (B, z_dim, out_dim)
        if not self.use_conditioned_query:
            return z_rep
        y = self.apply_condition_mask(y_gt)                  # (B, y_dim, out_dim)
        return torch.cat([z_rep, y], dim=1)                 # (B, in_dim, out_dim)

    def forward(self, z, y_gt):
        """
        z:    (B, z_dim)
        y_gt: (B, y_dim)
        out:  (B, out_dim)
        """
        x = self._concat(z, y_gt)        # (B, in_dim, out_dim)
        W1, W2 = self._build_W1_W2()     # constructed, NOT parameters

        p1 = torch.einsum('bdk,drk->brk', x, W1)
        p2 = torch.einsum('bdk,drk->brk', x, W2)

        out = torch.einsum('brk,brk->bk', p1, p2) * self.scale + self.bias
        return out



class MultilinearConcatReuseProbe(Probe):
    """
    Reused multilinear probe with feature-wise parameter sharing (generalizes QuadraticConcatReuseProbe).

    For G feature types with sizes feature_dims = [N1, ..., NG] and out_dim = ∏Ng,
    we learn per-feature-type z-projections (z_dim, rank, Ng) and reuse them across
    conjunction heads by expanding to (z_dim, rank, out_dim) according to the
    Cartesian-product ordering (last feature varies fastest).

    We also learn per-group y-projections (y_dim, rank, out_dim) (not reused),
    so the interface + conditioning behavior stays the same as QuadraticConcatReuseProbe.

    Score:
        p_g = x^T W_g   -> (B, rank, out_dim)
        out[b,k] = sum_r ∏_g p_g[b,r,k]   + bias[k]
    """
    def __init__(self, z_dim, y_dim, rank, out_dim, feature_dims, condition_mask=None, use_conditioned_query=True):
        super().__init__(y_dim, out_dim, condition_mask, use_conditioned_query=use_conditioned_query)

        self.feature_dims = [int(n) for n in feature_dims]
        assert len(self.feature_dims) >= 2, "Need at least 2 feature types."
        prod_out = 1
        for n in self.feature_dims:
            prod_out *= n
        assert prod_out == out_dim, "feature_dims must multiply to out_dim"

        self.z_dim = int(z_dim)
        self.y_dim = int(y_dim)
        self.in_dim = self.z_dim + (self.y_dim if self.use_conditioned_query else 0)
        self.rank = int(rank)
        self.out_dim = int(out_dim)
        self.n_groups = len(self.feature_dims)
        self.scale = 1.0 / math.sqrt(self.in_dim)

        # ---- learn ONLY these ----
        # per-group z parameters: (z_dim, rank, N_g)
        self.W_z = nn.ParameterList([
            nn.Parameter(torch.randn(self.z_dim, self.rank, Ng) * 0.02)
            for Ng in self.feature_dims
        ])
        # per-group y parameters (head-specific): (y_dim, rank, out_dim)
        if self.use_conditioned_query:
            self.W_y = nn.ParameterList([
                nn.Parameter(torch.randn(self.y_dim, self.rank, self.out_dim) * 0.02)
                for _ in range(self.n_groups)
            ])
        else:
            self.W_y = nn.ParameterList()

        self.bias = nn.Parameter(torch.zeros(self.out_dim))

    def _prod(self, xs):
        p = 1
        for x in xs:
            p *= int(x)
        return p

    def _expand_z_to_outdim(self, Wg_z: torch.Tensor, g: int) -> torch.Tensor:
        """
        Expand (z_dim, rank, N_g) -> (z_dim, rank, out_dim) by repeating across the
        Cartesian product, with the last feature type varying fastest.

        Pattern:
          later = ∏_{t>g} N_t  (how many times each value is repeated contiguously)
          earlier = ∏_{t<g} N_t (how many times the whole block is repeated)
        """
        later = self._prod(self.feature_dims[g + 1 :]) if g + 1 < self.n_groups else 1
        earlier = self._prod(self.feature_dims[:g]) if g > 0 else 1

        W_full = Wg_z.repeat_interleave(later, dim=2)   # (z_dim, r, N_g * later)
        W_full = W_full.repeat(1, 1, earlier)          # (z_dim, r, out_dim)
        return W_full

    def _build_W_list(self):
        """
        Construct reused W_g on the fly.
        Returns:
            Ws: list of length G, each (in_dim, rank, out_dim)
        """
        Ws = []
        for g in range(self.n_groups):
            Wz_full = self._expand_z_to_outdim(self.W_z[g], g)   # (z_dim, r, out_dim)
            if self.use_conditioned_query:
                Wg = torch.cat([Wz_full, self.W_y[g]], dim=0)    # (in_dim, r, out_dim)
            else:
                Wg = Wz_full
            Ws.append(Wg)
        return Ws

    def _concat(self, z, y_gt):
        z_rep = z.unsqueeze(-1).expand(-1, -1, self.out_dim) # (B, z_dim, out_dim)
        if not self.use_conditioned_query:
            return z_rep
        y = self.apply_condition_mask(y_gt)                  # (B, y_dim, out_dim)
        return torch.cat([z_rep, y], dim=1)                 # (B, in_dim, out_dim)

    def forward(self, z, y_gt):
        """
        z:    (B, z_dim)
        y_gt: (B, y_dim)
        out:  (B, out_dim)
        """
        x = self._concat(z, y_gt)       # (B, in_dim, out_dim)
        Ws = self._build_W_list()       # list of (in_dim, r, out_dim)

        # p_g: (B, r, out_dim)
        ps = [torch.einsum("bdk,drk->brk", x, Wg) for Wg in Ws]

        # multilinear product across groups, then sum over rank
        prod = ps[0]
        for g in range(1, self.n_groups):
            prod = prod * ps[g]         # (B, r, out_dim)

        out = prod.sum(dim=1) * self.scale + self.bias  # (B, out_dim)
        return out

    


def get_probe(
    probe_type,
    z_dim,
    y_dim,
    out_dim,
    rank=None,
    feature_dims=None,
    condition_mask=None,
    token_dim=None,
    attend_cls=False,
    dnn_hidden_dim=None,
    dnn_num_layers=3,
    dnn_dropout=0.0,
    use_conditioned_query=True,
):
    if probe_type == 'linear':
        return LinearProbe(
            z_dim,
            y_dim,
            out_dim,
            condition_mask,
            use_conditioned_query=use_conditioned_query,
        )
    elif probe_type in {'dnn_concat', 'dnn'}:
        return DNNConcatProbe(
            z_dim,
            y_dim,
            out_dim,
            hidden_dim=dnn_hidden_dim,
            num_layers=dnn_num_layers,
            dropout=dnn_dropout,
            condition_mask=condition_mask,
            use_conditioned_query=use_conditioned_query,
        )
    elif probe_type == 'attention_quadratic':
        return AttentionQuadraticProbe(
            z_dim,
            y_dim,
            out_dim,
            token_dim=token_dim,
            rank=rank,
            attend_cls=attend_cls,
            condition_mask=condition_mask,
            use_conditioned_query=use_conditioned_query,
        )
    elif probe_type == 'quadratic_concat':
        return QuadraticConcatProbe(
            z_dim,
            y_dim,
            rank,
            out_dim,
            condition_mask,
            use_conditioned_query=use_conditioned_query,
        )
    elif probe_type == 'quadratic_concat_reuse':
        return QuadraticConcatReuseProbe(
            z_dim,
            y_dim,
            rank,
            out_dim,
            feature_dims,
            condition_mask,
            use_conditioned_query=use_conditioned_query,
        )
    elif probe_type == 'multilinear_concat_reuse':
        return MultilinearConcatReuseProbe(
            z_dim,
            y_dim,
            rank,
            out_dim,
            feature_dims,
            condition_mask,
            use_conditioned_query=use_conditioned_query,
        )
    else:
        raise ValueError(f"Unknown probe type: {probe_type}")



        





        
    
