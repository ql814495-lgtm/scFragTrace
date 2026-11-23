"""Graph networks of scFragTrace on the single-cell module graph (764 modules, single-cell co-accessibility edges).

ResidualLineageGNN - tissue of origin, final 'wdat' variant (Region branch, FiLM conditioning, lineage-masked readout).
DetectionGNN - cancer detection head on the same Region branch.

Region branch: 1- and 2-hop graph convolution (A x, A^2 x) -> graph transformer with Graphormer biases (hop-distance and
edge terms) -> node tokens. ResidualLineageGNN: FiLM conditioning of the tokens on [TFx, CNA template r] -> lineage-masked
attention readout (each class attends only to modules of its own lineage) = residual logits; score = PRIOR + lambda * residual.
Training ('wdat'): TFx-weighted class-balanced cross-entropy on PRIOR + residual (w = clip(TFx/5%, 0.05, 1)),
KL distillation towards the prior posterior (weight 0.5, TFx-weighted), cohort-adversarial head through gradient reversal
(weight 0.3), transductive distillation on unlabelled samples (no labels used).

Parameter names are those of the shipped weight files (state_dicts). The fixed buffers (normalised
adjacency A, hop distances D, lineage mask, has-module flags) are not stored in the weight files; they are rebuilt from
model/weights/reference/module_graph.npz."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

BUFFERS = ("r.A", "r.D", "mask", "hasmod")


class Region(nn.Module):
    def __init__(s, A, D, fin, d=32, heads=4):
        super().__init__()
        s.register_buffer("A", torch.tensor(A)); s.register_buffer("D", torch.tensor(D)); s.h = heads; s.d = d
        s.inp = nn.Linear(2 * fin, d); s.emb = nn.Parameter(torch.zeros(len(A), d)); nn.init.normal_(s.emb, std=0.02)
        s.qkv = nn.Linear(d, 3 * d); s.o = nn.Linear(d, d); s.ln1 = nn.LayerNorm(d); s.ln2 = nn.LayerNorm(d)
        s.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        s.bsp = nn.Parameter(torch.zeros(heads, 4)); s.bedge = nn.Parameter(torch.ones(heads))   # Graphormer biases
        s.att = nn.Linear(d, 1); s.dp = nn.Dropout(0.2)

    def tokens(s, x):                                                     # x: B x M x F (standardised)
        x1 = s.A @ x; x2 = s.A @ x1                                       # graph convolution, 1 and 2 hops
        h = s.dp(Fn.gelu(s.inp(torch.cat([x1, x2], -1)) + s.emb))
        B_, M, d = h.shape; q, k, v = s.qkv(s.ln1(h)).view(B_, M, 3, s.h, d // s.h).permute(2, 0, 3, 1, 4)
        bias = s.bsp[:, s.D] + s.bedge[:, None, None] * s.A[None] * len(s.A) ** 0.5   # heads x M x M
        a = torch.softmax(q @ k.transpose(-1, -2) / (d // s.h) ** 0.5 + bias[None], -1)
        h = h + s.dp(s.o((s.dp(a) @ v).transpose(1, 2).reshape(B_, M, d)))
        return h + s.dp(s.ff(s.ln2(h)))

    def pool(s, h):
        """Attention pooling over all modules (detection head)."""
        w = torch.softmax(s.att(h), 1); return (w * h).sum(1), w.squeeze(-1)


class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x): return x.view_as(x)
    @staticmethod
    def backward(ctx, g): return -g


def tz(A):
    A = A - A.mean(1, keepdim=True); return A / (A.std(1, keepdim=True) + 1e-9)


class ResidualLineageGNN(nn.Module):
    """mask: M x (K+1) bool (module m may be attended by class k; last column = all modules, detection slot);
    hasmod: K floats (0 for classes without modules, e.g. prostate). g = prior inputs [zs(TFx*CNA), zs(TFx*S), CNA r, TFx]."""
    def __init__(s, A, Dh, fin, mask, hasmod, nco=5, d=32, heads=4):
        super().__init__(); K = len(hasmod); s.K = K
        s.r = Region(A, Dh, fin, d, heads); s.film = nn.Linear(1 + K, 2 * d); s.att = nn.Linear(d, K + 1)
        s.u = nn.Parameter(torch.randn(K + 1, d) * 0.1); s.b = nn.Parameter(torch.zeros(K + 1))
        s.wp = nn.Parameter(torch.tensor([1.0, 1.0, 0.0]))          # unused in the residual forward; kept for parity
        s.register_buffer("mask", torch.as_tensor(mask, dtype=torch.bool)); s.register_buffer("hasmod", torch.as_tensor(hasmod, dtype=torch.float32))
        s.adv = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, nco))

    def forward(s, x, g):
        K = s.K
        h = s.r.tokens(x); cnd = torch.cat([g[:, -1:], g[:, 2 * K:3 * K]], 1); gm, bt = s.film(cnd).chunk(2, 1)
        h = h * (1 + gm[:, None]) + bt[:, None]
        m = s.mask[None].expand(h.shape[0], -1, -1); a = s.att(h); a = torch.where(m, a, torch.full_like(a, -1e9)); w = torch.softmax(a, 1)
        gl = (w * (h @ s.u.T)).sum(1) + s.b; gnn = gl[:, :K] * s.hasmod
        pri = g[:, K:2 * K] + tz(g[:, 2 * K:3 * K])
        return dict(roo=pri + gnn, gnn=gnn, pri=pri, dom=s.adv(GRL.apply(h.mean(1))), attn=w)


class DetectionGNN(nn.Module):
    """Region branch + attention pooling + one logit (cancer vs non-cancer)."""
    def __init__(s, A, Dh, fin, d=32, heads=4):
        super().__init__(); s.r = Region(A, Dh, fin, d, heads); s.rdet = nn.Linear(d, 1)

    def forward(s, x, g=None):
        h = s.r.tokens(x); z, w = s.r.pool(h)
        return dict(rdet=s.rdet(z).squeeze(-1), w=w)


def state_dict_for_saving(net):
    """state_dict without the fixed graph buffers (rebuilt from the reference files at load time)."""
    return {k: v.detach().clone() for k, v in net.state_dict().items() if k not in BUFFERS}


def load_weights(net, sd):
    missing, unexpected = net.load_state_dict(sd, strict=False)
    if unexpected or set(missing) - set(BUFFERS):
        raise ValueError(f"weight file does not match the network: missing {missing}, unexpected {unexpected}")
    return net


@torch.no_grad()
def run(net, X, G=None, chunk=32):
    """Evaluate a network in chunks (keeps attention memory small). Returns dict of numpy arrays."""
    net.eval(); xt = torch.as_tensor(X, dtype=torch.float32)
    gt = None if G is None else torch.as_tensor(G, dtype=torch.float32)
    parts = [net(xt[i:i + chunk], None if gt is None else gt[i:i + chunk]) for i in range(0, len(xt), chunk)]
    return {k: np.concatenate([p[k].numpy() for p in parts]) for k in parts[0]}


def train_wdat(net, Xs, PRI, y, tf, cohort_id, fit, val, unl, seed, epochs=60, patience=12, dist=True, adv=True):
    """Training loop of the final variant ('wdat'). Returns residual logits for
    all samples and the number of epochs run. Rows of y outside fit/val are never used as labels."""
    K = net.K; wtf = np.clip(tf / 0.05, 0.05, 1.0).astype(np.float32)
    torch.manual_seed(seed); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2); r = np.random.default_rng(seed)
    xt = torch.tensor(Xs); gt = torch.tensor(PRI); yo = torch.tensor(y); wt = torch.tensor(wtf); ct = torch.tensor(cohort_id)
    cw = np.bincount(y[fit], weights=wtf[fit], minlength=K); cw = torch.tensor((cw.sum() / (K * np.maximum(cw, 1e-3))).astype(np.float32))
    def sup(o, b): return (Fn.cross_entropy(o["roo"], yo[b], weight=cw, reduction="none") * wt[b]).mean()
    def kl(o, b): return (Fn.kl_div(Fn.log_softmax(o["roo"], 1), Fn.softmax(o["pri"].detach(), 1), reduction="none").sum(1) * wt[b]).mean()
    best, bad, state = 1e9, 0, None; pool = np.concatenate([fit, unl]) if len(unl) else fit
    for ep in range(epochs):
        net.train()
        for b in torch.tensor(r.permutation(fit)).split(16):
            o = net(xt[b], gt[b]); l = sup(o, b)
            if adv: l = l + 0.3 * Fn.cross_entropy(o["dom"], ct[b])
            if dist:
                u = torch.tensor(r.choice(pool, 16)); ou = net(xt[u], gt[u]); l = l + 0.5 * kl(ou, u)
                if adv: l = l + 0.3 * Fn.cross_entropy(ou["dom"], ct[u])
            opt.zero_grad(); l.backward(); opt.step()
        net.eval()
        with torch.no_grad(): vl = float(np.mean([float(sup(net(xt[v], gt[v]), torch.tensor(v))) for v in np.array_split(val, max(1, len(val) // 32))]))
        if vl < best - 1e-4: best, bad, state = vl, 0, {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    net.load_state_dict(state); net.eval()
    return run(net, Xs, PRI)["gnn"], ep + 1


def train_detection(net, Xs, ydet, fit, val, seed, epochs=60, patience=12):
    """Training loop of the detection head: class-weighted binary cross-entropy (pos_weight = non-cancer / cancer in fit), AdamW 1e-3, weight decay
    1e-2, batches of 16, early stopping on the validation loss. Returns detection logits for all samples and epochs run."""
    torch.manual_seed(seed); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2); r = np.random.default_rng(seed)
    xt = torch.tensor(Xs); yd = torch.tensor(ydet, dtype=torch.float32)
    pw = torch.tensor((ydet[fit] == 0).sum() / max((ydet[fit] == 1).sum(), 1), dtype=torch.float32)
    def loss(o, b): return Fn.binary_cross_entropy_with_logits(o["rdet"], yd[b], pos_weight=pw)
    best, bad, state = 1e9, 0, None
    for ep in range(epochs):
        net.train()
        for b in torch.tensor(r.permutation(fit)).split(16):
            l = loss(net(xt[b]), b); opt.zero_grad(); l.backward(); opt.step()
        net.eval()
        with torch.no_grad(): vl = float(np.mean([float(loss(net(xt[v]), torch.tensor(v))) for v in np.array_split(val, max(1, len(val) // 32))]))
        if vl < best - 1e-4: best, bad, state = vl, 0, {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    net.load_state_dict(state); net.eval()
    return run(net, Xs)["rdet"], ep + 1
