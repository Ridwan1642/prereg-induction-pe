import os
import sys
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
torch.set_float32_matmul_precision('high')    # Supported only in RTX-30x, RTX-40x and enwer series

MODES      = ['learned', 'rope', 'nope']
COLORS     = {'learned': 'tab:blue', 'rope': 'tab:orange', 'nope': 'tab:green'}
KNOBS      = [192, 384, 512, 768, 1024]
_LS_CYCLE  = ['-', '--', '-.', ':', (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1))]
LINESTYLES = {bs: _LS_CYCLE[i % len(_LS_CYCLE)] for i, bs in enumerate(KNOBS)}
DATA_GEN   = torch.Generator() 

# ---------------- hyperparameters ----------------
batch_size       = 64        # default
max_iters        = 3000
eval_interval    = 200
learning_rate    = 3e-4
device           = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters       = 50
n_embd           = 384
n_head           = 6
n_layer          = 2
dropout          = 0.0
vocab_size       = 100
SEEDS            = [1337, 0, 42, 7, 2024]
BATCH_FOR        = {192: 64, 384: 32, 512: 16, 768: 8, 1024: 8}
TARGET_EFF_BATCH = 64
CONST_PREFIX     = 64 
IND_N            = 64   # distinct tokens per copy


def batch_for(block_size):
    return BATCH_FOR.get(block_size, batch_size)

def accum_for(block_size):
    return max(1, TARGET_EFF_BATCH // batch_for(block_size))

def probe_prefix_for(block_size):
    return min(CONST_PREFIX, block_size // 2)

    
# ---------------- RoPE helpers ----------------
def precompute_freqs_cis(head_dim, max_seq_len, base=10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, inv_freq)
    return torch.polar(torch.ones_like(freqs), freqs)

def apply_rope(x, freqs_cis):
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    out = torch.view_as_real(xc * freqs_cis)
    return out.flatten(-2).type_as(x)


# ---------------- data ----------------
def get_batch(block_size, bsz=None):
    B, T = (bsz or batch_for(block_size)), block_size
    p = min(CONST_PREFIX, T // 2)
    seqs = torch.zeros(B, T, dtype=torch.long)
    for b in range(B):
        half = max(2, p + torch.randint(-p // 8, p // 8 + 1, (1,), generator=DATA_GEN).item())
        prefix = torch.randint(0, vocab_size, (half,), generator=DATA_GEN)
        reps = math.ceil(T / half)
        seqs[b] = prefix.repeat(reps)[:T]
    idx = seqs[:, :-1].to(device)
    targets = seqs[:, 1:].to(device)
    return idx, targets


# ---------------- model ----------------
class Head(nn.Module):
    def __init__(self, head_size, pos_mode, block_size):
        super().__init__()
        self.pos_mode = pos_mode
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        if pos_mode == 'rope':
            self.register_buffer('freq',
                                 precompute_freqs_cis(head_size, block_size),
                                 persistent=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        if self.pos_mode == 'rope':
            q = apply_rope(q, self.freq[:T])
            k = apply_rope(k, self.freq[:T])
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        self.last_wei = wei.detach()
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, pos_mode, layer_idx, block_size):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size, pos_mode, block_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.layer_idx = layer_idx
        self.ablate_head = None

    def forward(self, x):
        outs = [h(x) for h in self.heads]
        if self.layer_idx == 0 and self.ablate_head is not None:
            outs[self.ablate_head] = torch.zeros_like(outs[self.ablate_head])
        out = torch.cat(outs, dim=-1)
        return self.dropout(self.proj(out))


class Block(nn.Module):
    def __init__(self, n_embd, n_head, pos_mode, layer_idx, block_size):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, pos_mode, layer_idx, block_size)
        self.ln1 = nn.LayerNorm(n_embd)

    def forward(self, x):
        return x + self.sa(self.ln1(x))


class GPTLanguageModel(nn.Module):
    def __init__(self, pos_mode, block_size):
        super().__init__()
        self.pos_mode = pos_mode
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        if pos_mode == 'learned':
            self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, pos_mode, i, block_size) for i in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding_table(idx)
        if self.pos_mode == 'learned':
            pos = self.position_embedding_table(torch.arange(T, device=idx.device))
            x = x + pos
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        Bc, Tc, Cc = logits.shape
        loss = F.cross_entropy(logits.view(Bc * Tc, Cc), targets.view(Bc * Tc))
        return logits, loss


# ---------------- metrics ----------------
@torch.no_grad()
def estimate_loss(model, block_size):
    model.eval()
    half = block_size // 2
    first, second = torch.zeros(eval_iters), torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch(block_size)
        logits, _ = model(X, Y)
        B, T, C = logits.shape
        ce = F.cross_entropy(
            logits.reshape(B * T, C), Y.reshape(B * T), reduction='none'
        ).view(B, T)
        first[k]  = ce[:, :half].mean()
        second[k] = ce[:, half:].mean()
    model.train()
    return {'first': first.mean().item(), 'second': second.mean().item()}


@torch.no_grad()
# The following function creates a probe that tiles the block size
def make_probe(prefix_len, block_size):
    g = torch.Generator().manual_seed(2718)  
    prefix = torch.randint(0, vocab_size, (prefix_len,), generator=g)
    reps = math.ceil(block_size / prefix_len)
    seq = prefix.repeat(reps)[:block_size]
    return seq.unsqueeze(0).to(device)


# More accurate probing
@torch.no_grad()
def make_induction_probe(n_distinct=IND_N, seed=2718):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(vocab_size, generator=g)[:n_distinct]
    seq = torch.cat([perm, perm])                 
    return seq.unsqueeze(0).to(device), n_distinct


@torch.no_grad()
def stripe_mass(model, layer, head, prefix_len, probe_seq):
    model.eval()
    _ = model(probe_seq)
    a = model.blocks[layer].sa.heads[head].last_wei[0]
    T = a.shape[0]
    qs = torch.arange(prefix_len, T)
    ks = qs - prefix_len + 1
    val = a[qs, ks].mean().item()
    model.train()
    return val

@torch.no_grad()
def stripe_mass_multi(model, layer, head, prefix_len, probe_seq):
    model.eval()
    _ = model(probe_seq)
    a = model.blocks[layer].sa.heads[head].last_wei[0]
    T = a.shape[0]
    total, count = 0.0, 0
    for i in range(prefix_len, T):
        max_k = (i + 1) // prefix_len
        ks = [i - k * prefix_len + 1 for k in range(1, max_k + 1)]  
        if ks:
            total += a[i, ks].sum().item()
        count += 1
    model.train()
    return total / max(count, 1)

@torch.no_grad()
def best_induction_head(model, layer, prefix_len, probe_seq):
    scores = [stripe_mass(model, layer, h, prefix_len, probe_seq) for h in range(n_head)]
    h = int(max(range(n_head), key=lambda i: scores[i]))
    return h, scores[h], scores

@torch.no_grad()
def induction_score(model, layer, head, n, probe):
    model.eval(); _ = model(probe)
    a = model.blocks[layer].sa.heads[head].last_wei[0]      
    qs = torch.arange(n, 2 * n - 1)                        
    ks = qs - n + 1                                         
    val = a[qs, ks].mean().item()
    model.train(); return val

@torch.no_grad()
def prevtok_stripe(model, layer, head, probe_seq):
    model.eval()
    _ = model(probe_seq)
    a = model.blocks[layer].sa.heads[head].last_wei[0]
    T = a.shape[0]
    qs = torch.arange(1, T)
    val = a[qs, qs - 1].mean().item()
    model.train()
    return val

@torch.no_grad()
def behavioral_induction_score(model, n, probe):
    model.eval()
    logits, _ = model(probe)
    logp = F.log_softmax(logits[0], dim=-1)
    seq = probe[0]
    qs = torch.arange(n, 2 * n - 1)
    correct = seq[qs + 1]                                   
    val = logp[qs, correct].exp().mean().item()
    model.train(); return val

@torch.no_grad()
def best_prevtok_head(model, layer, probe_seq):
    scores = [prevtok_stripe(model, layer, h, probe_seq) for h in range(n_head)]
    h = int(max(range(n_head), key=lambda i: scores[i]))
    return h, scores[h]


@torch.no_grad()
def ablate_and_measure(model, probe, prefix_len, block_size):
    base_loss = estimate_loss(model, block_size)
    h1_base, mass1_base, _ = best_induction_head(model, 1, prefix_len, probe)
    h0, mass0 = best_prevtok_head(model, 0, probe)
    ind_base   = induction_score(model, 1, h1_base, IND_N, probe)
    behav_base = behavioral_induction_score(model, IND_N, probe)

    model.blocks[0].sa.ablate_head = h0
    abl_loss = estimate_loss(model, block_size)
    _, _, scores_abl = best_induction_head(model, 1, prefix_len, probe)
    mass1_abl = scores_abl[h1_base]
    behav_abl = behavioral_induction_score(model, IND_N, probe)
    model.blocks[0].sa.ablate_head = None

    return {
        'l0_head': h0, 'l0_prevtok_mass': mass0,
        'l1_head': h1_base,
        'l1_mass_base': mass1_base, 'l1_mass_ablated': mass1_abl,
        'second_loss_base': base_loss['second'],
        'second_loss_ablated': abl_loss['second'],
        'ind_score_base': ind_base,
        'behav_base': behav_base, 'behav_ablated': behav_abl,
    }


# ---------------- training (one condition) ----------------
def run_condition(pos_mode, seed, block_size):
    torch.manual_seed(seed)
    DATA_GEN.manual_seed(seed)
    model = GPTLanguageModel(pos_mode, block_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    probe, prefix_len = make_induction_probe()
    hist = {'iter': [], 'first': [], 'second': [], 'stripe_l1': [], 'stripe_l1_multi': [], 'stripe_l0': []}
    for it in range(max_iters):
        if it % eval_interval == 0 or it == max_iters - 1:
            losses = estimate_loss(model, block_size)
            h1_eval, m_l1, _ = best_induction_head(model, 1, prefix_len, probe)
            m_l1_multi = stripe_mass_multi(model, 1, h1_eval, prefix_len, probe)
            model.eval()
            _ = model(probe)
            l0_scores = []
            for h in range(n_head):
                a = model.blocks[0].sa.heads[h].last_wei[0]
                T = a.shape[0]
                qs = torch.arange(1, T)
                l0_scores.append(a[qs, qs - 1].mean().item())
            model.train()
            m_l0 = max(l0_scores)
            hist['iter'].append(it)
            hist['first'].append(losses['first'])
            hist['second'].append(losses['second'])
            hist['stripe_l1'].append(m_l1)
            hist['stripe_l0'].append(m_l0)
            hist['stripe_l1_multi'].append(m_l1_multi)

        accum = accum_for(block_size)
        optimizer.zero_grad(set_to_none=True)
        for _ in range(accum):
            xb, yb = get_batch(block_size)
            _, loss = model(xb, yb)
            (loss / accum).backward()
        optimizer.step()

    h1, mass1, scores1 = best_induction_head(model, 1, prefix_len, probe)
    return model, hist, probe, h1


OUTDIR = f"runs/{datetime.now():%Y%m%d_%H%M%S}"
os.makedirs(OUTDIR, exist_ok=True)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


_logf = open(f"{OUTDIR}/console.log", "w")
sys.stdout = Tee(sys.__stdout__, _logf)
print(f"[harness] writing all outputs to: {OUTDIR}")


# ---------------- single-pass multiseed ----------------
def run_condition_multiseed(pos_mode, block_size):
    per_seed = []
    models = []
    for s in SEEDS:
        print(f"  [{pos_mode} | bs={block_size}] seed {s} ...")
        model, hist, probe, best_h = run_condition(pos_mode, s, block_size)
        per_seed.append(hist)
        models.append((model, probe, best_h))

    iters = per_seed[0]['iter']
    agg = {'iter': iters}
    for key in ['first', 'second', 'stripe_l1', 'stripe_l1_multi', 'stripe_l0']:
        stacked = torch.tensor([h[key] for h in per_seed])
        agg[f'{key}_mean'] = stacked.mean(0).tolist()
        agg[f'{key}_min']  = stacked.min(0).values.tolist()
        agg[f'{key}_max']  = stacked.max(0).values.tolist()
        agg[f'{key}_all']  = stacked.tolist()

    print(f"[{pos_mode} | bs={block_size}] final induction mass mean over seeds: "
          f"{agg['stripe_l1_mean'][-1]:.3f} "
          f"(range {agg['stripe_l1_min'][-1]:.3f}-{agg['stripe_l1_max'][-1]:.3f})")
    model0, probe0, best_h0 = models[0]
    return agg, model0, probe0, best_h0, models


def ablate_multiseed_from_models(models, block_size):
    per_seed = []
    prefix_len = probe_prefix_for(block_size)
    for (model, probe, _best_h) in models:
        res = ablate_and_measure(model, probe, prefix_len, block_size)
        per_seed.append(res)

    agg = {}
    for key in ['l1_mass_base', 'l1_mass_ablated',
                'second_loss_base', 'second_loss_ablated', 'l0_prevtok_mass',
            'ind_score_base', 'behav_base', 'behav_ablated']:
        vals = torch.tensor([r[key] for r in per_seed])
        agg[f'{key}_mean'] = vals.mean().item()
        agg[f'{key}_std']  = vals.std(unbiased=False).item()
        agg[f'{key}_all']  = vals.tolist()
    agg['l0_heads'] = [r['l0_head'] for r in per_seed]
    agg['l1_heads'] = [r['l1_head'] for r in per_seed]

    drop = agg['l1_mass_base_mean'] - agg['l1_mass_ablated_mean']
    print(f"  L1 induction mass "
          f"{agg['l1_mass_base_mean']:.3f} -> {agg['l1_mass_ablated_mean']:.3f} "
          f"(drop {drop:.3f}); second-half loss "
          f"{agg['second_loss_base_mean']:.3f} -> {agg['second_loss_ablated_mean']:.3f}")
    return agg, per_seed


# ---------------- run ----------------
print("=" * 70)
results = {}
ablation = {}
for block_size in KNOBS:
    for mode in MODES:
        print(f"\n--- training condition: {mode} bs={block_size} "
              f"(batch={batch_for(block_size)}, {len(SEEDS)} seeds) ---")
        agg, model0, probe0, best_h0, models = run_condition_multiseed(mode, block_size)
        print(f"--- causal ablation: {mode} bs={block_size} ---")
        abl_agg, abl_per_seed = ablate_multiseed_from_models(models, block_size)
        ablation[(mode, block_size)] = {
            'agg': abl_agg, 'per_seed': abl_per_seed, 'block_size': block_size}
        results[(mode, block_size)] = {
            'hist': agg, 'model': model0.cpu(), 'probe': probe0.cpu(),
            'best_h': best_h0, 'block_size': block_size}
        del models
        if device == 'cuda':
            torch.cuda.empty_cache()
    print("=" * 70)


# ---------------- figures ----------------
def band(ax, iters, h, key, color, label, ls='-'):
    ax.plot(iters, h[f'{key}_mean'], color=color, lw=2, label=label, ls=ls)
    ax.fill_between(iters, h[f'{key}_min'], h[f'{key}_max'], color=color, alpha=0.12)


fig1, ax = plt.subplots(figsize=(11, 6))
for block_size in KNOBS:
    for mode in MODES:
        h = results[(mode, block_size)]['hist']
        band(ax, h['iter'], h, 'second', COLORS[mode],
             f'{mode} bs={block_size}', ls=LINESTYLES[block_size])
ax.axhline(math.log(vocab_size), color='gray', ls=':',
           label=f'uniform baseline ({math.log(vocab_size):.2f})')
ax.set_xlabel('iteration'); ax.set_ylabel('second-half cross-entropy')
ax.set_title(f'Induction loss across encodings & block sizes (mean±range, {len(SEEDS)} seeds)')
ax.legend(fontsize=7, ncol=3); ax.grid(alpha=0.3); fig1.tight_layout()

fig2, ax = plt.subplots(figsize=(11, 6))
for block_size in KNOBS:
    for mode in MODES:
        h = results[(mode, block_size)]['hist']; c = COLORS[mode]
        band(ax, h['iter'], h, 'stripe_l1', c,
             f'{mode} bs={block_size}: L1', ls=LINESTYLES[block_size])
ax.set_xlabel('iteration'); ax.set_ylabel('attention mass on expected offset')
ax.set_title(f'Circuit formation (L1 induction, mean±range, {len(SEEDS)} seeds)')
ax.legend(fontsize=7, ncol=3); ax.grid(alpha=0.3); fig2.tight_layout()

fig3, axes = plt.subplots(len(MODES), len(KNOBS),
                          figsize=(3.2 * len(KNOBS), 3.2 * len(MODES)))
for r_i, mode in enumerate(MODES):
    for c_i, block_size in enumerate(KNOBS):
        ax = axes[r_i, c_i]
        r = results[(mode, block_size)]
        model, probe, best_h = r['model'], r['probe'], r['best_h']
        model.eval()
        with torch.no_grad():
            _ = model(probe)
            a = model.blocks[1].sa.heads[best_h].last_wei[0].cpu()
        im = ax.imshow(a, cmap='viridis', vmin=0, vmax=1, aspect='auto')
        prefix_len = probe_prefix_for(block_size)
        T = a.shape[0]; qs = torch.arange(prefix_len, T); ks = qs - prefix_len + 1
        ax.plot(ks.numpy(), qs.numpy(), color='red', lw=0.6, ls='--', alpha=0.7)
        ax.set_title(f'{mode} bs={block_size}\nL1.h{best_h}', fontsize=8)
        ax.tick_params(labelsize=6)
fig3.suptitle(f'Example induction heads (canonical induction probe (single repeat, {IND_N} distinct)')
fig3.tight_layout(rect=[0, 0, 1, 0.97])

fig4, (axA, axB) = plt.subplots(1, 2, figsize=(14, 5.5))
for mode in MODES:
    c = COLORS[mode]
    base = [ablation[(mode, bs)]['agg']['l1_mass_base_mean'] for bs in KNOBS]
    abl  = [ablation[(mode, bs)]['agg']['l1_mass_ablated_mean'] for bs in KNOBS]
    axA.plot(KNOBS, base, '-o', color=c, label=f'{mode} baseline')
    axA.plot(KNOBS, abl, '--x', color=c, label=f'{mode} ablated')

    lbase = [ablation[(mode, bs)]['agg']['second_loss_base_mean'] for bs in KNOBS]
    labl  = [ablation[(mode, bs)]['agg']['second_loss_ablated_mean'] for bs in KNOBS]
    axB.plot(KNOBS, lbase, '-o', color=c, label=f'{mode} baseline')
    axB.plot(KNOBS, labl, '--x', color=c, label=f'{mode} ablated')

axA.set_xlabel('block size'); axA.set_ylabel('L1 induction mass')
axA.set_title('Induction mass vs block size (baseline vs L0-ablated)')
axA.legend(fontsize=7); axA.grid(alpha=0.3)
axB.axhline(math.log(vocab_size), color='gray', ls=':', label='uniform')
axB.set_xlabel('block size'); axB.set_ylabel('second-half cross-entropy')
axB.set_title('Loss vs block size (baseline vs L0-ablated)')
axB.legend(fontsize=7); axB.grid(alpha=0.3)
fig4.suptitle('Causal test: ablating the L0 prev-token head, swept over block size')
fig4.tight_layout(rect=[0, 0, 1, 0.95])


# ---------------- save figures ----------------
for name, fig in [('induction_loss', fig1), ('circuit_formation', fig2),
                  ('example_heads', fig3), ('causal_ablation', fig4)]:
    fig.savefig(f"{OUTDIR}/{name}.png", dpi=150, bbox_inches='tight')
    fig.savefig(f"{OUTDIR}/{name}.pdf", bbox_inches='tight')
print(f"[harness] saved 4 figures (png+pdf) to {OUTDIR}")


# ---------------- numeric summary ----------------
print("\n========== SUMMARY (over seeds) ==========")
for block_size in KNOBS:
    for mode in MODES:
        h = results[(mode, block_size)]['hist']
        fm = h['stripe_l1_mean'][-1]; fmin = h['stripe_l1_min'][-1]; fmax = h['stripe_l1_max'][-1]
        fmm = h['stripe_l1_multi_mean'][-1]
        fl = h['second_mean'][-1]
        crosses = []
        for seed_curve in h['stripe_l1_all']:
            c = next((h['iter'][i] for i, m in enumerate(seed_curve) if m > 0.5), None)
            crosses.append(c)
        print(f"{mode:7s} bs={block_size:4d} | induction mass {fm:.3f} "
              f"(range {fmin:.3f}-{fmax:.3f}) | second-half loss {fl:.3f} | "
              f"iters>0.5 per seed: {crosses}"
        f"| multi {fmm:.3f}")
print("==========================================")

print("\n========== ABLATION (over seeds) ==========")
for block_size in KNOBS:
    for mode in MODES:
        a = ablation[(mode, block_size)]['agg']
        print(f"{mode:7s} bs={block_size:4d} | L1 mass {a['l1_mass_base_mean']:.3f} -> "
              f"{a['l1_mass_ablated_mean']:.3f} "
              f"(drop {a['l1_mass_base_mean'] - a['l1_mass_ablated_mean']:.3f}) | "
              f"2nd-half loss {a['second_loss_base_mean']:.3f} -> "
              f"{a['second_loss_ablated_mean']:.3f}")
        print(f"            | L0 heads/seed: {a['l0_heads']} "
              f"(prev-tok mass {a['l0_prevtok_mass_mean']:.3f}) | "
              f"L1 heads/seed: {a['l1_heads']}")
        print(f"            | canon induction {a['ind_score_base_mean']:.3f} | "
      f"behavioral {a['behav_base_mean']:.3f} -> {a['behav_ablated_mean']:.3f} "
      f"(chance {1.0/vocab_size:.3f})")
print("===========================================")


# ---------------- dump machine-readable results ----------------
payload = {
    'config': {
        'batch_size': batch_size, 'block_sizes': KNOBS,
        'max_iters': max_iters, 'eval_interval': eval_interval,
        'learning_rate': learning_rate, 'n_embd': n_embd,
        'n_head': n_head, 'n_layer': n_layer, 'dropout': dropout,
        'vocab_size': vocab_size, 'seeds': SEEDS,
        'probe_prefix': {bs: probe_prefix_for(bs) for bs in KNOBS},
        'device': device,
    },
    'descriptive': {f'{mode}_bs{bs}': results[(mode, bs)]['hist']
                    for bs in KNOBS for mode in MODES},
    'ablation': {f'{mode}_bs{bs}': ablation[(mode, bs)]['agg']
                 for bs in KNOBS for mode in MODES},
}
with open(f"{OUTDIR}/results.json", "w") as f:
    json.dump(payload, f, indent=2)
print(f"[harness] saved results.json to {OUTDIR}")

plt.show()