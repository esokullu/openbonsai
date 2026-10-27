"""Distillation parity with the proven experiment script.

The oracle below is a verbatim copy of research/experiments/e2e_kd.py
(TernaryLinear, TernaryEmbedding, signs_for, install_ternary,
install_endpoints, kd_chunk_loss and the ``cmd_train`` loop). The only
edits: the text model path (``model.model`` for Qwen3_5ForCausalLM instead
of ``model.model.language_model``), the endpoint Hadamard size (the tiny
model's hidden width instead of 4096), CPU instead of ``.cuda()``, and the
loop returns its per-step losses instead of logging and evaluating. Tests
never import research/.
"""
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from openbonsai.cli import execute
from openbonsai.stages import distill as D
from openbonsai.stages.common import install_ternary


# ---------------------------------------------------------------- oracle ----
def hadamard(n, device):
    h = torch.ones(1, 1, device=device)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(n)


class TernaryLinear(torch.nn.Module):
    """y = (C * s) (R x) with R = blockdiag(H) diag(signs); s = s0 * exp(u)."""

    def __init__(self, codes, scales, signs, block, h):
        super().__init__()
        self.out_features, self.in_features = codes.shape
        self.group = self.in_features // scales.shape[1]
        self.block = block
        self.register_buffer("codes", codes.to(torch.int8), persistent=False)
        self.register_buffer("s0", scales.float(), persistent=False)
        self.register_buffer("signs", signs.to(torch.bfloat16), persistent=False)
        self.h = h  # shared [block, block] BF16 Hadamard, not a parameter
        self.log_mult = torch.nn.Parameter(torch.zeros_like(self.s0))

    def scales(self):
        return self.s0 * torch.exp(self.log_mult)

    def weight_deq(self):
        s = self.scales().to(torch.bfloat16)
        c = self.codes.view(self.out_features, -1, self.group).to(torch.bfloat16)
        return (c * s[..., None]).view(self.out_features, self.in_features)

    def forward(self, x):
        shape = x.shape
        xr = (x * self.signs).view(*shape[:-1], self.in_features // self.block, self.block) @ self.h
        return F.linear(xr.view(shape), self.weight_deq())


class TernaryEmbedding(torch.nn.Module):
    """e = (C[ids] * s[ids]) R^T, i.e. stored rows live in the rotated basis."""

    def __init__(self, codes, scales, signs, block, h):
        super().__init__()
        self.num, self.dim = codes.shape
        self.group = self.dim // scales.shape[1]
        self.block = block
        self.register_buffer("codes", codes.to(torch.int8), persistent=False)
        self.register_buffer("s0", scales.float(), persistent=False)
        self.register_buffer("signs", signs.to(torch.bfloat16), persistent=False)
        self.h = h
        self.log_mult = torch.nn.Parameter(torch.zeros_like(self.s0))

    def scales(self):
        return self.s0 * torch.exp(self.log_mult)

    def forward(self, ids):
        c = self.codes[ids].view(*ids.shape, -1, self.group).to(torch.bfloat16)
        s = (self.s0[ids] * torch.exp(self.log_mult[ids])).to(torch.bfloat16)
        rows = (c * s[..., None]).view(*ids.shape, self.dim // self.block, self.block)
        return (rows @ self.h).view(*ids.shape, self.dim) * self.signs


def install_endpoints(model, ep, hidden):
    dev = next(model.parameters()).device
    lm = model.model
    h = hadamard(hidden, dev).to(torch.bfloat16)
    if "lm_head" in ep:
        st = ep["lm_head"]
        model.lm_head = TernaryLinear(st["codes"].to(dev), st["scales"].to(dev), signs_for(st["seed"], hidden).to(dev), st["block"], h)
    if "embed_tokens" in ep:
        st = ep["embed_tokens"]
        lm.embed_tokens = TernaryEmbedding(st["codes"].to(dev), st["scales"].to(dev), signs_for(st["seed"], hidden).to(dev), st["block"], h)


def signs_for(seed, in_features):
    g = torch.Generator().manual_seed(seed)
    return (torch.randint(0, 2, (in_features,), generator=g) * 2 - 1).float()


def install_ternary_oracle(model, state):
    layers = model.model.layers
    dev = next(model.parameters()).device
    hs = {}
    for name, st in state["modules"].items():
        _, li, sub = name.split(".", 2)
        parent_name, attr = sub.rsplit(".", 1)
        parent = layers[int(li)].get_submodule(parent_name)
        lin = getattr(parent, attr)
        block = st["block"]
        if block not in hs:
            hs[block] = hadamard(block, dev).to(torch.bfloat16)
        tl = TernaryLinear(st["codes"].to(dev), st["scales"].to(dev), signs_for(st["seed"], lin.in_features).to(dev), block, hs[block])
        setattr(parent, attr, tl)
        del lin


def kd_chunk_loss(h, head, tidx, tlogp):
    logq = head(h).float().log_softmax(-1)
    lqk = logq.gather(-1, tidx)
    p = tlogp.float().exp()
    tail_p = (1 - p.sum(-1)).clamp_min(0)
    log_tail_q = torch.log(-torch.expm1(lqk.logsumexp(-1).clamp_max(-1e-6)))
    return -(p * lqk).sum() - (tail_p * log_tail_q).sum()


def cmd_train(model, ids, tidx_all, tlogp_all, *, lr, epochs, accum, warmup, chunk):
    torch.manual_seed(0)
    for p in model.parameters():
        p.requires_grad_(False)
    scale_params = [m.log_mult for m in model.modules() if isinstance(m, (TernaryLinear, TernaryEmbedding))]
    for p in scale_params:
        p.requires_grad_(True)
    model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    n = ids.shape[0]
    groups = [{"params": scale_params, "lr": lr}]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0, eps=1e-12)
    total_steps = epochs * n // accum
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps)))
    head = model.lm_head
    lm = model.model
    model.train()
    step, run_loss, run_tok = 0, 0.0, 0
    order = []
    for ep in range(epochs):
        order += torch.randperm(n, generator=torch.Generator().manual_seed(ep)).tolist()
    losses = []
    for it, i in enumerate(order):
        x = torch.from_numpy(ids[i:i + 1].astype(np.int64))
        tidx = torch.from_numpy(tidx_all[i].astype(np.int64))
        tlogp = torch.from_numpy(np.asarray(tlogp_all[i]))
        h = lm(input_ids=x, use_cache=False).last_hidden_state[0]
        loss = 0.0
        tlen = h.shape[0] - 1
        for c0 in range(0, tlen, chunk):
            c1 = min(c0 + chunk, tlen)
            loss = loss + checkpoint(kd_chunk_loss, h[c0:c1], head, tidx[c0:c1], tlogp[c0:c1], use_reentrant=False)
        (loss / (tlen * accum)).backward()
        run_loss += loss.item()
        run_tok += tlen
        if (it + 1) % accum == 0:
            torch.nn.utils.clip_grad_norm_(scale_params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            losses.append(run_loss / run_tok)   # --log-every 1
            run_loss, run_tok = 0.0, 0
    mods = {}
    for name, m in model.model.layers.named_modules():
        if isinstance(m, TernaryLinear):
            mods[f"layers.{name}"] = m.scales().detach().half().cpu()
    for name, m in [("lm_head", model.lm_head), ("embed_tokens", model.model.embed_tokens)]:
        if isinstance(m, (TernaryLinear, TernaryEmbedding)):
            mods[name] = m.scales().detach().half().cpu()
    return losses, mods
# ------------------------------------------------------------ end oracle ----


@pytest.fixture
def endpointed(calibrated):
    assert execute(calibrated, "quantize") == "ran"
    assert execute(calibrated, "endpoints") == "ran"
    return calibrated


def _legacy_layout(state):
    """The run state in the scripts' layout: shared signs are seed = width."""
    return {key: {"codes": t.codes, "scales": t.scales, "seed": t.shape[1], "block": state.block}
            for key, t in state.tensors.items()}


def test_three_distillation_steps_match_e2e_kd(endpointed):
    ctx = endpointed
    adapter = ctx.adapter()
    D.run_teacher(ctx)
    ids, tidx, tlogp = D.load_teacher(ctx.run.stage_dir("distill") / "teacher")
    hp = dict(lr=0.05, epochs=3, accum=4, warmup=1, chunk=32)
    assert hp["epochs"] * len(ids) // hp["accum"] == 3

    decoder, endpoints = ctx.quantize_state(), ctx.endpoints_state()
    oracle, _ = ctx.load_model()
    install_ternary_oracle(oracle, {"modules": _legacy_layout(decoder)})
    install_endpoints(oracle, _legacy_layout(endpoints), adapter.hidden_size)
    want_losses, want_scales = cmd_train(oracle, ids, tidx, tlogp, **hp)

    student, _ = ctx.load_model()
    install_ternary(adapter, student, ctx.state_through("endpoints").records(), torch.device("cpu"))
    history = D.train(student, adapter, ids, tidx, tlogp, ctx.run.stage_dir("distill"), log_every=1, ckpt_every=100,
                      device=torch.device("cpu"), log=lambda m: None, **hp)
    got = D.trained_scales(student, adapter)

    assert [h["step"] for h in history] == [1, 2, 3]
    assert [h["kd_ce"] for h in history] == want_losses   # bitwise; the plan asked for 1e-6
    assert sorted(got) == sorted(want_scales) and len(got) == 27
    assert all(torch.equal(got[k], want_scales[k]) for k in got)
    assert any(not torch.equal(got[k], t.scales) for k, t in ctx.state_through("endpoints").tensors.items())
