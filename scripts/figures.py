"""Figure generation for the Boltz-2 nanobody report (matplotlib, Agg backend)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _chain_boundaries(chains):
    """Return [(chain_id, start, end_exclusive), ...] for a token chain array."""
    bounds = []
    ids = list(dict.fromkeys(chains.tolist()))
    for ch in ids:
        idx = np.where(chains == ch)[0]
        bounds.append((ch, int(idx.min()), int(idx.max()) + 1))
    return bounds


def pae_heatmap(pae, chains, out_path, chain_labels=None, vmax=30.0, title="Predicted Aligned Error"):
    n = pae.shape[0]
    fig, ax = plt.subplots(figsize=(7.2, 6.4), dpi=130)
    im = ax.imshow(pae, cmap="Greens_r", vmin=0, vmax=vmax, interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("PAE (Å)  -  lower = more confident", fontsize=9)
    labels = chain_labels or {}
    for ch, start, end in _chain_boundaries(chains):
        if start > 0:
            ax.axhline(start - 0.5, color="black", lw=0.8, alpha=0.7)
            ax.axvline(start - 0.5, color="black", lw=0.8, alpha=0.7)
        ax.text((start + end) / 2, -n * 0.012, f"{ch} {labels.get(ch, '')}".strip(),
                ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.text(-n * 0.012, (start + end) / 2, f"{ch} {labels.get(ch, '')}".strip(),
                ha="right", va="center", rotation=90, fontsize=9, fontweight="bold")
    ax.set_xlabel("Scored residue (token index)", fontsize=9)
    ax.set_ylabel("Aligned residue (token index)", fontsize=9)
    # Keep the title above the outside-axis chain labels, including single-chain plots.
    ax.set_title(title, fontsize=11, pad=24)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    try:
        fig.savefig(out_path, bbox_inches="tight")
    finally:
        plt.close(fig)


def pae_interface_heatmap(pae, tokens, chain_a, chain_b, out_path,
                          chain_labels=None, vmax=30.0):
    """PAE blocks for the two inter-chain directions (A->B and B->A)."""
    labels = chain_labels or {}
    idx_a = np.array([i for i, t in enumerate(tokens) if t["chain"] == chain_a])
    idx_b = np.array([i for i, t in enumerate(tokens) if t["chain"] == chain_b])
    if idx_a.size == 0 or idx_b.size == 0:
        return
    block_ab = pae[np.ix_(idx_a, idx_b)]  # rows: A aligned, cols: B scored
    block_ba = pae[np.ix_(idx_b, idx_a)]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=130)
    for ax, block, (r_ch, c_ch), ttl in (
        (axes[0], block_ab, (chain_a, chain_b), f"rows {chain_a} aligned / cols {chain_b} scored"),
        (axes[1], block_ba, (chain_b, chain_a), f"rows {chain_b} aligned / cols {chain_a} scored"),
    ):
        im = ax.imshow(block, cmap="Greens_r", vmin=0, vmax=vmax, interpolation="nearest",
                       aspect="auto")
        ax.set_title(f"{labels.get(r_ch, r_ch)} -> {labels.get(c_ch, c_ch)}\n({ttl})", fontsize=9)
        ax.set_xlabel(f"{c_ch} residue", fontsize=9)
        ax.set_ylabel(f"{r_ch} residue", fontsize=9)
        ax.tick_params(labelsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).set_label("PAE (Å)", fontsize=8)
    fig.suptitle("Interface PAE (nanobody/antigen binding block)", fontsize=11)
    fig.tight_layout()
    try:
        fig.savefig(out_path, bbox_inches="tight")
    finally:
        plt.close(fig)


def pae_interface_union_heatmap(pae, tokens, antigen_chains, nanobody_chain, out_path,
                                chain_labels=None, vmax=30.0):
    """PAE blocks between all antigen chains as one interface and the nanobody."""
    labels = chain_labels or {}
    ag = [c for c in antigen_chains if c]
    idx_ag = np.array([i for ch in ag for i, t in enumerate(tokens) if t["chain"] == ch])
    idx_nb = np.array([i for i, t in enumerate(tokens) if t["chain"] == nanobody_chain])
    if idx_ag.size == 0 or idx_nb.size == 0:
        return
    block_ag_nb = pae[np.ix_(idx_ag, idx_nb)]
    block_nb_ag = pae[np.ix_(idx_nb, idx_ag)]

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), dpi=130)
    blocks = (
        (axes[0], block_ag_nb, "antigen union aligned / nanobody scored",
         f"antigen union ({', '.join(ag)})", nanobody_chain, True),
        (axes[1], block_nb_ag, "nanobody aligned / antigen union scored",
         nanobody_chain, f"antigen union ({', '.join(ag)})", False),
    )
    for ax, block, ttl, row_label, col_label, antigen_on_rows in blocks:
        im = ax.imshow(block, cmap="Greens_r", vmin=0, vmax=vmax, interpolation="nearest",
                       aspect="auto")
        ax.set_title(ttl, fontsize=9)
        ax.set_xlabel(f"{col_label} residue", fontsize=9)
        ax.set_ylabel(f"{row_label} residue", fontsize=9)
        ax.tick_params(labelsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).set_label("PAE (A)", fontsize=8)
        cursor = 0
        for ch in ag:
            n = sum(1 for t in tokens if t["chain"] == ch)
            if n == 0:
                continue
            if cursor > 0:
                if antigen_on_rows:
                    ax.axhline(cursor - 0.5, color="black", lw=0.8, alpha=0.55)
                else:
                    ax.axvline(cursor - 0.5, color="black", lw=0.8, alpha=0.55)
            label = f"{ch} {labels.get(ch, '')}".strip()
            if antigen_on_rows:
                y = 1.0 - ((cursor + n / 2) / max(block.shape[0], 1))
                ax.text(-0.02, y, label, transform=ax.transAxes,
                        ha="right", va="center", fontsize=8)
            else:
                ax.text((cursor + n / 2) / max(block.shape[1], 1), -0.08, label,
                        transform=ax.transAxes, ha="center", va="top", fontsize=8)
            cursor += n
    fig.suptitle("Interface PAE (nanobody / full antigen-chain union)", fontsize=11)
    fig.tight_layout()
    try:
        fig.savefig(out_path, bbox_inches="tight")
    finally:
        plt.close(fig)


def plddt_track(tokens, plddt, out_path, nanobody_chain=None, cdr_ranges=None,
                chain_labels=None):
    """Per-residue pLDDT with chain shading and (optionally) CDR highlights.

    cdr_ranges: {"CDR1": (start_token, end_token), ...} in *token index* space.
    """
    labels = chain_labels or {}
    chains = np.array([t["chain"] for t in tokens])
    n = len(tokens)
    fig, ax = plt.subplots(figsize=(12.0, 3.8), dpi=130)
    ax.plot(np.arange(n), plddt, lw=1.0, color="#333333", zorder=3)
    ax.fill_between(np.arange(n), plddt, color="#bcd4f0", alpha=0.55, zorder=2)
    ax.axhline(90, color="#2e7d32", lw=0.8, ls="--", alpha=0.8)
    ax.axhline(70, color="#f9a825", lw=0.8, ls="--", alpha=0.8)
    ax.axhline(50, color="#c62828", lw=0.8, ls="--", alpha=0.8)
    ax.text(n * 0.999, 90, " 90", color="#2e7d32", fontsize=8, va="center")
    ax.text(n * 0.999, 70, " 70", color="#f9a825", fontsize=8, va="center")
    ax.text(n * 0.999, 50, " 50", color="#c62828", fontsize=8, va="center")

    for ch, start, end in _chain_boundaries(chains):
        if start > 0:
            ax.axvline(start - 0.5, color="black", lw=0.8, alpha=0.6)
        ax.text((start + end) / 2, 103, f"{ch} {labels.get(ch, '')}".strip(),
                ha="center", fontsize=9, fontweight="bold")

    if cdr_ranges:
        colors = {"CDR1": "#7b1fa2", "CDR2": "#00796b", "CDR3": "#d84315"}
        for name, (s, e) in cdr_ranges.items():
            if s is None or e is None:
                continue
            ax.axvspan(s, e, color=colors.get(name, "#999999"), alpha=0.18, zorder=1)
            ax.text((s + e) / 2, 8, name, ha="center", fontsize=8,
                    color=colors.get(name, "#555555"), fontweight="bold")

    ax.set_ylim(0, 108)
    ax.set_xlim(0, n - 1)
    ax.set_xlabel("Residue (token index)", fontsize=9)
    ax.set_ylabel("pLDDT", fontsize=9)
    ax.set_title("Per-residue confidence (pLDDT)", fontsize=11)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    try:
        fig.savefig(out_path, bbox_inches="tight")
    finally:
        plt.close(fig)


def ipsae_byres_track(byres_rows, nanobody_chain, out_path, cdr_ranges=None, labels=None):
    """Per-residue ipSAE (psae_d0res) for the nanobody chain as the aligned chain."""
    labels = labels or {}
    rows = [r for r in byres_rows if r["align_chain"] == nanobody_chain]
    if not rows:
        return
    # B-13: 같은 align_chain 이라도 파트너 체인이 다르면 행이 여러 개 나온다.
    # 파트너별로 색을 달리해 겹쳐 그리지 않는다.
    partners = sorted({r.get("scored_chain", "?") for r in rows})
    palette = ["#4a6fa5", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
    fig, ax = plt.subplots(figsize=(12.0, 3.2), dpi=130)
    if len(partners) > 1:
        width = 0.8 / len(partners)
        for k, partner in enumerate(partners):
            sub = [r for r in rows if r.get("scored_chain", "?") == partner]
            x = [r["i"] - 1 + (k - (len(partners) - 1) / 2) * width for r in sub]
            y = [r["psae_d0res"] for r in sub]
            ax.bar(x, y, color=palette[k % len(palette)], width=width,
                   label=f"partner {partner}")
        ax.legend(fontsize=8, ncols=min(len(partners), 4))
    else:
        x = [r["i"] - 1 for r in rows]
        y = [r["psae_d0res"] for r in rows]
        ax.bar(x, y, color=palette[0], width=0.9)
    if cdr_ranges:
        colors = {"CDR1": "#7b1fa2", "CDR2": "#00796b", "CDR3": "#d84315"}
        for name, (s, e) in cdr_ranges.items():
            if s is None or e is None:
                continue
            ax.axvspan(s - 0.5, e + 0.5, color=colors.get(name, "#999999"), alpha=0.18, zorder=0)
            ax.text((s + e) / 2, max(y) * 0.95 if y else 0.9, name, ha="center",
                    fontsize=8, color=colors.get(name, "#555555"), fontweight="bold")
    ax.set_xlabel(f"Residue (token index) - aligned chain {nanobody_chain} "
                  f"({labels.get(nanobody_chain, 'nanobody')})", fontsize=9)
    ax.set_ylabel("per-residue ipSAE", fontsize=9)
    title = "Per-residue ipSAE (interface contribution)"
    if len(partners) > 1:
        title += f" - partner chains: {', '.join(partners)}"
    ax.set_title(title, fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    try:
        fig.savefig(out_path, bbox_inches="tight")
    finally:
        plt.close(fig)


def model_comparison_chart(summary_rows, out_path):
    """Bar chart comparing key metrics across models."""
    if not summary_rows:
        return
    names = [r["model"] for r in summary_rows]
    scopes = {r.get("scope") for r in summary_rows if r.get("scope")}
    scope_label = "union/pair" if len(scopes) > 1 else (
        "union" if scopes == {"antigen-union"} else "pair"
    )
    # 0~1 지표는 왼쪽 축, interface pLDDT(0~100)는 오른쪽 축으로 분리한다 (R4-02)
    score_metrics = [
        (f"ipTM ({scope_label}; nanobody in antigen frame)", "iptm_nb", "#1f77b4"),
        (f"ipSAE ({scope_label}; d0res/max)", "ipsae", "#d62728"),
        ("ipSAE_d0chn (max)", "ipsae_d0chn", "#ff7f0e"),
        ("pDockQ2 (max)", "pdockq2", "#2ca02c"),
    ]
    def finite(v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        return v if np.isfinite(v) else None

    x = np.arange(len(names))
    width = 0.18
    fig, ax = plt.subplots(figsize=(9.5, 4.2), dpi=130)
    for k, (label, key, color) in enumerate(score_metrics):
        vals = [finite(r.get(key)) for r in summary_rows]
        plot_vals = [v if v is not None else 0.0 for v in vals]
        bars = ax.bar(x + (k - 2.0) * width, plot_vals, width, label=label, color=color, alpha=0.9)
        for b, v in zip(bars, vals, strict=False):
            if v is None:
                ax.text(b.get_x() + b.get_width() / 2, 0.015, "N/A", ha="center",
                        va="bottom", fontsize=6.5, color="#666666", rotation=90)
            else:
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center",
                        va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    peak = 0.0
    for _, key, _ in score_metrics:
        for r in summary_rows:
            v = finite(r.get(key))
            if v is not None:
                peak = max(peak, v)
    ax.set_ylim(0, max(1.0, peak * 1.12))
    ax.set_ylabel("score (0-1)", fontsize=9)
    ax.set_title("Model comparison", fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(labelsize=8)

    plddt_vals = [finite(r.get("iface_plddt_nb")) for r in summary_rows]
    if any(v is not None for v in plddt_vals):
        ax2 = ax.twinx()
        # 점수 4개(-2w..+1w) + pLDDT(+2w) 로 모두 1w 간격 (겹침 없음)
        bars2 = ax2.bar(x + 2.0 * width, [v if v is not None else 0.0 for v in plddt_vals],
                        width, label="interface pLDDT (right axis)", color="#9467bd", alpha=0.45)
        for b, v in zip(bars2, plddt_vals, strict=False):
            if v is None:
                ax2.text(b.get_x() + b.get_width() / 2, 1.5, "N/A", ha="center",
                         va="bottom", fontsize=6.5, color="#6a3d9a", rotation=90)
            else:
                ax2.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}", ha="center",
                         va="bottom", fontsize=7, color="#6a3d9a")
        ax2.set_ylim(0, 100)
        ax2.set_ylabel("interface pLDDT (0-100)", fontsize=9, color="#6a3d9a")
        ax2.tick_params(labelsize=8, colors="#6a3d9a")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = (ax2.get_legend_handles_labels() if any(v is not None for v in plddt_vals)
              else ([], []))
    ax.legend(h1 + h2, l1 + l2, fontsize=7.5, ncol=2, loc="upper center")
    fig.tight_layout()
    try:
        fig.savefig(out_path, bbox_inches="tight")
    finally:
        plt.close(fig)
