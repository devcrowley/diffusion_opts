# Dynamic Diffusion Fragment Optimization

Brief project overview and quick entry point. For full design details, see the whitepaper and architecture docs.

## Purpose
Enable consumer GPUs (e.g., 24GB VRAM) to run and iterate on large diffusion models by profiling a prompt, extracting a smaller "prompt-specific fragment" of the model, and reusing that fragment for rapid multi-seed or paraphrased generations with minimal quality loss.

## Core Idea
1. Profile one (or a small set of) generations to measure attention head and layer usage.
2. Select and materialize a reduced model fragment (pruned heads + optional low-rank compressed layers) that maintains output fidelity for that prompt family.
3. Cache & reuse this fragment to accelerate subsequent image variants and reduce VRAM footprint.

## Why It Matters
- Run larger base models than VRAM would normally allow.
- Accelerate batch & exploration workflows (many seeds / minor prompt tweaks).
- Provide a groundwork for future dynamic or conditional diffusion architectures.

## Key Advantages (Targeted)
- 25–40% VRAM reduction in early milestones, with stretch goals ≥50%.
- 1.5–2× batch throughput speedup after amortizing profiling.
- <5% degradation in perceptual & semantic quality (CLIP / LPIPS metrics).

## Current Status
Documentation & architecture phase. Implementation will proceed through a conservative head-pruning pilot (Phase 1.1) before deeper structural or channel-level pruning.

## High-Level Roadmap (Abbreviated)
- Phase 1: Profiling & head pruning proof-of-concept
- Phase 2: Intelligent extraction, validation & basic caching
- Phase 3: Optimization (memory, performance, advanced caching)
- Phase 4: Production hardening & community release

See: `diffusion-model-optimization-whitepaper.md`, `technical-architecture.md`, `implementation-roadmap.md`.

## Fragment Manifest (Snapshot)
```json
{
  "schema_version": 1,
  "base_model": "sdxl_base_1.0",
  "head_retention_ratio": 0.72,
  "pruning_strategy": "activation_energy_percentile",
  "low_rank_layers": [
    {"layer": "mid.attn.proj_out", "rank": 128, "energy_retained": 0.93}
  ],
  "quality": {"clip_delta_mean": 0.015, "lpips_mean": 0.028},
  "performance": {"peak_vram_full_mb": 18100, "peak_vram_fragment_mb": 13400}
}
```

## Initial Hypotheses (H1–H5)
- H1: Identify ≥30% prunable attention heads with ≤2% CLIP drop.
- H2: Achieve ≥25% VRAM reduction with <5% quality delta.
- H3: Reuse fragment across paraphrases with ≥80% pass rate (stretch after initial 70%).
- H4: ≥1.7× batch speedup for ≥32 variants post-profiling.
- H5: Fragment build amortizes in ≤10 variants.

## Metrics Tracked
CLIP similarity, LPIPS, SSIM, peak VRAM, latency, batch throughput, reuse pass rate, break-even variant count.

## Planned Tech Stack
- PyTorch + Diffusers / ComfyUI custom nodes
- Profiling via forward hooks & attention stats
- Optional low-rank / mixed precision (LoRA-style decomposition)

## Contributing (Future)
Early phases are experimental. Once core pipeline stabilizes a contribution guide and baseline benchmark suite will be published.

## License
TBD (likely Apache 2.0 or MIT once prototype stabilizes).

---
For detailed rationale and deeper architecture, consult the included markdown documents.

## Contact
[devin@devincrowley.com](mailto:devin@devincrowley.com)
