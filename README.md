# CHORD Leaderboard

**https://maps-research.github.io/CHORD-Leaderboard/**

The CHORD Leaderboard ranks open-ended text generators by their [CHORD](https://arxiv.org/abs/2609.34240)
distance to the human reference on unconditional generation on OpenWebText; lower is closer.

CHORD (**C**oherence-aware **H**idden-state **O**pen-generation **R**eference **D**istance) is a
coherence-sensitive distributional metric. It encodes generated and human-written corpora in the hidden-state
space of a frozen LLM using a coherence-eliciting prompt, and compares the resulting distributions using MMD
with an RBF kernel. Generators are tagged as AR, discrete diffusion/flow, or continuous diffusion/flow, and
each is evaluated at the setting its paper reports as best.

## Request a model

Want a generator evaluated, or evaluated sooner? Open a
[model evaluation request](https://github.com/MAPS-research/CHORD-Leaderboard/issues/new?template=model-request.yml).
The leaderboard lists generators trained on OpenWebText with public weights and an official code repository.

## Links

- Paper: [Coherence-Aware Distributional Evaluation of Open-Ended Text Generation](https://arxiv.org/abs/2609.34240)
- Code: [MAPS-research/CHORD](https://github.com/MAPS-research/CHORD) · `pip install chord-metric`
- Maintainers: see [docs/maintainers.md](docs/maintainers.md)

## Citation

```bibtex
@misc{liu2026coherenceawaredistributionalevaluationopenended,
      title={Coherence-Aware Distributional Evaluation of Open-Ended Text Generation},
      author={Jinnuo Liu and Junhao Zhu and Weifeng Jiang and Haoming Liu and Hongyi Wen},
      year={2026},
      eprint={2609.34240},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2609.34240},
}
```
