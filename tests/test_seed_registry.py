from collections import Counter

from discovery.keys import normalize_github
from discovery.registry import REGISTRY_DIR, load_models, load_rejected

IN_PAPER = {"gpt2-medium", "gpt2-large", "mdlm-owt", "sedd-small", "langflow-owt", "elf-l-owt"}


def test_models_registry_is_consistent():
    models = load_models(REGISTRY_DIR / "models.yaml")
    ids = [m.id for m in models]
    assert not [i for i, n in Counter(ids).items() if n > 1], "duplicate ids"
    assert {m.id for m in models if m.in_paper} == IN_PAPER
    assert all(m.status == "scored" for m in models if m.in_paper)
    # scored entries outside the paper must have a results file; excluded: below the mainstream threshold or not selected
    assert all(m.status in ("queued", "excluded") or (REGISTRY_DIR.parent / "results" / f"{m.id}.json").exists()
               for m in models if not m.in_paper)
    assert all(normalize_github(m.github) for m in models)
    for m in models:
        if m.checkpoint.kind == "sampler":
            assert m.checkpoint.ref in ids, f"{m.id}: sampler base {m.checkpoint.ref} is not a registry id"
    assert len(models) >= 30


def test_rejected_registry_loads_with_unique_keys():
    rejected = load_rejected(REGISTRY_DIR / "rejected.yaml")
    keys = [r.key for r in rejected]
    assert len(keys) == len(set(keys)) and len(keys) >= 10
    assert all(k.split(":", 1)[0] in {"arxiv", "hf", "github"} for k in keys)
