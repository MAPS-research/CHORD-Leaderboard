from datetime import date

from discovery.dedupe import filter_new
from discovery.models import Candidate, Checkpoint, CheckpointRef, Judgement, RegistryEntry, Verdict, Weight
from discovery.registry import known_keys

SEED_WITHOUT_ARXIV = RegistryEntry(
    id="radd-lambda-dce", name="RADD", paper=None, github="https://github.com/ML-GSAI/RADD",
    checkpoint=Checkpoint(kind="hf", ref="JingyangOu/radd-lambda-dce"), family="masked-dlm", train_data="owt",
    status="queued", added=date(2026, 10, 8), source="seed")
MDLM = RegistryEntry(
    id="mdlm-owt", name="MDLM", paper=None, github="https://github.com/kuleshov-group/mdlm",
    checkpoint=Checkpoint(kind="hf", ref="kuleshov-group/mdlm-owt"), family="masked-dlm", train_data="owt",
    status="scored", added=date(2026, 10, 8), source="seed")
KNOWN = known_keys([SEED_WITHOUT_ARXIV, MDLM], [])


def _c(arxiv, repo=None, ckpts=(), repos=()):
    v = Verdict(owt_trained=Judgement(value="yes"), official_repo=repo,
                official_checkpoints=[CheckpointRef(kind="hf", ref=r) for r in ckpts])
    return Candidate(arxiv_id=arxiv, sources=["x"], verdict=v, repos=list(repos))


def test_seed_without_arxiv_is_matched_by_official_repo_after_judging():
    c = _c("2406.03736", repo="https://github.com/ml-gsai/RADD")
    assert filter_new([c], KNOWN, judged=False) == [c]
    assert filter_new([c], KNOWN, judged=True) == []


def test_seed_matched_by_official_checkpoint():
    assert filter_new([_c("2406.03736", ckpts=["jingyangou/RADD-lambda-dce"])], KNOWN, judged=True) == []


def test_mentioning_a_known_repo_or_checkpoint_does_not_dedupe():
    c = _c("2609.00001", repo="https://github.com/new/method", repos=["https://github.com/kuleshov-group/mdlm"])
    assert filter_new([c], KNOWN, judged=True) == [c]


def test_rejected_and_issue_keys_dedupe_by_primary_key():
    c = _c("2609.00002")
    assert filter_new([c], frozenset({"arxiv:2609.00002"}), judged=False) == []


def test_hf_only_rejection_does_not_block_the_paper_that_later_appears():
    known = frozenset({"hf:chen-hao-chao/ermine-owt"})
    paper = _c("2610.00001", ckpts=["chen-hao-chao/ermine-owt"])
    hf_only = Candidate(sources=["hf-search:owt"], weights=[Weight(kind="hf", ref="chen-hao-chao/ermine-owt")])
    exempt = frozenset({"hf:chen-hao-chao/ermine-owt"})
    assert filter_new([paper, hf_only], known, judged=True, paper_exempt=exempt) == [paper]
