import pytest

from discovery.keys import arxiv_key, github_key, hf_key, normalize_arxiv_id, normalize_github


@pytest.mark.parametrize(
    "raw",
    ["2406.07524", "2406.07524v3", "arXiv:2406.07524", "https://arxiv.org/abs/2406.07524v2",
     "https://arxiv.org/pdf/2406.07524v1.pdf", " 2406.07524 "],
)
def test_arxiv_variants_normalise(raw):
    assert normalize_arxiv_id(raw) == "2406.07524"
    assert arxiv_key(raw) == "arxiv:2406.07524"


@pytest.mark.parametrize(
    "url",
    ["https://github.com/kuleshov-group/mdlm", "https://GitHub.com/Kuleshov-Group/MDLM.git",
     "github.com/kuleshov-group/mdlm/tree/main/scripts", "(https://github.com/kuleshov-group/mdlm).",
     "http://www.github.com/kuleshov-group/mdlm#readme"],
)
def test_github_variants_normalise(url):
    assert normalize_github(url) == "kuleshov-group/mdlm"
    assert github_key(url) == "github:kuleshov-group/mdlm"


@pytest.mark.parametrize("url", ["https://github.com/features/copilot", "https://example.com/a/b", "https://github.com/onlyowner"])
def test_non_repo_github_urls_are_rejected(url):
    assert normalize_github(url) is None
    assert github_key(url) is None


def test_hf_key_lowercases():
    assert hf_key(" Kuleshov-Group/MDLM-OWT ") == "hf:kuleshov-group/mdlm-owt"
