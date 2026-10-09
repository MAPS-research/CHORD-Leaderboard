"""Pipeline configuration."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")


class Config(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    repo: str
    lookback_days: int = 21
    arxiv_categories: list[str]
    arxiv_query_terms: list[str]
    hf_queries: list[str]
    prefilter_terms: list[str]
    judge_model: str = "claude-opus-5-5"
    max_judge_calls: int = 60
    judge_workers: int = 4
    max_readme_chars: int = 30000


def load_config(path: Path | None = None) -> Config:
    data = yaml.safe_load((path or DEFAULT_CONFIG).read_text(encoding="utf-8"))
    return Config.model_validate(data)
