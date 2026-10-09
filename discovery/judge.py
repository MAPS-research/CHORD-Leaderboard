"""Ask Claude, through the Claude Code CLI, whether a candidate meets the criteria; keep only verifiable evidence."""

import json
import subprocess
import time
from typing import get_args

from discovery.keys import github_key
from discovery.models import Candidate, CheckpointRef, Evidence, Family, Judgement, TypeJudgement, Verdict, WeightKind

SYSTEM_PROMPT = """\
You screen research papers for the CHORD Leaderboard, which scores open-ended text generators under a strict protocol.
A checkpoint qualifies only if ALL of the following hold:
1. It was pretrained on OpenWebText (OWT or OpenWebText2).
2. It can generate text unconditionally, i.e. from scratch without a prompt.
3. Its weights are publicly downloadable.
4. The authors publish an official GitHub repository.

Judge ONLY from the ABSTRACT and README given by the user. Rules:
- Support every judgement with quotes copied character-for-character from the ABSTRACT or README, and name the source of each quote. Do not paraphrase, shorten with ellipses, translate, or fix typos inside a quote.
- If the text does not settle a question, answer "unclear" with no quotes. Never rely on outside knowledge about the method or model family.
- owt_trained: "yes" if the text states the released or evaluated model was trained on OpenWebText/OWT; "no" if it states training only on other corpora.
- unconditional: "yes" if the text reports unconditional samples or the generative perplexity of samples generated from scratch, or the README shows a sampling command without a prompt; "no" if the model only supports conditional tasks.
- official_checkpoints: only checkpoints the authors present as their own release for an OWT-trained model. Models in a Hugging Face collection that the README links (they appear in WEIGHT LINKS FOUND) count as presented by the authors; support them with the README sentence that links the collection or describes those checkpoints. Exclude datasets, checkpoints the work starts from or compares against, and checkpoints trained on other corpora.
- official_repo: the authors' own repository for this paper, chosen from REPOSITORIES; null if none is clearly theirs.
- generation_type: how the model generates text, as one of three types: "ar" for autoregressive (left-to-right next-token) models; "discrete" for diffusion or flow models over discrete tokens (masked, absorbing, uniform, block or hybrid discrete diffusion, discrete flow matching); "continuous" for diffusion or flow models in a continuous space (token embeddings, the simplex, or a latent space). Quote the text that states it; answer "unclear" if the text does not say.
- family_guess: one of ar, masked-dlm, uniform-dlm, hybrid-dlm, block-hybrid, continuous, flow, distilled; null if unclear.
- notes: at most two sentences for the human reviewer, e.g. which checkpoints are OWT-trained or which sampler settings the paper uses.
Reply only with the JSON object required by the output schema."""

_EVIDENCE = {
    "type": "array",
    "items": {"type": "object", "properties": {"quote": {"type": "string"}, "source": {"enum": ["abstract", "readme"]}},
              "required": ["quote", "source"]},
}
_JUDGEMENT = {"type": "object", "properties": {"value": {"enum": ["yes", "no", "unclear"]}, "evidence": _EVIDENCE},
              "required": ["value", "evidence"]}
VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "owt_trained": _JUDGEMENT,
        "unconditional": _JUDGEMENT,
        "official_repo": {"type": ["string", "null"]},
        "official_checkpoints": {"type": "array", "items": {
            "type": "object",
            "properties": {"kind": {"enum": list(get_args(WeightKind))}, "ref": {"type": "string"}, "evidence": _EVIDENCE},
            "required": ["kind", "ref", "evidence"]}},
        "family_guess": {"type": ["string", "null"]},
        "generation_type": {"type": "object", "properties": {"value": {"enum": ["ar", "discrete", "continuous", "unclear"]},
                                                             "evidence": _EVIDENCE}, "required": ["value", "evidence"]},
        "notes": {"type": "string"},
    },
    "required": ["owt_trained", "unconditional", "official_repo", "official_checkpoints", "family_guess", "generation_type", "notes"],
}
ATTEMPTS = 3
BASE_DELAY_S = 5
TIMEOUT_S = 300


class JudgeError(RuntimeError):
    pass


def build_user_message(cand: Candidate) -> str:
    repos = "\n".join(f"- {r}" for r in cand.repos) or "- none found"
    weights = "\n".join(f"- {w.kind} {w.ref}: {w.check}" for w in cand.weights) or "- none found"
    arxiv = f"https://arxiv.org/abs/{cand.arxiv_id}" if cand.arxiv_id else "none"
    return (f"TITLE: {cand.title or 'none'}\nARXIV: {arxiv}\nREPOSITORIES:\n{repos}\n"
            f"WEIGHT LINKS FOUND (automatic check result):\n{weights}\n\n"
            f"ABSTRACT:\n<<<\n{cand.abstract}\n>>>\n\nREADME (from {cand.readme_repo or 'none'}):\n<<<\n{cand.readme}\n>>>")


def build_command(model: str) -> list[str]:
    return ["claude", "-p", "--model", model, "--tools", "", "--no-session-persistence",
            "--setting-sources", "", "--strict-mcp-config", "--system-prompt", SYSTEM_PROMPT,
            "--json-schema", json.dumps(VERDICT_SCHEMA), "--output-format", "json"]


def run_claude(cmd: list[str], prompt: str) -> str:
    proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=TIMEOUT_S)
    if proc.returncode != 0:
        raise JudgeError(f"claude exited with {proc.returncode}: {(proc.stdout or proc.stderr)[-500:]}")
    return proc.stdout


def _parse(stdout: str) -> dict:
    out = json.loads(stdout)
    if out.get("is_error") or out.get("subtype") != "success":
        raise JudgeError(f"claude reported an error: {str(out.get('result'))[:500]}")
    data = out.get("structured_output")
    if not isinstance(data, dict):
        raise JudgeError("claude returned no structured_output")
    return data


def _norm(text: str) -> str:
    return " ".join(text.split())


def validate_verdict(raw: dict, cand: Candidate) -> Verdict:
    texts = {"abstract": _norm(cand.abstract), "readme": _norm(cand.readme)}
    urls = {"abstract": f"https://arxiv.org/abs/{cand.arxiv_id}" if cand.arxiv_id else "", "readme": cand.readme_repo or ""}

    def evidence(items: list[dict]) -> list[Evidence]:
        out = []
        for item in items:
            quote = _norm(item["quote"])
            where = next((s for s in (item.get("source"), "abstract", "readme") if s in texts and quote and quote in texts[s]), None)
            if where:
                out.append(Evidence(quote=quote, url=urls[where]))
        return out

    def judgement(d: dict) -> Judgement:
        ev = evidence(d["evidence"])
        return Judgement(value=d["value"] if ev else "unclear", evidence=ev)

    found = {w.ref.lower() for w in cand.weights}
    checkpoints = []
    for c in raw["official_checkpoints"]:
        ev = evidence(c["evidence"])
        if ev or c["ref"].lower() in found:
            checkpoints.append(CheckpointRef(kind=c["kind"], ref=c["ref"], evidence=ev))
    repo_keys = {github_key(r) for r in cand.repos}
    repo = raw["official_repo"] if raw["official_repo"] and github_key(raw["official_repo"]) in repo_keys else None
    family = raw["family_guess"] if raw["family_guess"] in get_args(Family) else None
    gt = raw["generation_type"]
    gt_evidence = evidence(gt["evidence"])
    generation_type = TypeJudgement(value=gt["value"] if gt_evidence else "unclear", evidence=gt_evidence)
    return Verdict(owt_trained=judgement(raw["owt_trained"]), unconditional=judgement(raw["unconditional"]),
                   official_repo=repo, official_checkpoints=checkpoints, family_guess=family,
                   generation_type=generation_type, notes=raw.get("notes", ""))


def _call(runner, model: str, cand: Candidate, sleep) -> dict:
    last: Exception | None = None
    for i in range(ATTEMPTS):
        try:
            return _parse(runner(build_command(model), build_user_message(cand)))
        except Exception as exc:  # CLI failure, timeout, bad JSON, error result: retry, then report on the issue
            last = exc
            if i < ATTEMPTS - 1:
                sleep(BASE_DELAY_S * 2**i)
    raise JudgeError(f"judge failed after {ATTEMPTS} attempts: {last}") from last


def judge(runner, model: str, cand: Candidate, sleep=time.sleep) -> Candidate:
    try:
        verdict = validate_verdict(_call(runner, model, cand, sleep), cand)
    except Exception as exc:  # JudgeError, KeyError/TypeError/ValidationError from malformed output
        return cand.model_copy(update={"verdict": None, "judge_error": f"{type(exc).__name__}: {exc}"})
    return cand.model_copy(update={"verdict": verdict, "judge_error": None})
