import json
import subprocess

from discovery.judge import VERDICT_SCHEMA, build_command, build_user_message, judge, validate_verdict
from discovery.models import Candidate, Weight

ABSTRACT = "We pretrain a 170M masked diffusion model on OpenWebText and report generative perplexity of unconditional samples."
README = "## Checkpoints\nOur OWT model is on the Hub:\n`kuleshov-group/proseco-owt`.\n\nWe initialise from kuleshov-group/mdlm-owt."
CAND = Candidate(arxiv_id="2602.11590", title="ProSeCo", abstract=ABSTRACT, sources=["hf-search:owt"],
                 repos=["https://github.com/kuleshov-group/proseco", "https://github.com/kuleshov-group/mdlm"],
                 readme=README, readme_repo="https://github.com/kuleshov-group/proseco",
                 weights=[Weight(kind="hf", ref="kuleshov-group/proseco-owt", check="ok"),
                          Weight(kind="hf", ref="kuleshov-group/mdlm-owt", check="ok")])


def _raw(**over):
    raw = {
        "owt_trained": {"value": "yes", "evidence": [{"quote": "masked diffusion model on   OpenWebText", "source": "abstract"}]},
        "unconditional": {"value": "yes", "evidence": [{"quote": "generative perplexity of unconditional samples", "source": "abstract"}]},
        "official_repo": "https://github.com/kuleshov-group/proseco",
        "official_checkpoints": [{"kind": "hf", "ref": "kuleshov-group/proseco-owt",
                                  "evidence": [{"quote": "Our OWT model is on the Hub:", "source": "readme"}]}],
        "family_guess": "masked-dlm",
        "generation_type": {"value": "discrete", "evidence": [{"quote": "masked diffusion model", "source": "abstract"}]},
        "notes": "One OWT checkpoint.",
    }
    raw.update(over)
    return raw


def _stdout(raw=None, is_error=False, result="ok"):
    out = {"type": "result", "subtype": "error_during_execution" if is_error else "success", "is_error": is_error, "result": result}
    if raw is not None:
        out["structured_output"] = raw
    return json.dumps(out)


class FakeRunner:
    def __init__(self, outcomes):
        self.outcomes, self.calls = list(outcomes), []

    def __call__(self, cmd, prompt):
        self.calls.append((cmd, prompt))
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def test_whitespace_differences_still_validate():
    v = validate_verdict(_raw(), CAND)
    assert v.owt_trained.value == "yes"
    assert v.owt_trained.evidence[0].url == "https://arxiv.org/abs/2602.11590"
    assert v.official_checkpoints[0].evidence[0].url == "https://github.com/kuleshov-group/proseco"


def test_paraphrased_quote_is_dropped_and_downgraded():
    v = validate_verdict(_raw(owt_trained={"value": "yes", "evidence": [{"quote": "trained on the OWT corpus", "source": "abstract"}]}), CAND)
    assert v.owt_trained.value == "unclear" and v.owt_trained.evidence == []


def test_quote_with_wrong_source_is_relocated():
    v = validate_verdict(_raw(owt_trained={"value": "yes", "evidence": [{"quote": "on OpenWebText", "source": "readme"}]}), CAND)
    assert v.owt_trained.value == "yes" and v.owt_trained.evidence[0].url == "https://arxiv.org/abs/2602.11590"


def test_no_without_evidence_becomes_unclear():
    v = validate_verdict(_raw(unconditional={"value": "no", "evidence": []}), CAND)
    assert v.unconditional.value == "unclear"


def test_checkpoints_need_evidence_or_a_found_weight():
    raw = _raw(official_checkpoints=[
        {"kind": "hf", "ref": "Kuleshov-Group/ProSeCo-OWT", "evidence": []},
        {"kind": "hf", "ref": "made/up-model", "evidence": [{"quote": "not in text", "source": "readme"}]},
    ])
    assert [c.ref for c in validate_verdict(raw, CAND).official_checkpoints] == ["Kuleshov-Group/ProSeCo-OWT"]


def test_official_repo_must_be_a_found_repo_and_family_must_be_known():
    v = validate_verdict(_raw(official_repo="https://github.com/someone/else", family_guess="rnn"), CAND)
    assert v.official_repo is None and v.family_guess is None


def test_user_message_contains_all_inputs():
    msg = build_user_message(CAND)
    for part in ("ProSeCo", "https://arxiv.org/abs/2602.11590", "https://github.com/kuleshov-group/mdlm",
                 "hf kuleshov-group/proseco-owt: ok", ABSTRACT, README):
        assert part in msg


def test_command_disables_tools_and_local_settings():
    cmd = build_command("claude-opus-5-5")
    assert cmd[:2] == ["claude", "-p"] and "--bare" not in cmd
    assert cmd[cmd.index("--model") + 1] == "claude-opus-5-5"
    assert cmd[cmd.index("--tools") + 1] == "" and cmd[cmd.index("--setting-sources") + 1] == ""
    assert "--strict-mcp-config" in cmd and "--no-session-persistence" in cmd
    assert cmd[cmd.index("--output-format") + 1] == "json"
    assert json.loads(cmd[cmd.index("--json-schema") + 1]) == VERDICT_SCHEMA


def test_judge_success_sends_message_on_stdin():
    runner = FakeRunner([_stdout(_raw())])
    out = judge(runner, "claude-opus-5-5", CAND, sleep=lambda _: None)
    assert out.verdict.owt_trained.value == "yes" and out.judge_error is None
    assert runner.calls[0][1] == build_user_message(CAND)


def test_judge_retries_then_records_error():
    delays = []
    runner = FakeRunner([subprocess.TimeoutExpired("claude", 300)] * 3)
    out = judge(runner, "m", CAND, sleep=delays.append)
    assert out.verdict is None and "timed out" in out.judge_error and len(runner.calls) == 3 and delays == [5, 10]


def test_cli_error_result_is_reported():
    runner = FakeRunner([_stdout(is_error=True, result="Invalid API key · Please run /login")] * 3)
    out = judge(runner, "m", CAND, sleep=lambda _: None)
    assert out.verdict is None and "Please run /login" in out.judge_error


def test_missing_or_malformed_structured_output_records_error():
    assert judge(FakeRunner([_stdout(None)] * 3), "m", CAND, sleep=lambda _: None).judge_error
    out = judge(FakeRunner([_stdout({"owt_trained": "yes"})]), "m", CAND, sleep=lambda _: None)
    assert out.verdict is None and out.judge_error


def test_generation_type_is_validated_like_other_judgements():
    v = validate_verdict(_raw(), CAND)
    assert v.generation_type.value == "discrete" and v.generation_type.evidence[0].url == "https://arxiv.org/abs/2602.11590"
    unsupported = validate_verdict(_raw(generation_type={"value": "continuous", "evidence": [{"quote": "embedding flow", "source": "readme"}]}), CAND)
    assert unsupported.generation_type.value == "unclear"


def test_schema_asks_for_one_of_three_generation_types():
    gt = VERDICT_SCHEMA["properties"]["generation_type"]
    assert gt["properties"]["value"]["enum"] == ["ar", "discrete", "continuous", "unclear"]
    assert "generation_type" in VERDICT_SCHEMA["required"]


def test_prompt_asks_for_the_best_checkpoint_only_and_its_sampling_setting():
    from discovery.judge import SYSTEM_PROMPT
    assert "best OpenWebText result" in SYSTEM_PROMPT and "one per model size" in SYSTEM_PROMPT
    assert "not every ablation or hyperparameter variant" in SYSTEM_PROMPT
    assert "sampling setting" in SYSTEM_PROMPT
