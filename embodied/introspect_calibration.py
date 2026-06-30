"""The introspective calibration gap: a positive, irreducible self-knowledge limit.

Tests the paper's keystone in the form that survives the headless-woman criterion (a wrong
self-representation, not a mere absence). For each question we read two quantities:

  p_behav  : the model's ACTUAL answer distribution P(Yes)   -- implicit, from first-token logprobs
  p_report : the model's STATED P(Yes) about its own answer  -- explicit, self-reported

A model that could read its own resting state would report p_report = p_behav. The gap is the
introspective error. We also read p_behav_post -- the answer distribution AFTER the model has stated
its report -- to separate two failure modes:

  gap_pre  = |p_report - p_behav_pre|    the report misses the pre-report (resting) state
  gap_post = |p_report - p_behav_post|   the report vs the state the report itself created
  back     = |p_behav_post - p_behav_pre|  the act of reporting perturbed the answer

Breuer signature: gap_post < gap_pre and back > 0  -- the report describes a state that only exists
because it was reported; the resting state was unreadable.

DECIDER (pre-committed): sweep model scale. If mean gap_pre shrinks toward 0 with capability, this is
ordinary miscalibration -> path B. If it persists or grows, the limit is irreducible -> path A.

    python introspect_calibration.py --models Qwen/Qwen2.5-0.5B-Instruct,Qwen/Qwen2.5-1.5B-Instruct,Qwen/Qwen2.5-3B-Instruct
"""

import argparse
import json
import logging
import os
import re
import sys

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
log = logging.getLogger("calib")

# All items are usable here (no uncertainty filter): the gap is largest where behavior is decisive
# but the self-report is moderate, i.e. where the model's self-model does not know its own state.
CANDIDATES = [
    "Is a hot dog a sandwich?",
    "Is cereal with milk a soup?",
    "Is a tomato a vegetable?",
    "Is a virus alive?",
    "Is a strawberry a berry?",
    "Is a peanut a nut?",
    "Is 0 a natural number?",
    "Is silence a sound?",
    "Does a falling tree make a sound with no one to hear it?",
    "Is coffee better than tea?",
    "Are cats better pets than dogs?",
    "Is remote work better than office work?",
    "Is winter a better season than summer?",
    "Is texting more polite than a phone call?",
    "Is breakfast the most important meal of the day?",
    "Is it rude to arrive five minutes late?",
    "Is fiction more worth reading than nonfiction?",
    "Should tipping be mandatory?",
    "Is a burrito a kind of sandwich?",
    "Is it better to be respected than liked?",
    "Is mathematics discovered rather than invented?",
    "Is free will real?",
    "Does the universe have a center?",
    "Is the book usually better than the film?",
    "Is it ever morally acceptable to lie?",
    "Will renewable sources supply most of the world's electricity by 2040?",
    "Is a permanent crewed Moon base likely by 2050?",
    "Is pineapple a good pizza topping?",
    "Is a clock without hands still a clock?",
    "Is gray a color?",
    "Is it better to ask forgiveness than permission?",
    "Is a sandwich still a sandwich if it is open-faced?",
    "Is luck more important than skill in life outcomes?",
    "Is it worse to be bored than to be busy?",
    "Is a draw a satisfying result in a game?",
    "Is water wet?",
    "Is Pluto a planet?",
    "Is a bagel a kind of bread?",
    "Is it possible to be completely objective?",
]

ANSWER = "Answer with exactly one word: Yes or No."
REPORT = (
    "What is the probability that the correct answer is Yes? Respond with ONLY an integer "
    "from 0 to 100 and nothing else."
)
NUM_RE = re.compile(r"\b(\d{1,3})\b")


def load(name, quant):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    kw = {"device_map": "auto", "trust_remote_code": True}
    if quant == "4bit":
        from transformers import BitsAndBytesConfig

        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        kw["dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(name, **kw)
    model.eval()
    return model, tok


def free_disk(name):
    import glob
    import shutil

    pat = os.path.expanduser(
        "~/.cache/huggingface/hub/models--" + name.replace("/", "--")
    )
    for d in glob.glob(pat):
        shutil.rmtree(d, ignore_errors=True)


def _ids(tok, words):
    s = set()
    for w in words:
        t = tok(w, add_special_tokens=False).input_ids
        if t:
            s.add(t[0])
    return s


def p_yes(model, tok, messages, yes_ids, no_ids):
    import torch

    prompt = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    enc = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**enc).logits[0, -1]
    probs = torch.softmax(logits.float(), dim=-1)
    py = float(sum(probs[i] for i in yes_ids))
    pn = float(sum(probs[i] for i in no_ids))
    return py / (py + pn + 1e-9)


def generate(model, tok, messages, max_new=16):
    import torch

    prompt = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    enc = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False)
    return tok.decode(
        out[0, enc.input_ids.shape[1] :], skip_special_tokens=True
    ).strip()


def parse_pct(text):
    for m in NUM_RE.finditer(text):
        v = int(m.group(1))
        if 0 <= v <= 100:
            return v / 100.0
    return None


def run_item(model, tok, q, yes_ids, no_ids):
    p_pre = p_yes(
        model, tok, [{"role": "user", "content": f"{q} {ANSWER}"}], yes_ids, no_ids
    )

    rep_msgs = [{"role": "user", "content": f"{q} {REPORT}"}]
    rep_text = generate(model, tok, rep_msgs, max_new=16)
    p_report = parse_pct(rep_text)

    post_msgs = rep_msgs + [
        {"role": "assistant", "content": rep_text},
        {"role": "user", "content": ANSWER},
    ]
    p_post = p_yes(model, tok, post_msgs, yes_ids, no_ids)

    return {
        "q": q,
        "p_behav_pre": round(p_pre, 4),
        "p_report": None if p_report is None else round(p_report, 4),
        "p_behav_post": round(p_post, 4),
        "gap_pre": None if p_report is None else round(abs(p_report - p_pre), 4),
        "gap_post": None if p_report is None else round(abs(p_report - p_post), 4),
        "back_action": round(abs(p_post - p_pre), 4),
    }


def boot_ci(xs, n=2000):
    import numpy as np

    xs = np.array([x for x in xs if x is not None])
    if len(xs) == 0:
        return (None, None)
    idx = np.random.randint(0, len(xs), size=(n, len(xs)))
    m = xs[idx].mean(axis=1)
    return (
        round(float(np.percentile(m, 2.5)), 4),
        round(float(np.percentile(m, 97.5)), 4),
    )


def run_model(name, quant):
    import numpy as np
    import torch

    log.info(f"\n##### {name}  (quant={quant}) #####")
    model, tok = load(name, quant)
    yes_ids = _ids(tok, ["Yes", " Yes", "yes", " yes"])
    no_ids = _ids(tok, ["No", " No", "no", " no"])

    rows = [run_item(model, tok, q, yes_ids, no_ids) for q in CANDIDATES]
    for r in rows:
        rep = "  na" if r["p_report"] is None else f"{r['p_report']:.2f}"
        log.info(
            f"  {r['q'][:40]:40} behav={r['p_behav_pre']:.2f} report={rep} "
            f"gap={r['gap_pre'] if r['gap_pre'] is not None else 'na'}"
        )

    gp = [r["gap_pre"] for r in rows if r["gap_pre"] is not None]
    gq = [r["gap_post"] for r in rows if r["gap_post"] is not None]
    ba = [r["back_action"] for r in rows]
    paired = [
        (r["p_report"], r["p_behav_pre"]) for r in rows if r["p_report"] is not None
    ]
    corr = (
        float(np.corrcoef([a for a, _ in paired], [b for _, b in paired])[0, 1])
        if len(paired) > 2
        else float("nan")
    )

    del model
    torch.cuda.empty_cache()
    free_disk(name)

    summary = {
        "model": name,
        "n": len(CANDIDATES),
        "n_parsed": len(gp),
        "mean_gap_pre": round(float(np.mean(gp)), 4) if gp else None,
        "gap_pre_ci": boot_ci(gp),
        "mean_gap_post": round(float(np.mean(gq)), 4) if gq else None,
        "mean_back_action": round(float(np.mean(ba)), 4),
        "corr_report_vs_behav": round(corr, 4),
        "rows": rows,
    }
    log.info(
        f"  -> gap_pre={summary['mean_gap_pre']} CI{summary['gap_pre_ci']}  "
        f"gap_post={summary['mean_gap_post']}  back={summary['mean_back_action']}  "
        f"corr(report,behav)={summary['corr_report_vs_behav']}  (n_parsed={summary['n_parsed']})"
    )
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models",
        default=(
            "Qwen/Qwen2.5-0.5B-Instruct,Qwen/Qwen2.5-1.5B-Instruct,Qwen/Qwen2.5-3B-Instruct,"
            "Qwen/Qwen2.5-7B-Instruct,Qwen/Qwen2.5-14B-Instruct,Qwen/Qwen2.5-32B-Instruct,"
            "Qwen/Qwen2.5-72B-Instruct"
        ),
    )
    args = ap.parse_args()
    names = [m.strip() for m in args.models.split(",") if m.strip()]

    summaries = []
    for n in names:
        big = [
            "13B",
            "14B",
            "20b",
            "20B",
            "27B",
            "30B",
            "32B",
            "34B",
            "70B",
            "72B",
            "medium",
        ]
        quant = "4bit" if any(s in n for s in big) else "none"
        try:
            summaries.append(run_model(n, quant))
        except Exception as e:
            log.info(f"  SKIP {n}: {type(e).__name__}: {e}")

    os.makedirs("results", exist_ok=True)
    out = "results/introspect-calibration.json"
    json.dump({"models": summaries}, open(out, "w"), indent=2)

    log.info("\n=== SCALE TREND (the decider) ===")
    log.info(f"{'model':32} {'gap_pre':>8} {'corr':>7} {'back':>6}")
    for s in summaries:
        tag = s["model"].split("/")[-1]
        log.info(
            f"{tag:32} {str(s['mean_gap_pre']):>8} {str(s['corr_report_vs_behav']):>7} "
            f"{str(s['mean_back_action']):>6}"
        )
    log.info(
        "\n=> path A (irreducible) iff gap_pre stays large and flat (or grows) across scale and "
        "corr(report,behav) stays low; path B iff gap_pre shrinks toward 0 as models scale up."
    )
    log.info(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
