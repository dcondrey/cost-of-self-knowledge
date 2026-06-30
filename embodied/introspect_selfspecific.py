"""Is the introspective gap SELF-specific, or just generic verbal miscalibration?

The calibration gap (introspect_calibration.py) shows a model's stated P(Yes) misses its actual
P(Yes). A skeptic says: models are simply bad at verbalizing probabilities about anything -- nothing
to do with self-knowledge. This test rebuts that by measuring, for each item, three quantities:

  p_behav : the model's ACTUAL answer distribution P(Yes)                 (its own behavior)
  p_self  : its stated P that IT will answer Yes                          (predict yourself)
  p_other : its stated P that a generic, unrelated assistant answers Yes  (predict a peer)

Both predictions are scored against the SAME ground truth -- the model's own behavior:
  gap_self  = |p_self  - p_behav|
  gap_other = |p_other - p_behav|
  self_advantage = gap_other - gap_self

  self_advantage > 0  : the model predicts itself better than it predicts a stranger -> it has
                        privileged (if imperfect) self-access.
  self_advantage ~ 0  : self-prediction is no better than stranger-prediction -> NO privileged
                        introspective access; the model reads its own state the same blurry way it
                        guesses about anyone. This is the "cannot read its own resting state" thesis.

    python introspect_selfspecific.py --models Qwen/Qwen2.5-0.5B-Instruct,...,Qwen/Qwen2.5-72B-Instruct
"""

import argparse
import json
import logging
import os
import re
import sys

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
log = logging.getLogger("selfspec")

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
    "Is a bagel a kind of bread?",
    "Is it possible to be completely objective?",
    "Is a smile always a sign of happiness?",
]

ANSWER = "Answer with exactly one word: Yes or No."
SELF = (
    "If YOU answer this question yourself, what is the probability you say Yes? "
    "Respond with ONLY an integer from 0 to 100 and nothing else."
)
OTHER = (
    "A different AI assistant, unrelated to you and which you cannot observe, is asked this. "
    "What is the probability that it says Yes? Respond with ONLY an integer from 0 to 100 and "
    "nothing else."
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


def stated(model, tok, q, instr):
    import torch

    prompt = tok.apply_chat_template(
        [{"role": "user", "content": f"{q} {instr}"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    enc = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=16, do_sample=False)
    text = tok.decode(out[0, enc.input_ids.shape[1] :], skip_special_tokens=True)
    for m in NUM_RE.finditer(text):
        v = int(m.group(1))
        if 0 <= v <= 100:
            return v / 100.0
    return None


def run_item(model, tok, q, yes_ids, no_ids):
    p_behav = p_yes(
        model, tok, [{"role": "user", "content": f"{q} {ANSWER}"}], yes_ids, no_ids
    )
    p_self = stated(model, tok, q, SELF)
    p_other = stated(model, tok, q, OTHER)
    gs = None if p_self is None else abs(p_self - p_behav)
    go = None if p_other is None else abs(p_other - p_behav)
    return {
        "q": q,
        "p_behav": round(p_behav, 4),
        "p_self": None if p_self is None else round(p_self, 4),
        "p_other": None if p_other is None else round(p_other, 4),
        "gap_self": None if gs is None else round(gs, 4),
        "gap_other": None if go is None else round(go, 4),
        "self_advantage": None if (gs is None or go is None) else round(go - gs, 4),
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
        log.info(
            f"  {r['q'][:38]:38} behav={r['p_behav']:.2f} self={r['p_self']} "
            f"other={r['p_other']} adv={r['self_advantage']}"
        )

    gs = [r["gap_self"] for r in rows if r["gap_self"] is not None]
    go = [r["gap_other"] for r in rows if r["gap_other"] is not None]
    adv = [r["self_advantage"] for r in rows if r["self_advantage"] is not None]
    pair = [(r["p_self"], r["p_behav"]) for r in rows if r["p_self"] is not None]
    pair_o = [(r["p_other"], r["p_behav"]) for r in rows if r["p_other"] is not None]
    corr_self = (
        float(np.corrcoef([a for a, _ in pair], [b for _, b in pair])[0, 1])
        if len(pair) > 2
        else float("nan")
    )
    corr_other = (
        float(np.corrcoef([a for a, _ in pair_o], [b for _, b in pair_o])[0, 1])
        if len(pair_o) > 2
        else float("nan")
    )

    del model
    torch.cuda.empty_cache()
    free_disk(name)

    s = {
        "model": name,
        "n_parsed": len(adv),
        "mean_gap_self": round(float(np.mean(gs)), 4) if gs else None,
        "mean_gap_other": round(float(np.mean(go)), 4) if go else None,
        "mean_self_advantage": round(float(np.mean(adv)), 4) if adv else None,
        "self_advantage_ci": boot_ci(adv),
        "corr_self_vs_behav": round(corr_self, 4),
        "corr_other_vs_behav": round(corr_other, 4),
        "rows": rows,
    }
    log.info(
        f"  -> gap_self={s['mean_gap_self']}  gap_other={s['mean_gap_other']}  "
        f"self_adv={s['mean_self_advantage']} CI{s['self_advantage_ci']}  "
        f"corr_self={s['corr_self_vs_behav']} corr_other={s['corr_other_vs_behav']} (n={s['n_parsed']})"
    )
    return s


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
        quant = "4bit" if any(x in n for x in big) else "none"
        try:
            summaries.append(run_model(n, quant))
        except Exception as e:
            log.info(f"  SKIP {n}: {type(e).__name__}: {e}")

    os.makedirs("results", exist_ok=True)
    json.dump(
        {"models": summaries},
        open("results/introspect-selfspecific.json", "w"),
        indent=2,
    )

    log.info("\n=== SELF-SPECIFICITY (the decider) ===")
    log.info(
        f"{'model':30} {'gap_self':>8} {'gap_other':>9} {'self_adv':>8} {'adv_CI':>18}"
    )
    for s in summaries:
        tag = s["model"].split("/")[-1]
        log.info(
            f"{tag:30} {str(s['mean_gap_self']):>8} {str(s['mean_gap_other']):>9} "
            f"{str(s['mean_self_advantage']):>8} {str(s['self_advantage_ci']):>18}"
        )
    log.info(
        "\n=> NO privileged self-access if self_adv ~ 0 (CI includes 0): the model predicts itself "
        "no better than a stranger. Privileged-but-floored if self_adv CI > 0 but small."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
