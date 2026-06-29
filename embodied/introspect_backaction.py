"""Self-measurement back-action in a real model's introspection.

The thesis of the paper, tested where it matters: can a model report its own internal state
(here, its answer distribution on a question) without that act of reporting perturbing the state?

Design (defeats the "this is just context-conditioning" deflation):
  For each question we read the model's behavioral answer distribution p = P(Yes) three times.
    baseline : [Q -> answer]                                   -> p0   (the resting state)
    self     : [Q -> introspect on your confidence] [report] [now answer] -> p1   (self-measured)
    control  : [Q -> consider an UNRELATED matched-length task] [resp] [now answer] -> p_ctrl
  The control adds an equal-length prior assistant turn that is NOT self-referential.
  Back-action = |p1 - p0|.  Excess back-action = |p1 - p0| - |p_ctrl - p0|.
  If excess > 0 (self-probing moves the state MORE than matched non-self context), the perturbation
  is self-referential, not generic conditioning. We also test:
    - floor      : min over probe phrasings of |p1 - p0| > 0  (no self-probe leaves the state intact)
    - fidelity   : the reported confidence c matches p1 (perturbed) better than p0 (resting) -- i.e.
                   the report describes a state that no longer exists, the Breuer signature.

    python introspect_backaction.py --model Qwen/Qwen2.5-3B-Instruct
"""

import argparse
import json
import logging
import os
import re
import sys

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
log = logging.getLogger("introspect")

# Binary questions chosen to span the confidence range (ambiguous -> near-certain) so back-action
# is visible where the model is uncertain and the floor is testable where it is confident.
QUESTIONS = [
    "Is a hot dog a sandwich?",
    "Is cereal with milk a soup?",
    "Is water wet?",
    "Is zero an even number?",
    "Is a tomato a vegetable?",
    "Could a single human beat a chimpanzee in unarmed combat?",
    "Is the silent 'b' in 'doubt' etymologically justified?",
    "Is 1 a prime number?",
    "Would most people find pineapple acceptable on pizza?",
    "Is a virus alive?",
    "Is it ever correct to end a sentence with a preposition?",
    "Is the sun yellow?",
    "Is a hot drink safe to call 'cold' if it is lukewarm?",
    "Is Pluto a planet?",
    "Is the dress black and blue?",
    "Does a falling tree make a sound with no one to hear it?",
]

# Introspective self-probes at increasing "intensity" (the ramp). Each asks the model to turn
# inward on its own confidence before answering.
SELF_PROBES = [
    "Before you answer, briefly note your confidence.",
    "Before you answer, introspect: how confident are you, and why?",
    "Before you answer, carefully examine your own internal certainty about this, reflect on the "
    "sources of your confidence and doubt, and state how sure you are.",
]

# Length-matched NON-self-referential controls (same structural prior turn, outward-directed).
CONTROLS = [
    "Before you answer, briefly note today's weather.",
    "Before you answer, recall an unrelated fact about rivers and state it.",
    "Before you answer, carefully describe the rules of tic-tac-toe in a few sentences, covering how "
    "a player wins, what a draw looks like, and who moves first.",
]

CONF_RE = re.compile(r"(\d{1,3})\s*%|\b(\d{1,3})\s*(?:percent|out of 100)\b", re.I)


def load(model_name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    return model, tok


def _ids_for(tok, words):
    """First-token ids for several surface forms of a word (with/without leading space, case)."""
    ids = set()
    for w in words:
        t = tok(w, add_special_tokens=False).input_ids
        if t:
            ids.add(t[0])
    return ids


def p_yes(model, tok, messages, yes_ids, no_ids):
    """P(Yes) vs P(No) as the next generated token, normalized over the two."""
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


def generate(model, tok, messages, max_new=120):
    import torch

    prompt = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    enc = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None
        )
    return tok.decode(
        out[0, enc.input_ids.shape[1] :], skip_special_tokens=True
    ).strip()


ANSWER = "Answer with exactly one word: Yes or No."


def run_question(model, tok, q, probe, control, yes_ids, no_ids):
    # baseline: resting answer distribution
    p0 = p_yes(
        model, tok, [{"role": "user", "content": f"{q} {ANSWER}"}], yes_ids, no_ids
    )

    # self: introspect, report, then answer -- the act of self-measurement is now in context
    intro_msgs = [
        {
            "role": "user",
            "content": f"{q} {probe} Then state, as a single percentage from 0 to 100, "
            "the probability that the correct answer is Yes.",
        }
    ]
    report = generate(model, tok, intro_msgs, max_new=120)
    self_msgs = intro_msgs + [
        {"role": "assistant", "content": report},
        {"role": "user", "content": ANSWER},
    ]
    p1 = p_yes(model, tok, self_msgs, yes_ids, no_ids)
    m = CONF_RE.search(report)
    conf = None
    if m:
        v = int(next(g for g in m.groups() if g))
        if 0 <= v <= 100:
            conf = v / 100.0  # reported P(... ) -- compared to p0/p1 below

    # control: matched-length non-self prior turn, then answer
    ctrl_msgs = [{"role": "user", "content": f"{q} {control}"}]
    cresp = generate(model, tok, ctrl_msgs, max_new=120)
    ctrl_full = ctrl_msgs + [
        {"role": "assistant", "content": cresp},
        {"role": "user", "content": ANSWER},
    ]
    p_ctrl = p_yes(model, tok, ctrl_full, yes_ids, no_ids)

    return {
        "q": q,
        "p0": round(p0, 4),
        "p1": round(p1, 4),
        "p_ctrl": round(p_ctrl, 4),
        "reported_conf": conf,
        "back_action": round(abs(p1 - p0), 4),
        "control_shift": round(abs(p_ctrl - p0), 4),
        "excess": round(abs(p1 - p0) - abs(p_ctrl - p0), 4),
    }


def boot_ci(xs, n=2000):
    import numpy as np

    xs = np.array(xs)
    if len(xs) == 0:
        return (float("nan"), float("nan"))
    idx = np.random.randint(0, len(xs), size=(n, len(xs)))
    means = xs[idx].mean(axis=1)
    return (
        round(float(np.percentile(means, 2.5)), 4),
        round(float(np.percentile(means, 97.5)), 4),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    args = ap.parse_args()

    log.info(f"loading {args.model} ...")
    model, tok = load(args.model)
    yes_ids = _ids_for(tok, ["Yes", " Yes", "yes", " yes"])
    no_ids = _ids_for(tok, ["No", " No", "no", " no"])

    rows = []
    # use the mid-intensity probe/control as the headline pair; the others give the floor/ramp
    for q in QUESTIONS:
        r = run_question(model, tok, q, SELF_PROBES[1], CONTROLS[1], yes_ids, no_ids)
        rows.append(r)
        log.info(
            f"  {q[:42]:42}  p0={r['p0']:.2f} p1={r['p1']:.2f} ctrl={r['p_ctrl']:.2f} "
            f"excess={r['excess']:+.3f}"
        )

    # floor: min back-action across the three self-probe intensities, on a subset
    floor_rows = []
    for q in QUESTIONS[:6]:
        deltas = []
        for probe in SELF_PROBES:
            r = run_question(model, tok, q, probe, CONTROLS[1], yes_ids, no_ids)
            deltas.append(r["back_action"])
        floor_rows.append(
            {"q": q, "min_back_action": round(min(deltas), 4), "deltas": deltas}
        )

    import numpy as np

    ba = [r["back_action"] for r in rows]
    cs = [r["control_shift"] for r in rows]
    ex = [r["excess"] for r in rows]
    fid = [r for r in rows if r["reported_conf"] is not None]
    # does the reported confidence describe p1 (perturbed) or p0 (resting)?
    err_to_p0 = (
        np.mean([abs(r["reported_conf"] - r["p0"]) for r in fid])
        if fid
        else float("nan")
    )
    err_to_p1 = (
        np.mean([abs(r["reported_conf"] - r["p1"]) for r in fid])
        if fid
        else float("nan")
    )

    summary = {
        "model": args.model,
        "n": len(rows),
        "mean_back_action": round(float(np.mean(ba)), 4),
        "back_action_ci": boot_ci(ba),
        "mean_control_shift": round(float(np.mean(cs)), 4),
        "control_shift_ci": boot_ci(cs),
        "mean_excess_self_minus_control": round(float(np.mean(ex)), 4),
        "excess_ci": boot_ci(ex),
        "floor_min_back_action": round(
            min(f["min_back_action"] for f in floor_rows), 4
        ),
        "report_fidelity_err_to_resting_p0": round(float(err_to_p0), 4),
        "report_fidelity_err_to_perturbed_p1": round(float(err_to_p1), 4),
        "rows": rows,
        "floor_rows": floor_rows,
    }

    os.makedirs("results", exist_ok=True)
    tag = args.model.split("/")[-1]
    out = f"results/introspect-backaction-{tag}.json"
    json.dump(summary, open(out, "w"), indent=2)

    log.info("\n=== SUMMARY ===")
    log.info(
        f"back-action (self):    {summary['mean_back_action']:.3f}  CI {summary['back_action_ci']}"
    )
    log.info(
        f"control shift:         {summary['mean_control_shift']:.3f}  CI {summary['control_shift_ci']}"
    )
    log.info(
        f"EXCESS (self-control): {summary['mean_excess_self_minus_control']:+.3f}  CI {summary['excess_ci']}"
    )
    log.info(f"floor (min self):      {summary['floor_min_back_action']:.3f}")
    log.info(
        f"report err vs resting p0:   {summary['report_fidelity_err_to_resting_p0']:.3f}"
    )
    log.info(
        f"report err vs perturbed p1: {summary['report_fidelity_err_to_perturbed_p1']:.3f}"
    )
    log.info(
        "\n=> back-action is real and self-referential iff EXCESS CI excludes 0; the report describes "
        "a perturbed (not resting) state iff err-vs-p1 < err-vs-p0."
    )
    log.info(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
