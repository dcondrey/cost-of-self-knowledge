# ASDS simulation — kill-or-keep testbed

Implements the protocol in `../EXPERIMENT.md` (v2). Pure numpy, no ML framework dependency.

## Files

| file | role |
|------|------|
| `env.py` | Registered fatigue-limited-throughput environment (Section A). Variants H/L. Committed constants. `Stepper` for policy rollouts. |
| `phi_static.py` | Analytic validation gate (Section A.5): exact recursive grid Bayesian filter for `phi_static`. |
| `absorber.py` | Closed-loop absorber: windowed MLP predicting `u_{t+1}` from A's observable history. |
| `validate_absorber.py` | Milestone 2: confirms `phi_effective` (trained absorber) reproduces `phi_static`. |
| `agent.py` | Active agent A: REINFORCE policy (numpy), with oracle (sees z) mode for the grip metric. |
| `validate_agent.py` | Milestone 3a: confirms the agent learns and grip/collapse behave as predicted. |

## Run

```
python3 phi_static.py          # gate: validates Section A constants
python3 validate_absorber.py   # milestone 2: validates the phi_effective measurement
```

## Status

- **Milestone 1 (PASS).** `phi_static.py`: base-rate 11.9%, `phi_static^H = 0.50`, `phi_static^L ≈ 0`.
  Committed constants: `sigma_w = 0.8352`, `z_crit = 12.59`.
- **Milestone 2 (PASS).** `validate_absorber.py`: under a random agent, `phi_eff^H ≈ 0.55` (≈ static),
  `phi_eff^L ≈ 0.07` (collapses, capacity-driven). The measurement is trustworthy.
- **Milestone 3a (PASS, after env amendment).** First attempt (throttle tax only) exposed a degenerate
  test: the optimal policy was open-loop (`a=1`, eat the tax), so hidden `z` was worthless and active A
  trivially collapsed `phi^H` to 0.23 — a hollow result the gate caught. Fix: added `damage_penalty=2.0`
  (boundary-criticality), so the optimum is interior (`a≈0.4`) and closed-loop `z`-tracking matters.
  Re-validated: H `phi_eff=0.58` / grip 1.19; L `phi_eff=0.13` / grip 0.70. Effects directionally
  correct; grip magnitudes small and the H/L grip contrast not yet sharp at small scale.
- **Milestone 3b (DONE).** `experiment.py`: 3 agent scales x H/L x revelation toggle, 5 seeds, robust
  median+-MAD (one REINFORCE seed diverged at the largest L agent; median absorbs it). The verdict:

      phi_eff^H persists ~0.65 across agent scale ; phi_eff^L collapses 0.14 -> 0.05   (clean)
      grip_H positive & persistent (0.62 -> 0.98) ; grip_L collapses (-0.04 -> -1.37)  (clean)
      grip_H = 1.9% of return  <<  10% materiality bar  -> NOT material
      revelation: grip_H 1.09 (sees u) -> 9.07 (hidden u)  ->  ~88% of grip leaked by B's throttle

## Final result (the ASDS verdict)

Regime (iii) — hidden physical grounding — is exactly the mechanism the whole analytical arc pointed
to, and it **does** deliver real, scale-robust *predictive non-absorption*: an acting, scaling A cannot
predict B's interventions when they depend on genuinely hidden state (phi_eff^H ~0.5-0.65, floored by
the exogenous innovation), while reconstructable state collapses (phi_eff^L -> 0.05).

But it does **not** deliver durable *control grip*. The grip A cannot optimize away is positive and
scale-persistent but immaterial (~2% of return). The reason is measured, not asserted: B's *faithful*
throttle leaks the hidden state through its own bin — grip jumps ~9x (1.1 -> 9.1) when that signal is
hidden from A. ~88% of the potential grip is leaked by B doing its job. That is the engagement dilemma,
quantified: a controller faithful enough to constrain A is thereby legible enough for A to neutralize.

So the deepest ASDS goal — durable non-absorption that gives B power an acting A cannot optimize away —
is **not achieved**. Predictive non-absorption is real; it does not convert into control grip, exactly
because faithfulness and opacity trade off (DESIGN.md sovereignty bound / engagement dilemma).
