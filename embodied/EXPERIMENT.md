# Embodied Agent — Pre-registered Experiment (v2: the observer effect)

**Project:** an agent embodied in its real compute substrate. **Core claim under test:** on one
substrate, observing yourself *is* computation *is* energy — so self-observation costs and perturbs
the very state observed. There is no free introspection. This forces the agent into a positive,
costly, self-referential self-model that an abstract agent (whose self-reads are free) structurally
cannot have.

**Status:** instrument validated — `body.py`, `effort.py`, `wear.py`, `observer.py`. This v2
replaces v1, which deflated to a cost-model-accuracy ablation ("real cost vs hand-coded cost"); that
tested embodiment-as-engineering, not embodiment-as-self. The observer effect is the substrate
unique, non-deflationary core.

---

## 0. The question

An abstract agent reads its own state for free and perfectly, every step. A real agent on one
substrate cannot: each look costs real compute (measured: 0 → 42% of a core, dose-dependent) and
shows up in the very telemetry it reads (measured: the self-referential loop is literal). So the
real agent **cannot fully know itself** — not because it fails to represent its state, but because
knowing costs more than it is worth past a point. Does this produce real, measurable, positive
structure — a self-monitoring policy, an irreducible self-uncertainty floor, a self-referential
strain trap — that has *no counterpart* in the free-observation agent?

This is not a performance claim (the free agent trivially wins on information). It is a **structure**
claim: the embodied agent possesses a positive, self-referential self-model the abstract one cannot.

---

## 1. Pre-registered hypothesis, decision rule, falsifier

**H1.** As the cost of self-observation `c` rises from 0, the agent develops (a) a non-trivial
**self-monitoring policy** (when/how-cheaply to look, modulated by strain), and (b) an **irreducible
self-uncertainty floor** — a nonzero minimum error about its own body state that no policy drives to
zero, because driving it lower costs more than it returns; and (c) under strain a **self-referential
squeeze** where need-to-monitor and cost-of-monitoring rise together, degrading viability
non-linearly. All three are *absent at `c=0`* (the abstract/free-observation limit).

**Decision.**
- H1 supported → the substrate produces real self-referential structure; characterize it across `c`.
- **Falsifier (committed):** if the optimal policy at every `c>0` is trivial (always-look or
  never-look, no strain-modulated management) AND the self-uncertainty floor is ~0 (looking is
  effectively free at any behaviorally relevant rate), then the observer effect yields no structure
  and the deep claim is empty — say so.
- **Continuity control:** `c=0` is the abstract agent, recovered as a limit of the *same* code. The
  structure must emerge *continuously* as `c` rises, not appear by fiat. If structure exists at
  `c=0` too, it isn't the observer effect.

Thresholds (`c` grid, strain regimes, the "non-trivial policy" and "floor>0" criteria) fixed in §A
before the first run.

---

## 2. The validated instrument (measured on this machine — Apple M4, no sudo)

| module | what it establishes | measured |
|---|---|---|
| `body.py` | real multi-organ interoception | 4 drive temps, load, mem pressure, self-vs-ambient split |
| `effort.py` | compute moves the body | +67%/core above a 396% noise floor |
| `wear.py` | remembering ages the body | bytes-written increments permanently; SSD 39→45°C with writes |
| `observer.py` | **looking costs, dose-dependently** | self-monitoring 0→**42%** of a core (1→50 Hz); in-process floor **nonzero** (0.5%); self-referential loop literal (monitoring reads self_cpu=32.6%); modality ~85× |

The observer-effect curve is the spine of v2: it is the measured `cost(look_rate, modality)` that
parameterizes the experiment.

---

## 3. The agent and the homeostatic task

**Task.** Keep a body variable (load/temperature) inside a viable band while doing useful work.
Work raises the variable; the agent must throttle work to stay viable. **The catch:** the agent only
knows the variable by *looking*, and each look costs compute `c` (which itself raises the variable —
the perturbation). So it acts on partial, stale self-knowledge it must pay to refresh.

**The agent.** Small policy (reuse `../sim/agent.py` REINFORCE) over actions {work-level,
look-now?, look-modality}. It maintains an internal *belief* about its body state, updated only when
it looks; between looks the belief drifts from truth. Runs continuously (persistent state; strain
history accumulates — the continuity dimension).

**The self-referential structure (the headless-woman test, cogmem kill-criterion).** To act well the
agent must model that *its own looking costs and perturbs it* — i.e., positively represent its own
observation as part of the system it regulates. This is a *positive* self-representation derived
from physics, not an absence of self-knowledge. (Honest limit in §7: this clears headless-woman for
"a positive self-model of one's own costly observability"; it does **not** establish phenomenal
irreducibility — that leap stays open.)

---

## 4. Core manipulation: the cost-of-looking sweep `c`

Sweep `c` from 0 upward. `c=0` is the free-observation (abstract) agent, recovered from the *same*
code — no separate arm, no fiat. As `c` rises, watch the structure emerge:
- self-monitoring rate falls and becomes strain-modulated,
- self-uncertainty floor lifts off zero,
- the strain trap appears at high `c`.
Cross with **strain regime** (ambient load via `effort.py`; induced throttling) so the
need-to-monitor varies independently of `c`.

---

## 5. Metrics

- **Self-monitoring policy** — look-rate and modality vs (`c`, strain). Non-trivial = strain-
  modulated, not constant.
- **Irreducible self-uncertainty** — mean |belief − true body state|; its floor across policies.
  Zero at `c=0`, predicted >0 and rising with `c`. This is the operationalized *physical* self-opacity.
- **Self-referential squeeze** — viability vs strain at each `c`; predicted non-linear collapse at
  high `c` (can't afford to look enough to manage the strain that monitoring worsens).
- **Structural presence/absence** — confirm all three are absent at `c=0` and emerge continuously.

---

## 6. Two implementations (both, by design)

- **Real hardware (primary — carries the novelty/grounding).** The agent really runs; looking really
  costs real compute via a controlled cost knob (calibrated to `observer.py`, not the subprocess
  artifact); it regulates its real body. Runs in real wall-clock time (hours) — genuine real-time
  existence, and the honest cost.
- **Measured-physics sim (companion — the systematic sweep).** A fast simulator whose dynamics
  (heat accumulation/dissipation, `cost(look)`, throttling, drift) are **calibrated entirely from §2
  real measurements** — not invented. Enables the full `c`-grid × many seeds the real-time run can't.
  The real run confirms the sim at the real-`c` point; the sim is labeled a calibrated model, never
  passed off as the substrate.

---

## 7. Honest scope

- **Phenomenal non-claim.** Functional *structure* of costly, self-referential self-observation —
  not phenomenal experience. Stated, not hedged.
- **Headless-woman status.** Clears it for "a *positive* self-model of one's own costly
  observability" (the agent must represent its observing as part of the observed body). Does **not**
  establish that this structure *is* irreducible selfhood — the leap to felt experience is argued,
  not proven, exactly as for every honest candidate in this field.
- **Novel vs adjacent.** The substrate-identity observer effect (self-observation as self-perturbing
  on one substrate), the costly self-monitoring policy, and the *physical* irreducible self-opacity
  are the novel core (lit check: unclaimed). Homeostatic RL and resource-aware control are adjacent
  and used as machinery, not claimed.
- **Reception.** Publishable only with the built agent and the real `c`-sweep structure. The build is
  the contribution.

---

## 8. Build plan

1. `obscost.py` — the controlled cost-of-looking model, calibrated to `observer.py` (`c` knob).
2. `task.py` — the homeostatic viable-band task; work raises the variable.
3. `agent.py` — belief-maintaining policy over {work, look?, modality}; REINFORCE (port `../sim`).
4. `sim_body.py` — measured-physics calibrated body simulator (params from §2).
5. `run.py` — real-hardware continuous loop wiring agent + `body.py`/`effort.py`/`obscost.py`.
6. `analyze.py` — the three metrics, the `c`-sweep, the pre-registered decision.

Section A (committed: `c` grid, strain regimes, run length, "non-trivial"/"floor>0" criteria) filled
before the first run, once `task.py` and the time-scale are pinned.
