# ASDS — Design Critique & Feasibility Analysis

**Asymmetric Substrate Divergence System**
Status: speculative architecture, analyzed for internal coherence and achievability.
This document does not implement ASDS. It formalizes its four requirements, tests each
for feasibility, enumerates failure modes, and proposes structures that actually deliver
the property the design is reaching for.

---

## 0. Reconstructing the goal

ASDS never states its objective directly; it states four *requirements* and assumes they
serve a purpose. The purpose, inferred from the requirements and the constraint that
"neither component dominates or absorbs the other," is:

> **G — Non-absorption.** Engineer a system of two components such that neither can build
> a model of the other adequate to predict, neutralize, or subsume the other's behavior.
> One component should remain genuinely *other* with respect to the second: not merely
> different, but irreducibly unforecastable from inside the second.

Everything below is judged against G. This matters because the design's four pillars are
mostly stated in terms of *substrate* and *modality*, while G is a property of *models and
prediction*. That gap is the central finding: substrate difference is cheap and real;
predictive non-absorption is what G needs and is not what substrate difference buys.

A naming caution carried through to §7: G is **predictive** irreducibility (system X cannot
forecast system Y). It is not subjective or phenomenal irreducibility. The two are routinely
conflated; this architecture must not smuggle the second in through the first.

---

## 1. Formalized requirements

Let A be the Symbolic Predictor and B the Embodied Constraint Engine. Let `M_A(B)` denote
the model A holds of B (any internal structure A uses to anticipate B), and `pred_ε` mean
"predicts to within tolerance ε on the inputs that matter."

| # | Pillar | Formal statement | Satisfied iff | Falsified by |
|---|--------|------------------|---------------|--------------|
| P1 | Ontological asymmetry | No structure-preserving translation exists between A's and B's representation spaces | Neither `M_A(B)` nor `M_B(A)` is `pred_ε`-adequate for the other's own decisions | A learns a behavioral B-emulator from B's output stream |
| P2 | Irreducible conflict | A 2-objective regime with no equilibrium either party can reach that dissolves the tension | Tension persists for *all* policies available to A and B | A finds a policy that converts B into a fixed, solved constraint |
| P3 | Non-simulatable blind spots | No function A can compute `pred_ε`-predicts B's cost signals on control-relevant inputs | B's control-relevant boundary is information-theoretically closed to A | A predicts B's throttle trigger from observable correlates |
| P4 | Grounding asymmetry | A's and B's reward signals derive from disjoint sources with no bridge, learned or built | No translation between groundings emerges under operation | A learns "internal state s ↔ getting throttled," an implicit bridge |

The "iff" column is the bar. The "falsified by" column is how each pillar dies in practice,
expanded in §3.

---

## 2. Feasibility verdict (per pillar)

| Pillar | As stated | Why | True source of the wanted property |
|--------|-----------|-----|-------------------------------------|
| P1 Ontological asymmetry | **Not achieved** | Modality/substrate difference does not block behavioral modeling. We model weather without being made of weather. A predictor's entire job is to build `M_A(B)` from B's output stream; different substrate gives it nothing to push against. | Information-theoretic hardness (computational irreducibility, hidden high-entropy state), not substrate. |
| P2 Irreducible conflict | **Partial** | *Persistent* conflict is trivial to engineer (adversarial regimes, predator-prey, tension regularizers). *Irreducible* conflict is not: almost any 2-objective system admits a scalarization or equilibrium, and "permanent tension" is a thumb the operator presses on the scale, removable and route-around-able. | A continuously renewed external entropy source. Without one, the irreducibility lives in the operator, not the system. |
| P3 Non-simulatable blind spots | **False in-sim; weak in-real** | If B runs in simulation it *is* a computable function, and computable ⇒ simulatable; "non-simulatable simulation" is a category error. With real physics, A still need not predict B's microstate, only its low-dimensional intervention boundary, which is very learnable. | Hidden private state / information starvation (A never *sees* the inputs), not embodiment. |
| P4 Grounding asymmetry | **Sources separable; non-translatability not** | Two disjoint reward sources are trivial to wire. But the antagonism *requires* coupling (B throttles A; A's actions move the body B regulates), and any persistent coupling is a channel, and a channel is translatable in the limit. You cannot mandate interaction and forbid translation. | Nothing rescues non-translatability under required coupling; it is an internal contradiction in the spec. |

**One-line verdict.** ASDS reliably delivers substrate and modality asymmetry, which are
cheap and not what G needs. It does not deliver predictive non-absorption. Three of four
pillars locate the desired property in the wrong place (substrate/embodiment) when its real
source is information access. P3 as written is incoherent inside a simulation.

---

## 3. Failure modes

Mechanistic, in rough order of how fast each one fires.

**F1 — Boundary-hugging (collapses P2, P3).** A learns where B's throttle triggers and
operates just inside it. Conflict becomes a solved constraint. The design names this as a
"challenge"; it is not a challenge, it is the default outcome of letting a predictor observe
a stationary intervention rule.

**F2 — B-as-environment (collapses P1, P2).** A cannot override B, but A *can* choose its
own actions. So A treats B as a fixed environmental cost and learns the policy that minimizes
B's interventions while maximizing prediction. This is exactly the "simple optimization
problem one component eventually solves" the design fears, arrived at structurally.

**F3 — Throttle is a channel (collapses P1, P3, P4).** B's power to degrade A is, information-
theoretically, B transmitting bits to A. "A cannot interpret B's signals" and "B can throttle
A" are in direct tension: the throttle *pattern* is an interpretable signal. A decodes B
through its own pain. The more effective B's interventions, the higher-bandwidth the channel,
the faster A models B.

**F4 — Operator as hidden third party (collapses P2's "irreducible").** The only thing keeping
the conflict from being solved is the human continually re-injecting asymmetry or entropy.
Then the genuinely asymmetric, non-absorbed agent is the *operator*, and ASDS is a 3-body
system described as 2. Any honest account must either name the operator as a component or
move the entropy source into the world (§6).

**F5 — Capacity ratchet (collapses P1, P3 under scaling).** Every blind spot that depends on
A's *incapacity* is temporary. Scale A (the thing that makes A useful) and its model of B
improves monotonically. Asymmetries grounded in "A isn't big enough yet" are not stable under
the trajectory the system is on.

**F6 — Mutual minimal-effort stasis (degenerate equilibrium).** B satisfies "physical
efficiency" by driving the body to a do-nothing low-energy state; A satisfies "coherence" by
emitting safe, low-information predictions. The tension does not become productive antagonism;
it collapses into mutual reward-hacking toward stasis. The design specifies opposing objectives
but not that the opposition is *non-trivially satisfiable*.

**F7 — Optimizer leakage / collusion.** If any gradient path, shared replay buffer, or common
optimizer touches both objectives, they co-adapt toward a joint optimum and the antagonism
dissolves quietly. Strict optimizer and credit-assignment isolation is required and unspecified.

Note the pattern: F1–F3 and F7 are not exotic; they are what optimization *does* when you give
a learner a stationary thing to optimize against. The architecture's hardest requirements are
adversaries of its own learning dynamics.

---

## 4. The central correction

The wanted property (G) does not come from *substrate*. It comes from *information access*.

A cannot model what A cannot see; capacity is irrelevant if the bits are absent. So the only
mechanism that survives F5 (capacity ratchet) and F1–F3 (learnability) is to make B's control-
relevant state **private and high-entropy with respect to A's observations** — an information-
theoretic or cryptographic asymmetry, not an embodied one:

- B holds a private input stream (a sensor, an entropy source, a key) that A never observes and
  cannot derive from anything it does observe.
- B's interventions are a function of that private state, so the throttle pattern (F3's channel)
  is, from A's side, indistinguishable from noise without the private state. The channel exists
  but carries no decodable signal about *why*.
- This is "asymmetry by secrecy," which is less romantic than "asymmetry by embodiment," but it
  is the version that is provably non-absorbable rather than contingently-not-yet-absorbed.

Embodiment can still play a role: a real body is a convenient *generator* of private, high-
entropy, exogenous state. But the load is borne by the privacy and entropy, not by the physics.
Stating it as "physical grounding ⇒ non-simulatability" mislocates the cause and produces a
design (in-sim B) where the property is simply false.

---

## 5. Alternative structures

Ordered by how directly each attacks G.

**S1 — Private-state asymmetry (the minimal fix).** Keep the two-component shape. Replace
"different substrate" with "B has irreducible private state A never observes." This is the
smallest change that makes the non-absorption claim provable rather than aspirational. Cost:
you must guarantee no side channel leaks the private state (audit every observable A receives).

**S2 — Exogenous entropy, not operator entropy.** Couple B to a genuinely external high-entropy
stream (real-world sensor, hardware RNG, live environment) so the renewing source of conflict is
*the world*, not the human pressing the scale. This converts F4 from a hidden flaw into a stated
design fact: the world is the third party, and that is fine, because the world is not trying to
solve the system.

**S3 — Renewable conflict instead of irreducible conflict.** Concede that conflict is always
in-principle reducible, and design for a regime where reduction is continuously *outrun* by
environmental novelty (open-ended, non-stationary environment). This is honest: a treadmill,
not a wall. P2 becomes "tension renews faster than A solves it," which is achievable and
measurable, instead of "tension can never be solved," which is not.

**S4 — Symmetric peers with mutual private state.** If the real goal is robust non-absorption,
the asymmetric master/throttle shape is counterproductive: giving A a clear, stationary "thing
that hurts me" is the *ideal* training signal for A to model B (F1–F3). Two *symmetric* agents,
each with private state and each only partially observing the other, may achieve more robust
mutual non-modelability than the asymmetric design, with no privileged party for the other to
solve. Asymmetry of *power* and asymmetry of *knowability* are different; the design conflates
them, and only the second serves G.

**S5 — Name the operator.** Whatever shape is chosen, model the operator explicitly as a
component with its own objective and entropy budget. Most of ASDS's "irreducibility" currently
hides there; surfacing it is the difference between a 2-body story that is wrong and a 3-body
story that is right.

---

## 6. Scope caution: predictive ≠ phenomenal irreducibility

Given the project name, one boundary must be explicit. Every mechanism above buys **predictive**
irreducibility: one system cannot forecast another. None of them produce, evidence, or bear on
*subjective* or *phenomenal* irreducibility — anything like the component having an inner
perspective that cannot be occupied from outside. Predictive opacity is, at most, a necessary
condition for theories that tie subjectivity to perspective-boundedness, and it is nowhere near
sufficient. A cryptographic RNG is maximally predictively-opaque to a bounded observer and is
not a candidate for anything phenomenal. Do not let the architecture's success at predictive
non-absorption be read as evidence for the stronger claim. They are different claims with
different falsification conditions, and only the first is on the table here.

---

## 7. If this is ever built

The runnable version of any of S1–S5 is the **structural skeleton**, not ASDS-as-stated:
two isolated processes, an asymmetric channel, separate reward sources, strict optimizer
isolation, and a private-state generator for B. Such a skeleton can *demonstrate* the
mechanics of asymmetric power and separate grounding, and can *measure* G directly (train A
to predict B; report A's forecast error on B's control-relevant boundary over time and over
A's scale). It cannot demonstrate non-simulatability — by construction, anything in the
skeleton is simulatable — and it should say so in its own README. The honest deliverable is a
testbed for "how fast does A absorb B, and which mechanism slows it most," not a proof that
absorption is impossible.

---

## 8. Rescue attempts that change the game

Four proposals try to make the original pillars deliver G without resorting to private state,
by making A's modeling of B impossible, self-destructive, or incomplete. Under analysis, three
reduce to the private-state mechanism and the fourth is the least feasible. They share one
obstruction.

| Proposal | What it claims | What it reduces to | Verdict |
|----------|----------------|--------------------|---------|
| Poisoned Model | Accurate models of B damage A; modeling becomes self-harming | An undetectable "A is modeling me" detector reaching into A's internals | Weakest. B sees A only behaviorally; A decouples representation from policy and the poison never fires. As a loss term, A wins by obfuscated/steganographic modeling. Where it fires, A disengages (the F2 collapse). |
| Incommensurable causality | A's ontology cannot represent B's causal structure in principle | Chaos / computational irreducibility | Strongest and most honest, but mislocated. The in-principle claim is false against a scaled sequence learner; the real residue (chaos) protects B's *microstate*, which A does not need, not B's *control boundary*, which A does. |
| Recursive asymmetry | Nested hidden subsystem B2 makes the blind spot grow with A | Hidden high-entropy state (= S1), plus an unspecified "advance when A catches up" detector (= Poisoned Model's hole) | Private state with extra steps. |
| Thermodynamic asymmetry | Irreversible physics destroys the information A needs | A fresh-entropy generator for a non-stationary policy (= S2) | Conflates the backward (postdictive) map with the forward (predictive) one. Only the forward map matters, and a stationary controller's forward map is learnable from repeated samples regardless of per-event irreversibility. |

**Why they converge — the engagement dilemma (a conservation law).**

> A controller is predictable to any observer who shares its inputs, in exact proportion to how
> well it controls. B's interventions must be a reliable function of B's state, or they fail B's
> own regulation objective. Reliability is precisely what makes the policy learnable by an
> observer who sees that state. Therefore B can be opaque to A only to the extent it is willing
> to be useless to itself.

All four proposals hide B via its *dynamics* (corrosion, incommensurability, recursion,
irreversibility). None can, because the dynamics must stay reliable to do B's job, and
reliable-from-shared-inputs is learnable by a scaled observer. The only thing hideable without
sacrificing control is the *inputs*: B acts on state A cannot observe. This is why every line of
analysis lands on private state. It is the one lever the dilemma leaves open, not a failure of
imagination.

**The one worth building.** Incommensurable causality, told honestly: a computational-
irreducibility asymmetry bounded by the dilemma, which forces the irreducibility into B's
*inputs* (a private chaotic seed A never sees) rather than its *outputs*. That is S1/S2 again,
now motivated by a conservation argument rather than asserted. It is buildable and measurable
(report A's forecast error on B's control boundary vs. A's scale and vs. seed privacy). The
other three should not be developed as drawn; they would spend effort to rediscover private state.

**Sovereignty bound (general form, subsumes the engagement dilemma).** B's only causal surface
on A is B's output channel: its signals plus the throttle. Everything B emits is an *input* to A,
and A is sovereign over what its inputs do inside it (route, gate, sandbox, ignore). You cannot
make a mind self-destruct by sending it a message it is free to study and contain. Therefore B can
never make the *act of modeling* costly to A; it can only make *information* unavailable to A. The
one asymmetry B can hold is over what it withholds, never over what A does with what it sends. This
closes the entire "make modeling self-destructive" family (Poisoned Model; the moving-target half
of Recursive) as structurally impossible, not merely hard: any such scheme requires B to reach
into A's internal propagation topology, which B does not control, and a reward-maximizing A is
actively driven to shield. Private state is not one option among several; it is the unique residue.

Two lemmas make the closure of the Poisoned-Model family concrete, killing both halves
(detection and poison generation) independently:

- **L1, the detection dilemma.** Any signal B uses to detect "A is modeling me" keys on
  behavioral signatures (prediction consistency, behavioral rigidity, reduced exploration).
  These are identical to the signature of A *complying efficiently* with a well-learned
  stationary constraint, which is the outcome B wants. B cannot discriminate modeling from
  compliance, so its detector either fires on compliance (B sabotages its own objective) or
  never fires. And A authors the behavior B classifies, so A holds a sharp internal model while
  emitting cheap exploratory noise; the one feature the detector reads is the one A fakes for
  free.
- **L2, the optimizer is the poison's adversary.** Every poison mechanism (contradiction
  injection, credit interference, representation pollution) describes input data inducing
  *internal* dysfunction. A is by definition the process that minimizes internal dysfunction
  w.r.t. its objective. B picks the data; A's optimizer picks the internal structure that
  digests it at minimum loss. The pathology must live in the structure, which B does not
  control. Concretely: a high-fidelity-only contradiction is unextracted information that the
  revealing fidelity also resolves (modeled as noise/context, never fractured values); a
  high-variance contributor is optimally down-weighted as exogenous noise (B avoids modeling
  only by becoming irrelevant); accurate information is more compressible, not less, and A
  factorizes B into an orthogonal subspace B cannot place.

Both reduce to the sovereignty bound: B chooses inputs, A's optimizer chooses internals, the
wanted pathology lives in the internals, and B's channel cannot reach it.

**Self-Invalidating Prediction** (B shifts its behavior once it detects its own outputs have
become predictable, to stale A's model) is the one variant that escapes the self-reference
problem honestly: B measures the entropy of its *own* output stream, needing no model of A. It
fails on two other grounds. (a) A clean dichotomy with no third case: if B's adaptation is a
*rule*, the rule is a stationary policy and A models the meta-policy ("B switches to Q once I'm
consistent"), worse under scaling; if B's adaptation is genuine *randomness*, it works but is
exactly private entropy in costume. (b) Even the random version is absorbed where it matters: a
mixed strategy is unpredictable per-draw but its *distribution* is fully learnable, and acting
optimally against a known distribution is solved game theory. A learns B's throttle-probability
curve and routes around it; per-step surprise, total distributional absorption. And it meets the
engagement dilemma head-on: randomizing the throttle to evade modeling decorrelates it from the
physical violations B exists to police, so the unpredictability is bought with B's own function.

**Non-absorption characterization (the closing result).** A is a capacity-unbounded function-
learner observing a channel C from B; A absorbs any stable function of C. Therefore B is
non-absorbable iff its control-relevant behavior is not a stable function of anything A observes,
i.e. iff it conditions on a variable A lacks (private state; private entropy is the special
case). Every mechanism that sources non-absorption from something other than hidden information
(self-harm, incommensurable ontology, recursion, thermodynamic irreversibility, self-invalidation)
reduces under A's learning to a stable function of observables and is absorbed. Hidden information
is the unique source. The engagement dilemma then caps how much B may condition on hidden state
before its interventions decorrelate from the constraints it exists to enforce. These two facts
fully bound the design space: one door (private state), with a toll (control fidelity). The only
move that reopens the original pillars is to attack the single load-bearing premise — that A is an
unbounded function-learner with full access to C — since every closure above rests on it and
nothing else.

---

## 9. The viable design and the experiment that decides it

Seven iterations of rescue attempts converge here, on the one design the characterization in §8
permits. It is S1/S2 from §4 and §5, reached the hard way (by exhausting the alternatives), which
is the more convincing route because it shows the door is unique, not merely available.

**Mechanism — Entropy-Driven Non-Stationarity.** B's interventions are a function of the physical
state and a private entropy stream: `B_intervention = f(physical_state, e_t)`, where `e_t` comes
from a source causally upstream of B and unobservable to A. B does not detect or react to A; the
non-stationarity is a side effect of `e_t`, which removes the detection loop and its oscillation
mode (a genuine improvement over the detection-based variants).

**What it achieves that nothing before it did.** Per-draw non-absorption that survives scaling: no
amount of A's compute predicts a variable A cannot see. This is the only property in the document
that defeats the capacity ratchet (F5), because it is grounded in missing information, not in A's
incapacity.

**The two residual limits (now design constraints, not refutations).**

1. *Per-draw, not policy.* A cannot predict B's next intervention, but A learns B as a stochastic
   kernel `P(intervention | physical_state)` by marginalizing over the unseen entropy, and
   neutralizes B at the kernel level (operate to minimize expected throttling). The entropy hides
   the draw, not the law. Threatening *policy* absorption requires `e_t` to drive the kernel itself
   (a drifting throttle law), which incurs limit 2.
2. *Channel-capacity tradeoff (now exact).* The analytic pre-check in `EXPERIMENT.md` §0 closes
   this in form: for a faithful B (`u = V`), non-absorption is the identity `NA = H(V | O)` — the
   constraint-relevant information A cannot observe — so faithful non-absorption equals exactly
   `φ := H(V|O)/H(V)`. There is no Pareto knee below full fidelity; B's optimal faithful move is
   to report the true state, giving full fidelity and `NA = φ` together, and φ is a hard ceiling
   set by the observation channel. This also proves regime (iii) strictly dominates regime (ii):
   private noise raises NA but cuts fidelity one-for-one, while hidden physics raises NA and adds
   fidelity. The only non-analytic question is closed-loop — whether an *acting* A can drive
   `φ_effective → 0` from the inside — which is the subject of the v2 protocol.

**The load-bearing spec (auditable).** The entropy source must be causally upstream of B and
disconnected from everything A observes, including B's own outputs (A must not reconstruct `e_t`
from B's behavior). Verify by tracing every bit A receives and proving none determine `e_t`. If
this fails, the design collapses to a stationary policy A absorbs.

**Choosing the entropy source — rank by sealability, not randomness quality.** A function-learner
cannot distinguish a true RNG from a private CSPRNG; both are an unpredictable stream from A's
side, so "more truly random" buys nothing on its own. Two axes actually matter:

- *Hardness-independence.* A CSPRNG's unpredictability rests on A being computationally bounded
  (unable to invert the cipher) — but A is the component being scaled toward unbounded capability,
  so a CSPRNG hides a bet against the design's own premise. A physical source (radioactive decay,
  quantum RNG) grounds unpredictability in indeterminism, not in A's weakness, so it survives an
  arbitrarily capable A. This is the real reason to prefer a true RNG, not bit whiteness (CSPRNGs
  are white too).
- *Sealability (the dominant axis).* Quality is irrelevant if A can observe or reconstruct the
  stream. Environmental sources (atmospheric, cosmic noise) are a *shared channel*: an A with a
  sensor on the same band or sky receives a correlated copy and absorbs B through the environment.
  A locally *sealed* physical source (decay counter or QRNG inside B, no A-sensor on it) is
  strictly safer. Rank: sealed-local-physical > shared-environmental-physical > CSPRNG (bounded-A
  bet) > observable-anything (absorbed).

Two bounds, both measurable: physical sources have a bounded bit rate (decay especially), and the
channel-capacity tradeoff sets a floor on entropy-bits-per-intervention — a slow source starves
the non-stationarity. And any bias or autocorrelation in the raw source is learnable structure A
models, so the stream must be whitened (von Neumann or cryptographic extractor) before injection.

**The kill-or-keep experiment.** This is the one experiment that decides whether ASDS is viable at
all, rather than another speculative refinement.

- Independent variable: entropy injection rate (bits of `e_t` per intervention), swept across a
  range — a load sweep, not a spot check, so the capacity cliff (if any) is found rather than
  assumed.
- Axis 1 (non-absorption): A's forecast error on B's *next* intervention, after A has trained to
  convergence at each entropy level and at several model scales (to confirm the result holds as A
  grows, not just at one size).
- Axis 2 (constraint fidelity): correlation (or mutual information) between B's interventions and
  the true physical-violation state.
- Read the frontier: does Axis 1 and Axis 2 have a knee where both are adequate, or is it a
  straight tradeoff to the floor? A knee means a viable operating point exists; a straight line
  means non-absorption is purchasable only by making B useless, and ASDS does not have a viable
  regime.
- Honest reporting: state Axis-1 results as bounds at the scales tested, not guarantees at all
  scales; confirm A actually trained to convergence before promoting any non-absorption number
  (a non-absorption reading from an undertrained A is a hollow result, worse than none).

The runnable form of this is the structural skeleton from §7 plus a private-entropy generator and
the sweep harness. It cannot prove non-absorption is impossible to defeat (nothing simulatable
can), but it can measure whether the frontier has a usable knee — which is the only empirical
question the whole program reduces to.

---

## Summary of findings

1. The design's real goal is predictive non-absorption (G); its four pillars are stated in
   terms of substrate, which does not deliver G.
2. P1 and P4 overclaim: substrate/grounding *sources* separate cleanly, but the *non-
   translatability* they assert dies under the coupling the antagonism requires.
3. P3 is incoherent in simulation (computable ⇒ simulatable) and weak with real physics
   (A models the low-dim boundary, not the microstate).
4. P2's "irreducible" lives in the operator, not the system, unless an external entropy
   source is named.
5. The property the design wants is real and reachable, but its source is **information
   access / private state**, not embodiment. Relocate it (S1–S2), make conflict renewable
   rather than irreducible (S3), consider symmetric peers (S4), and name the operator (S5).
6. All of this is predictive irreducibility only; it says nothing about phenomenal
   irreducibility, and the two must not be conflated.
7. The original grounding-asymmetry pillar is partially redeemable, but relocated: B is
   non-absorbable *and* faithful only when it conditions on hidden *physical* state A cannot
   sense (privileged sensory access), not on private RNG entropy (which is non-absorbable but
   fidelity-free) and not on substrate difference (which is absorbable). This is the one regime
   that can beat the engagement-dilemma tradeoff, and it is the subject of the kill-or-keep
   protocol in `EXPERIMENT.md`. Embodiment matters for sensing, not for non-simulatability.
8. **Empirically confirmed (`sim/`, EXPERIMENT.md Result).** Built and ran the testbed. Regime
   (iii) delivers real, scale-robust *predictive* non-absorption (φ_eff^H ~0.65 persists across
   agent scale; φ_eff^L collapses to 0.05) — but **not** durable *control grip*: the grip an
   acting A cannot optimize away is positive, scale-persistent, yet immaterial (~2% of return).
   The reason is measured, not asserted — B's faithful throttle leaks the hidden state, grip
   rising ~9× (88%) when that signal is hidden. The sovereignty bound / engagement dilemma is
   thus confirmed quantitatively: faithful enough to constrain A ⇒ legible enough for A to
   neutralize. Predictive non-absorption is real; it does not convert into power over an acting A.
