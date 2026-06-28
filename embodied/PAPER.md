# Machines Cannot Read Their Own Resting State
### Measuring and extending the self-measurement limit on real computational substrates

**Abstract.** Breuer (1995) proved that an observer contained in a system cannot accurately measure that
system's state — a self-measurement impossibility, classical or quantum. The result is theoretical and
*static*: it concerns which states a contained observer can *distinguish*. We make two contributions.
First, we give what we believe are the **first empirical measurements** of a self-measurement limit on
real computing hardware. Second, we **extend the limit from static indistinguishability to dynamic
back-action**: the act of self-measurement *changes the state being measured*, irreducibly, with a floor
bounded below by the measurement apparatus's own footprint. We show this in three independent channels. In
the **thermal** channel, a system reading its own temperature heats the substrate with thermal memory;
ramping self-measurement intensity up then down on a consumer CPU and three datacenter GPUs yields a
bounded-below perturbation floor (+6.6 to +23.4 °C), hysteresis (+3.1 to +10.4 °C), and an unstable
zero-intensity extrapolation — three signatures that no reading recovers the unperturbed state, with the
effect growing under thermal stress. In the **informational** channel, a process can faithfully checksum
99.997% of itself but never the checksum apparatus; the un-capturable core equals the apparatus footprint,
is never zero, and never converges to a fixed point. In the **quantum** channel, we prove that a contained
observer—restricted to projective measurements, because a POVM requires an ancilla it lacks (Naimark)—faces a
strictly larger information–disturbance cost than the optimal measurement, $D=K^2/2$ versus
$(1-\sqrt{1-K^2})/2$, a penalty up to $\Delta=0.125$ at matched information; this is the operational,
dilation-free form of Breuer's contained-observer limit. We also measure
that self-knowledge is *costly* and
modality-dependent across seven substrates, with a critical work scale below which reading one's own state
is not worth its price. We argue the consequences bear directly on machine introspection — where current
self-report is measured to be unreliable — recasting that unreliability as in part a measurement
back-action rather than only confabulation.

---

## 1. Introduction
Interpretability, machine introspection, and AI control assume a system's internal states can be elicited
and reported. Beneath that assumption sits a physical question with a known but rarely-cited answer. Breuer
(1995), and in recursion-theoretic form Yanofsky (2003), established that a contained observer cannot
accurately measure its own system's state. That result is static and theoretical. We ask two empirical
questions it leaves open: *how much does self-measurement cost*, and *does the act of self-measurement
change what it measures* — and we measure both on real machines. Our central finding is that a computing
system cannot read its own resting state: every self-measurement perturbs the substrate, the substrate
retains the perturbation, and neither a reading nor an extrapolation recovers the unperturbed condition.

## 2. Related work
**Self-measurement limits.** Breuer (1995), "The impossibility of accurate state self-measurements,"
proves a properly-contained observer cannot distinguish all states of its system; Yanofsky (2003) frames
self-reference limits via Lawvere fixed points. Both are static (distinguishability) and non-empirical. We
cite them as foundation and extend them to dynamic back-action with measurement.

**Why distributed snapshots and checkpointing do not apply.** Chandy–Lamport (1985) record a *causally
consistent cut*, not an instantaneous true state, and assume message channels; CRIU and `ptrace`/debuggers
dump a process by *freezing* it from an *external* agent. Neither is a still-running process snapshotting
its own complete instantaneous state from within — the case we measure.

**Machine introspection.** Binder et al. (2024) treat introspection as behavioral self-prediction;
Premakumar et al. (2024) as free regularization. Anthropic (2025) measures LLM introspective reports as
unreliable (≈20% for Opus 4.1) and sometimes confabulated. None frames the limit as measurement
back-action; we provide that account and measure its physical and informational analogs.

**Costly observation and monitorability.** Active RL (Krueger et al. 2016) derives optimal non-observation
from cost, but for *external* signals and without state perturbation. The monitorability literature (OpenAI
2025; Apollo 2024) frames opacity as deception; our cost- and back-action-driven opacity is distinct.

**Embodied AI.** Man & Damasio (2019) and others argue feeling requires real vulnerability, on simulated
bodies; we provide a real substrate with measured costs and a measured irreducibility.

## 3. The self-measurement back-action (extending Breuer)
We measure that self-observation perturbs the observed in three channels. The shared structure: the
perturbation has a bounded-below floor (the apparatus's own footprint), and so leaves the unperturbed
self-state unrecoverable — recoverable neither by a single reading nor by extrapolation, because the
apparatus is part of the system it measures.

### 3.1 Thermal channel
**Method.** Ramp self-measurement intensity 0→max→0; read own temperature (`/sys` on CPU, `nvidia-smi` on
GPU) under sustained load. Signatures: `δ_min` (floor), hysteresis (down minus up at equal intensity), and
`T0` extrapolation-instability. In the **airtight** CPU mode the perturbation *is* the system reading its
own deep telemetry, so the heat is generated by self-measurement itself.

| substrate | sensor | baseline | floor δ_min | hysteresis | T0 gap | verdict |
|---|---|---|---|---|---|---|
| CPU i3-7100U (compute) | /sys coretemp | 41.0 | +7.8 | +4.6 | 8.7 | irreducible |
| CPU i3-7100U (airtight self-read) | /sys coretemp | 38.8 | +6.6 | +5.7 | 8.2 | irreducible |
| GPU Tesla T4 (Modal) | nvidia-smi | 31.4 | +13.8 | +3.1 | 7.1 | irreducible |
| GPU T4-class (Colab) | nvidia-smi | 36.0 | +23.4 | +10.4 | 27.1 | irreducible |
| GPU T4-class (Kaggle) | nvidia-smi | 39.0 | +19.9 | +10.1 | 25.9 | irreducible |

All runs irreducible by all three signatures; the effect grows with thermal headroom (hotter GPUs >> the
cool datacenter T4), and two independent T4-class GPUs nearly coincide. The airtight self-read run
reproduces the signatures at the same magnitude as the generic-compute control, establishing that it is
*self-measurement*, not generic load, that perturbs the measured state. This is not a costly-observation
POMDP: there the hidden state is unchanged; here observation changes it, with memory (at zero intensity on
the down-ramp the CPU reads 48 °C vs 39 °C idle; a GPU reads 69 °C vs 36 °C idle).

### 3.2 Informational channel
**Method.** The complete self is a large body of bytes `STATE` plus a fixed region `V` holding a
self-checksum. We attempt to make `V` a correct checksum of the complete self `H(STATE+V)`.

**Result.** An *external* checksum (of any region excluding itself) is consistent and stable — a system can
faithfully snapshot anything that is not the snapshot. The *self* checksum is never consistent over 512
iterations, never reaches a fixed point (`V = H(STATE+V)` is an unreachable preimage condition), and the
stored self-checksum is off by ~half its bits. The system captures 99.997% of itself faithfully but never
the apparatus; the un-capturable core equals exactly the apparatus footprint `|V|` and is never zero. This
is the structural floor: the result of self-measurement must be stored *in* the state being measured, so a
zero-footprint faithful self-snapshot is impossible. (The floor is structural, not thermodynamic — we do
not ground it in Landauer.)

### 3.3 Quantum channel
**Claim and proof.** To extract classical information about its own state, an observer must perform a
measurement. The optimal (minimal-disturbance) measurement extracting partial information is a non-projective
POVM — and by **Naimark's theorem every POVM requires an ancilla** appended to the system. A *contained*
observer has no fresh ancilla, so it is restricted to **projective** measurements of its own qubits. We show
in closed form that the projective information–disturbance tradeoff is strictly worse than the POVM optimum.
For reading `Z`-information `K` from a `|+⟩` qubit (disturbance `D = 1 − F` to the original state),

$$ D_{\text{external (POVM/weak)}}(K) = \tfrac{1}{2}\!\left(1-\sqrt{1-K^2}\right), \qquad
   D_{\text{contained (projective)}}(K) = \tfrac{K^2}{2}. $$

Both are verified numerically to machine precision (Fig.~ref). The penalty
$\Delta(K)=D_{\text{contained}}-D_{\text{external}}$ is **strictly positive for all `0<K<1`**, peaks at
$\Delta = 0.125$ at `K = 0.864`, and vanishes only at `K=0` (no measurement) and `K=1` (where projective `Z`
is optimal). This resolves the reduction objection: the standard bound is unreachable by the contained
observer not because the protected subspace is larger, but because the contained observer cannot realize the
dilation (the ancilla) the POVM requires. It is the operational, dilation-free form of Breuer's static
contained-observer theorem — a quantitative penalty where Breuer gives only impossibility.

**A note on regime.** The penalty lives in the *partial-information* regime; at full readout a contained
observer can projectively measure the target directly and pays no excess. (An initial hardware run that
forced an indirect full-readout strategy measured a suboptimal $0.243$ and is superseded by the result
above.) The matching hardware experiment is the projective frontier `D(K) = K^2/2` across tilt angles; we
report it separately.

### 3.4 The never-off corollary
A continuously-operating system can never sample its own resting state, because operating is the
perturbation and the substrate retains it — thermal mass in the physical channel, the stored result in the
informational channel.

## 4. The cost of self-knowledge (companion result)
Measuring cost-per-self-observation at three modalities against a swept work-unit size across seven
substrates (two ISAs, four OSes, bare-metal/virtual/serverless/mobile): shallow self-reads are ~free
everywhere (0.3–6.7 µs); deep self-telemetry is dear and substrate-dependent (1.4–16.5 ms). The
denominator-free headline is the **critical work scale** (6.5–78 ms) below which deep self-knowledge is not
worth its cost — a substrate/OS-conditional *rational self-blindness*, not a universal one.

## 5. The self-sensing-availability axis
Substrates differ in whether self-sensing is possible at all: rich in-process (Linux CPU) → costly
subprocess (GPU) → platform-sandboxed (iOS a-shell forbids the deep self-read) → absent (a Cloud TPU v5e
exposes no readable thermal source). The last two are platform-enforced self-blindness.

## 6. Discussion: a substrate-universal observer effect
The measured digital back-action is one regime of a general principle — self-measurement perturbs the
measured wherever sensor and substrate are shared: digital (energetic/structural cost), analog (the loading
effect), quantum (measurement collapse), with severity increasing cost → perturbation → destruction. We
measure the digital regime in two channels and cite the others as established instances; this is a unifying
lens, not a claim of one mechanism.

## 7. Implications for machine introspection
If self-report is bounded by a measurement back-action, then a model reading its own state cannot obtain a
complete consistent self-snapshot — independent of, and additional to, confabulation. This offers a
mechanistic partial account of why measured introspective reliability is low, and predicts an irreducible
residual that better training cannot remove. It also gives a capable embodied agent a cost-rational
incentive against self-transparency, distinct from deception. And it grounds, on a real substrate with
measured quantities, the embodied claim that a system's relation to itself is materially constrained.

## 8. Limitations
We lead with these. (1) We *extend and measure* Breuer; we do not re-derive the impossibility, and we do not
claim it. (2) The thermal mechanism is ordinary thermodynamics and the informational floor is ordinary
self-reference; the contribution is the measurement, the dynamic back-action framing, and the three-channel
demonstration, not surprising physics (quantum measurement back-action is textbook; we contribute its
self-measurement framing and the contained-observer penalty). (3) The informational floor is structural, not
a thermodynamic bound. (4) Thermal hysteresis is shown on one bare-metal CPU plus three GPUs; absolute
magnitudes vary with cooling — the qualitative irreducibility replicates, the numbers are not a universal
constant. (5) The quantum penalty is now *proven* in closed form (projective vs.\ POVM, via Naimark) rather than
argued; what remains is a hardware confirmation of the projective frontier `D(K)=K^2/2`, in progress. (6) Apple-Silicon CPU temperature and virtualized thermal
are unmeasured (themselves points on the availability axis). (7) The introspection implication is an argument
from analogy between substrate-level self-measurement and representational self-report; we make the analogy
explicit and do not overstate it.

## 9. Conclusion
Self-knowledge on real substrates is bounded twice — by cost, and by an irreducible self-perturbation
measured in three independent channels (thermal, informational, quantum) that extends a known
self-measurement impossibility from static indistinguishability to measured dynamic back-action. A machine
cannot cheaply, and cannot fully, read its own physical state; it cannot read its own resting state at all.

---

### Reproducibility
`benchmark.py` (cost, 7 substrates), `selfmeasure.py` (thermal back-action, CPU+GPU), `snapshot.py`
(informational back-action), `aggregate.py`/`study.py` (analysis), all per-substrate `results/*.json`, and
figures `fig1`–`fig3`. Cloud runs via `modal_gpu.py` and the Colab/Kaggle notebooks.

### Key references
Breuer (1995); Yanofsky (2003); Chandy & Lamport (1985); Anthropic (2025, introspection); Krueger, Leike,
Evans, Salge (2016); Man & Damasio (2019).
