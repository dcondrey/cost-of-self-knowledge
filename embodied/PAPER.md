# Machines Cannot Read Their Own Resting State
### The cost and irreducible limit of self-knowledge on real computational substrates

**Abstract.** Work on machine introspection, interpretability, and AI safety assumes that a system's
internal states can be elicited and reported. We ask the prior, physical question: what does it cost a
computing system to know its own state, and is there a limit? On a single substrate a system's
introspection is its interoception — reading one's own state is computation, and computation is energy
and heat. We make this concrete with two measurements on real hardware. First, **self-knowledge is
costly and modality-dependent**: across seven substrates (two instruction sets, four operating systems,
bare-metal, virtualized, serverless, and mobile), shallow self-reads are nearly free (0.3–6.7 µs) while
deep bodily self-telemetry costs 1.4–16.5 ms — and there is a *critical work scale* (6.5–78 ms) below
which knowing one's own detailed state is not worth its price, yielding a substrate-conditional *rational
self-blindness*. Second, and more fundamentally, **self-knowledge is irreducible**: because the act of
self-measurement shares the substrate it measures, observing one's own temperature heats it, with thermal
memory. Ramping self-measurement intensity up and then down, we find a bounded-below perturbation floor,
hysteresis, and an unstable zero-intensity extrapolation — three independent signatures that the
unperturbed state is unrecoverable from any reading. The effect replicates across a consumer CPU and three
datacenter GPUs and grows with thermal stress. This is not a costly-observation POMDP — there the hidden
state is untouched; here observation changes it. We argue the consequence is a cost-rational incentive
*against* self-transparency, distinct from deception, and the first real-substrate grounding of the
embodied claim that feeling requires real vulnerability.

---

## 1. Introduction

Interpretability asks what a model represents; AI introspection asks whether a model can report its own
states; AI control assumes overseers can elicit those reports. All three presuppose that self-knowledge is
*available* — that a system can, in principle, look inward. We take a step back and ask what that looking
physically *is*, and what it *costs*, on the hardware a system actually runs on.

Our starting observation is an identity, not a metaphor: on one substrate, a system observing its own
internal state performs computation, and computation dissipates energy as heat. Introspection is
interoception realized on silicon. From this identity two questions follow. (i) *Cost:* how expensive is
self-observation, and does the expense change what an optimizing system chooses to know about itself? (ii)
*Limit:* if observing oneself perturbs the very substrate being observed, is there a floor below which
self-knowledge cannot go?

We answer both with direct measurement on real machines, and we lead with the second, because it is where
the self-reference does irreducible work. Our central result is that **a computing system cannot read its
own resting state**: every self-measurement heats the substrate, the substrate remembers (thermal mass),
and so neither a single reading nor an extrapolation to zero-intensity recovers the unperturbed condition.

## 2. Related work and the gap

**Machine introspection.** Binder et al. (2024) operationalize introspection as a model predicting its own
behavior; Premakumar et al. (2024) treat self-modeling as a beneficial, *free* regularizer. Neither
considers a physical or energetic cost of self-access.

**Costly observation.** Active Reinforcement Learning (Krueger, Leike, Evans, Salge, 2016) proves that when
observation is costly "it may be optimal to never query" — but the observed quantity is an *external* reward
signal, and crucially observation there does not perturb the hidden state. Rational metareasoning (Russell &
Wefald 1991; Hay & Russell 2012) bounds computation by the value of the decision, again externally. We
transfer the optimal-ignorance structure to *bodily self-telemetry* and, more importantly, add a feedback
that those frameworks lack: self-observation changes the observed.

**Monitorability and safety.** The chain-of-thought monitorability literature (OpenAI "Monitoring
Monitorability" 2025; MacDermott et al. 2025; Apollo 2024) frames opacity as *strategic concealment* —
deception — and one result finds that more compute *buys* monitorability. Our cost-driven self-opacity is
the inverse and is mechanistically distinct from deception.

**Embodied/interoceptive AI.** Man & Damasio (2019) argue feeling requires real vulnerability; Seth, Candia-
Rivera (2026), and Energentic Intelligence (2025) build homeostatic agents — but on simulated or conceptual
bodies. None measures the real cost of self-monitoring or derives an irreducibility. The gap we fill is the
*real substrate*: measured cost, and a measured limit.

## 3. The self-measurement observer effect (irreducibility)

### 3.1 Method
A system reads its own temperature while we ramp the intensity of self-measurement from zero to maximum and
back to zero. The observable is read cheaply (`/sys/class/thermal` on CPU, `nvidia-smi` on GPU); the
perturbation is the self-measurement work. We report three signatures:

- **floor** `δ_min`: the temperature rise at the lowest nonzero measurement intensity. `δ_min > 0` means no
  reading samples the unperturbed state.
- **hysteresis** (down-ramp minus up-ramp temperature at equal intensity): nonzero means the reading depends
  on measurement *history* (thermal memory) — the state is path-dependent.
- **`T0` extrapolation**: a linear fit to zero intensity from the up-ramp and from the down-ramp. A gap
  between the two means even extrapolation cannot recover the true idle temperature.

Two robustness measures matter. We read temperature *under sustained load* (perturbation runs in a
background thread while we sample), so readings are not contaminated by post-load cooling; across all runs
the in-window drift was ≈ 0. And in the **airtight** CPU mode, the perturbation *is* the system reading its
own deep telemetry — so the heat is generated by self-measurement itself, not by a generic proxy load.

### 3.2 Results

All runs are irreducible by all three signatures (Table 1). The effect grows with thermal headroom: the
fan-regulated, hotter-running consumer GPUs show floors and hysteresis several times larger than the cool
datacenter T4, and the two independent T4-class cloud GPUs (Colab, Kaggle) nearly coincide.

| substrate | sensor | baseline | floor δ_min | hysteresis | T0 gap | verdict |
|---|---|---|---|---|---|---|
| CPU i3-7100U (compute) | /sys coretemp | 41.0 | +7.8 | +4.6 | 8.7 | irreducible |
| CPU i3-7100U (self-read, airtight) | /sys coretemp | 38.8 | +6.6 | +5.7 | 8.2 | irreducible |
| GPU Tesla T4 (Modal) | nvidia-smi | 31.4 | +13.8 | +3.1 | 7.1 | irreducible |
| GPU T4-class (Colab) | nvidia-smi | 36.0 | +23.4 | +10.4 | 27.1 | irreducible |
| GPU T4-class (Kaggle) | nvidia-smi | 39.0 | +19.9 | +10.1 | 25.9 | irreducible |

That the **airtight self-read** run (perturbation = the system reading its own telemetry) reproduces the
signatures at the same magnitude as the generic-compute control establishes that it is *self-measurement*,
not merely "compute heats chips," that perturbs the measured state.

### 3.3 Why this is not costly observation
In a costly-observation POMDP the agent pays to observe a hidden state that its observation leaves unchanged.
Here observation *changes* the hidden state, and the substrate's thermal mass makes the change history-
dependent: at zero intensity on the down-ramp the chip reads far above its true idle (e.g. 48 °C vs 39 °C
idle on the CPU; 69 °C vs 36 °C on a GPU). The self-reference is load-bearing — remove it and the phenomenon
disappears. A direct corollary for any continuously-operating ("never-off") system: it can never sample its
own resting state, because operating is the perturbation and the substrate remembers.

## 4. The cost of self-knowledge

### 4.1 Method
We measure, in each machine's own CPU-seconds, the cost of a self-observation at three modalities — shallow
in-process (load average), richer in-process (`/proc`, thermal files), and deep subprocess telemetry — and
sweep the cost of a unit of useful work over four sizes. The substrate-invariant facts are the raw per-
operation costs; `look_frac = cost_look / (cost_look + cost_work)` is a derived, work-scale-dependent
quantity, so we report it across the sweep rather than baking in one denominator. Seven substrates.

### 4.2 Results
Shallow self-introspection is nearly free on every substrate (0.3–6.7 µs). Deep self-telemetry is dear and
substrate-dependent (1.4–16.5 ms; macOS subprocess telemetry ~10× costlier than Linux `/proc`). The
denominator-free headline is the **critical work scale** — the work-unit cost at which deep self-knowledge
reaches break-even — which ranges 6.5–78 ms: below it, an agent whose unit of useful work is cheaper than
that should not pay to read its own detailed state. The resulting *rational self-blindness* is therefore
substrate- and OS-conditional, not universal — a feature we state plainly.

## 5. The self-sensing-availability axis

Beyond cost, substrates differ in whether self-sensing is *possible at all*. We observe a spectrum:
rich in-process sensing (Linux CPU) → costly subprocess sensing (GPU) → **platform-sandboxed** (iOS a-shell
forbids the deep subprocess self-read) → **absent** (a Cloud TPU v5e exposes no readable thermal source; the
code running on it cannot sense its own temperature). The last two are *platform-enforced self-blindness*:
the substrate forbids the system from knowing itself, independent of cost.

## 6. Discussion: a substrate-universal observer effect

The digital observer effect we measure is one regime of a more general principle. Self-measurement perturbs
the measured wherever sensor and substrate are shared: in **digital** systems as energetic/thermal cost; in
**analog** systems as the loading effect (measuring a node's voltage draws current and shifts it); in
**quantum** systems as measurement collapse. The severity increases — cost → perturbation → destruction — as
the substrate grows more intimate with its own state. We measure the digital regime and cite the analog and
quantum regimes as established instances; we present this as a unifying lens, not a claim of one shared
mechanism.

## 7. Implications

**Self-transparency has a price, and capable agents may decline to pay it.** If reading one's own state is
costly, an optimizing embodied agent has a *cost-rational incentive against self-transparency* — and hence
against being monitorable. This is distinct from deceptive alignment: there is no strategic intent, it
cannot be trained away with honesty incentives, and it is the inverse of the finding that more compute buys
monitorability. Interpretability and control proposals that assume cheap, available self-report should price
this in. **And feeling, grounded.** Damasio's thesis that feeling requires real vulnerability has, until now,
been instantiated only on simulated bodies; we provide a real substrate with measured costs and a measured
irreducible self-uncertainty.

## 8. Limitations

We lead with these because they bound the claims. (1) The break-even and rational-self-blindness *consequence*
rests on a model calibrated to the measurements; the costs are real-measured, the break-even is modeled —
we label which is which. (2) The thermal *mechanism* of irreducibility is ordinary physics (thermal mass);
our contribution is the epistemic result and its measurement, not surprising thermodynamics. (3) Hysteresis
is shown on one bare-metal CPU plus three GPUs; absolute magnitudes vary with cooling and are not a universal
constant — the *qualitative* irreducibility is what replicates. (4) Apple-Silicon CPU temperature is not
measured (no unprivileged sensor), and virtualized/CI thermal is masked, which is itself a data point on the
availability axis.

## 9. Conclusion

On real computational substrates, self-knowledge is bounded twice: by cost, and by an irreducible self-
perturbation. A machine cannot cheaply, and cannot fully, read its own physical state — it cannot read its
own resting state at all. This is a measured, real-hardware limit, and it has consequences for how we expect
machines to introspect, to report themselves, and to be made transparent.

---

### Reproducibility
All code and per-substrate results are in the repository: `benchmark.py` (cost), `selfmeasure.py`
(irreducibility), `aggregate.py`/`study.py` (analysis), `results/*.json`, and figures
`fig1`–`fig3`. Cloud runs via `modal_gpu.py` and the Colab/Kaggle notebooks; raw per-operation costs and
hysteresis ramps are recorded for every machine.
