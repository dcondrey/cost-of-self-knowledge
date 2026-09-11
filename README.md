<!-- repo-header:start -->
<h3 align="center">Cost Of Self Knowledge</h3>

<p align="center"><strong>The cost of self-knowledge: measuring the real energetic cost of self-introspection across compute substrates</strong></p>

<p align="center">
  <a href="https://github.com/dcondrey/cost-of-self-knowledge/actions/workflows/benchmark.yml"><img src="https://img.shields.io/github/actions/workflow/status/dcondrey/cost-of-self-knowledge/benchmark.yml?style=flat-square&labelColor=20232a&branch=main&label=CI" alt="CI"></a>
  <a href=".bestpractices.json"><img src="https://img.shields.io/badge/best%20practices-evidence%20reviewed-6a4c93?style=flat-square&labelColor=20232a" alt="Best Practices Evidence"></a>
  <a href="https://github.com/sponsors/dcondrey"><img src="https://img.shields.io/badge/GitHub%20Sponsors-Sponsor-EA4AAA?style=flat-square&labelColor=20232a" alt="GitHub Sponsors"></a>
</p>
<!-- repo-header:end -->

---

Two lines of work on one question: what does it cost a system to look at itself?

**The measurement.** `embodied/` holds a cross-substrate benchmark that times self-introspection
at three depths -- shallow in-process, rich in-process, and deep subprocess telemetry -- against
the cost of a unit of real work on the same machine, in that machine's own CPU-seconds. Each run
emits a `benchmark_result.json`; `aggregate.py` collects runs from several substrates into one
table and figure. `benchmark.py` runs it locally, and the Colab and Modal notebooks run it on
hardware you do not own.

**The design question.** `DESIGN.md` and `EXPERIMENT.md` work through ASDS (Asymmetric Substrate
Divergence System): can you build two components such that neither can model the other well
enough to predict or absorb it, while still enforcing real constraints between them? `DESIGN.md`
formalizes the four requirements and tests each for feasibility. `EXPERIMENT.md` is the
pre-registered protocol that decides the open case, with thresholds committed before any run --
including the analytic pre-check that retired the first version's sweep. `sim/` holds the
simulation code and its recorded results.

## Layout

| Path | What it is |
|---|---|
| `embodied/` | cross-substrate introspection-cost benchmark, runners and results |
| `sim/` | ASDS simulation: agent, absorber, environment, experiments, results |
| `DESIGN.md` | ASDS feasibility analysis |
| `EXPERIMENT.md` | pre-registered protocol and decision rule |
| `seeds_conscious.json`, `seeds_embodied.json` | seed sets for the two tracks |

## Status

Research in progress. The documents are explicit about what is analytic and what is empirical;
nothing here is a finished result, and the pre-registration exists so a null answer stays
reportable.
