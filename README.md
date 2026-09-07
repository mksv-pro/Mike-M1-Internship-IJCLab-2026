# Computational Emulation of Nuclear Dynamics

M1 internship, IJCLab (2026). Coupled-channel DBMM scattering solver and
reduced-order emulators (RBM, LROM) for optical-model cross sections.

## Layout

```
core/          physics + numerics, no I/O (DBMM solver, potentials, special
               functions, RBM / LROM emulators, calibration)
pipeline/      orchestration: system registry, cache, compute blocks, figures,
               tables. Entry points: pipeline.api (notebook use) and
               pipeline.run.Run (declare -> plan -> execute)
tests/         verification suite: frozen reference, physical invariants,
               output contracts, emulator properties
notebooks/     pipeline.ipynb — runnable tour of the API
studies/       stand-alone side analyses (scaling scans, code comparisons)
results/       the computed cache: one .npz per system/block, plus
               _manifest.json (settings and library versions each block was
               produced with) and _calibration.json
figures_out/   the report figures (PNG)
tables_out/    the report tables (LaTeX)
docs/          the report (PDF)
requirements.txt   exact validated environment
```

Systems in `results/`: suite `n40Ca`, `alpha12C_band`, `alpha24Mg`, `n238U`;
controls `p12C`, `alpha_d`. The Coulomb-function cache

## Environment

Developed and validated on **Python 3.14**; exact versions in `requirements.txt`.

```bash
python3.14 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`python-flint` is strongly recommended (Arb backend for the charged Coulomb
functions); without it the chain falls back to `mpmath`, ~10x slower cold, same
results. `numba` is optional and not in the default path.

## Reproduce

```bash
python -m pytest -q                                       # ~82 tests, frozen reference
python -c "from pipeline import api; api.report()"        # figures + tables from the cache
python -c "from pipeline import api; print(api.check())"  # confirm no cached number moved
```

Full recompute from scratch (clears and rebuilds `results/`):
`from pipeline.run import Run; Run().execute()` — order of hours.

## Not included

LaTeX sources of the report (PDF only); the third-party ROSE / LROM packages
(P. Giuliani) required by `studies/compare_rose*.py`.
