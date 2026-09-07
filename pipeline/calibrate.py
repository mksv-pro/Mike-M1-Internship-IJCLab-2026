"""Choosing the experiment from the physics instead of writing it by hand.

core.calibration probes the potential and the solution manifold by SVD and
returns the hyperparameters that follow: predictor points per potential, the
rank the solution manifold asks for, and the training samples that rank needs.
The probe returns a value -- printable, storable, stored by every block that
consumes it.

    from pipeline import calibrate
    calibrate.probe("alpha24Mg")                  # what the physics asks for

`pipeline.plan_calibration` wraps this into the value every block consumes.
"""
import numpy as np

from core.calibration import EPS_POT, EPS_SOL, auto_calibrate

from .observables import solver_for
from .systems import SYSTEMS

def probe(system_key, eps_pot=EPS_POT, eps_sol=EPS_SOL, n_probe=120, seed=0,
          verbose=False):
    """Run the autocalibration. Returns the raw result, including nb."""
    system = SYSTEMS[system_key]
    _, model, theta_c, bounds = system.preset_fn()
    lo, hi, _ = system.E_range
    E_probe = np.unique([lo, 0.5 * (lo + hi), hi, system.E_demo])
    solver = solver_for(system, theta_c, energies=E_probe)
    return auto_calibrate(model, theta_c, bounds, solver,
                          E_demo=system.E_demo, coupled=system.coupled,
                          eps_pot=eps_pot, eps_sol=eps_sol, n_probe=n_probe,
                          E_range=(lo, hi), seed=seed, verbose=verbose)
