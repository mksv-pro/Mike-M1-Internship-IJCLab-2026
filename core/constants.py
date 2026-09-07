"""Physical constants, defined once and re-exported by the modules that need them.

e2 is the rounded 1.44 MeV fm both solvers run on, not CODATA 1.4399645:
changing it shifts every charged-system result by 2.5e-5 relative and
invalidates the cache.
"""

#: hbar * c, MeV fm.
HBARC = 197.3269804

#: e^2, MeV fm. See the module docstring before changing it.
E2 = 1.44

#: Nucleon mass, MeV. Builds reduced masses from mass numbers.
AMU = 938.918
