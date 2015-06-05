#Change Log

## (v0.2) Feature Release

- Improved test coverage & refactored much of the code
- Refactored datasets objects to be picklable for easier parallel analysis
- New feature: ``gatspy.periodic.LombScargleFast`` implements the fast,
  O[N log(N)] periodogram of Press et al.

## (v0.1.1) Bug Fix (28 January 2015)

- Catch NaNs in generated rrlyrae light curves
- correctly handle the period=0 case (previously led to assertion error)

## (v0.1) Initial Release (28 January 2015)

- Single-band: ``LombScargle``, ``LombScargleAstroML``, ``SuperSmoother``
- Multi-band: ``LombScargleMultiband``, ``SuperSmootherMultiband``
- Datasets: Sesar 2010 RR Lyrae, generated RR Lyrae from templates.
