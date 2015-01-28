#Change Log

## (v0.1.1) Bug Fix

- Catch NaNs in generated rrlyrae light curves
- correctly handle the period=0 case (previously led to assertion error)

## (v0.1) Initial Release

- Single-band: ``LombScargle``, ``LombScargleAstroML``, ``SuperSmoother``
- Multi-band: ``LombScargleMultiband``, ``SuperSmootherMultiband``
- Datasets: Sesar 2010 RR Lyrae, generated RR Lyrae from templates.
