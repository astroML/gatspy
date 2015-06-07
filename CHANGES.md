#Change Log

## (v0.3) (not yet releaset)

## (v0.2) Feature Release (07 June 2015)

- Improved test coverage & refactored much of the code
- Refactored datasets objects to be picklable for easier parallel analysis
- New feature: ``gatspy.periodic.LombScargleFast`` implements the fast,
  O[N log(N)] periodogram of Press et al.
- New feature: ``gatspy.periodic.RRLyraeTemplateModeler`` implements a
  template-based fitting method, using the RR Lyrae templates from Sesar (2010)
- sphinx-based documentation build added: http://astroML.org/gatspy/

## (v0.1.1) Bug Fix (28 January 2015)

- Catch NaNs in generated rrlyrae light curves
- correctly handle the period=0 case (previously led to assertion error)

## (v0.1) Initial Release (28 January 2015)

- Single-band: ``LombScargle``, ``LombScargleAstroML``, ``SuperSmoother``
- Multi-band: ``LombScargleMultiband``, ``SuperSmootherMultiband``
- Datasets: Sesar 2010 RR Lyrae, generated RR Lyrae from templates.
