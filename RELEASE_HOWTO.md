# How to Release

Here's a quick step-by-step for cutting a new release of gatspy.

## Pre-release

1. update version in ``gatspy.__version__``

2. update version in ``docs/conf.py`` (two places!)

3. update change log in ``CHANGES.md``

4. create a release tag; e.g.
   ```
   $ git tag -a v0.2 -m 'version 0.2 release'
   ```

5. push the commits and tag to github

6. confirm that CI tests pass on github

7. under "tags" on github, update the release notes

8. push the new release to PyPI:
   ```
   $ python setup.py sdist upload
   ```

9. change directories to ``doc`` and build the documentation:
   ```
   $ cd doc/
   $ make html     # build documentation
   $ make publish  # publish to github pages
   ```

## Post-release

1. update version in ``gatspy/__init__.py`` to next version; e.g. '0.3-git'

2. update version in ``doc/conf.py`` to the same (in two places)
