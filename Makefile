test:
	nosetests gatspy

test-docstrings:
	nosetests --with-doctest gatspy

test-coverage:
	nosetests --with-coverage --cover-package=gatspy gatspy
