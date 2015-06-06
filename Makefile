test:
	nosetests gatspy

doctest:
	nosetests --with-doctest gatspy

test-coverage:
	nosetests --with-coverage --cover-package=gatspy gatspy
