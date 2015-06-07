test:
	nosetests gatspy

doctest:
	nosetests --with-doctest gatspy

test-coverage:
	nosetests --with-coverage --cover-package=gatspy

test-coverage-html:
	nosetests --with-coverage --cover-html --cover-package=gatspy
