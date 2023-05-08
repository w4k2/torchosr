.PHONY: all clean test

profile:
	kernprof -l -v torchosr/tests/test_common.py

clean:
	find . -name "*.so" -o -name "*.pyc" -o -name "*.md5" -o -name "*.pyd" -o -name "*~" | xargs rm -f
	find . -name "*.pyx" -exec ./tools/rm_pyx_c_file.sh {} \;
	rm -rf coverage
	rm -rf dist
	rm -rf build
	rm -rf docs/_build
	# rm -rf docs/auto_examples
	rm -rf docs/generated
	rm -rf docs/modules
	rm -rf examples/.ipynb_checkpoints

docs: clean install
	cd docs && make html

test-code:
	py.test torchosr

test-coverage:
	rm -rf coverage .coverage
	py.test --cov-report term-missing:skip-covered --cov=torchosr torchosr

test: clean test-coverage

code-analysis:
	flake8 torchosr | grep -v __init__
	pylint -E torchosr/ -d E1103,E0611,E1101

upload:
	python setup.py sdist bdist_wheel
	twine upload dist/*
	pip3 install --upgrade torchosr

install: clean
	python setup.py clean
	python setup.py develop
