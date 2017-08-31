#!/usr/bin/env bash
# Create the docs and push them to S3
# -----------------------------------
echo "About to install numpydoc, s3cmd"
pip install -I sphinx==1.3.5 sphinx_bootstrap_theme
pip install numpydoc s3cmd msmb_theme sphinx_rtd_theme nbsphinx
conda list -e
mkdir -p docs/_build
echo "About to build docs"
sphinx-apidoc -f -o docs/source deepchem
sphinx-build -b html docs/source docs/_build
# Copy 
cp -r docs/_build/ website/docs/
echo "About to push docs to s3"
python devtools/travis-ci/push-docs-to-s3.py
