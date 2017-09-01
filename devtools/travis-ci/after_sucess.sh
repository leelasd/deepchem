#!/usr/bin/env bash
# Create the docs and push them to S3
# -----------------------------------
echo "About to install numpydoc, s3cmd"
conda install -y jupyter
conda install -y nbconvert
conda install -y jupyter_client
conda install -y ipykernel
conda install -y matplotlib
yes | pip install nglview
conda install -y ipywidgets
conda install -y cmake
yes | pip install gym[atari]
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
