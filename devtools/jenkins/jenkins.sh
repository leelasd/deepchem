#!/usr/bin/env bash
envname=`cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 16 | head -n 1`
conda create --name $envname python=2.7
bash scripts/install_deepchem_conda.sh $envname
pip install tensorflow-gpu==0.12.1
source activate $envname
python setup.py install

rm examples/results.csv || true
cd examples/pdbbind
if [ ! -f pdbbind_v2015.tar.gz ]; then
    bash get_pdbbind.sh
fi
cd ..
python pdbbind/pdbbind_datasets.py

source deactivate
conda remove --name $envname --all
