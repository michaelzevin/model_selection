#!/bin/bash/

model0=$1
Nobs=$2

source activate modelselect-py37

python /Users/michaelzevin/research/github/model_selection/model_selection/model_select \
--file-path /Users/michaelzevin/research/model_selection/model_selection/data/spin_models/models.hdf5 \
--gwpath /Users/michaelzevin/research/ligo/O2/PE/GWTC-1_sample_release/ \
--model0 chi05 \
--beta 0.5 0.3 0.2 \
--params mchirp q chieff z \
--Nobs 100 \
--Nsamps 100 \
--weights design_network \
--uncertainty gwevents \
--save-samples \
--name 'test' \
--specific-channels CE CHE GC
#--evidence
#--rate-priors
