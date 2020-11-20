#!/bin/bash/

model0=$1
model0='chi00/alpha10'
Nobs=$2
Nobs=10
betas="0.2 0.2 0.2 0.2 0.2"
channels="CE CHE GC NSC SMT"
params="mchirp q chieff z"

source activate modelselect-py37

python /Users/michaelzevin/research/github/model_selection/model_selection/model_select \
--file-path /Users/michaelzevin/research/model_selection/model_selection/data/spin_models/models_reduced.hdf5 \
--psd-path '/Users/michaelzevin/research/ligo/PSDs/' \
--gw-path '/Users/michaelzevin/research/model_selection/model_selection/data/gw_events/' \
--model0 ${model0} \
--betas ${betas} \
--channels ${channels} \
--params ${params} \
--Nobs ${Nobs} \
--Nsamps 1 \
--sensitivity midhighlatelow_network \
--uncertainty delta \
--save-samples \
--verbose \
--random-seed 11 \
--make-plots \
--name 'example' \
