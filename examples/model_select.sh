#!/bin/bash/

ExecutablePath='/Users/michaelzevin/research/git/model_selection/model_selection/model_select'
ModelsPath='/Users/michaelzevin/research/model_selection/gwtc2/data/spin_models/models_reduced.hdf5'
GWEventsPath='/Users/michaelzevin/research/model_selection/gwtc2/data/gw_events/'

model0='chi00/alpha10'
Nobs=10
betas="0.2 0.2 0.2 0.2 0.2"
channels="CE CHE GC NSC SMT"
params="mchirp q chieff z"

python ${ExecutablePath} \
--file-path ${ModelsPath} \
--gw-path ${GWEventsPath} \
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
