#!/bin/bash/

model0=$1
beta=$2
Nobs=$3

source activate modelselection-py36

python /Users/michaelzevin/research/github/model_selction/spins/model_select \
-d /Users/michaelzevin/research/model_selection/spins/data/detection_weighted/spin_models/ \
-m chi00 \
-n 10 \
-N 100 \
-gw /Users/michaelzevin/research/model_selection/spins/data/gw_events/ \
-p mchirp q chieff z \
-w pdet_designnetwork \
-S \
-b 0.5 0.2 0.3 \
-u gaussian \
#-E
#--name ${model0}_${beta}_${Nobs} \
