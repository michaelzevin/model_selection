#!/bin/bash

model0=chi01

for nobs in 10 30 50 100 200 300 500 1000; 
    do cat << EOF > submit_n${nobs}.sh
#!/bin/bash

#SBATCH -A b1095
#SBATCH -p grail-std
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH --output="model_select_n${nobs}.out"

module purge all
module load python/anaconda3.7
source activate modelselect-py37

python /projects/b1095/michaelzevin/github/model_selction/spins/model_select \
-d /projects/b1095/michaelzevin/model_selection/spins/data/detection_weighted/spin_models/ \
-m ${model0} \
-n ${nobs} \
-N 100 \
-gw /projects/b1095/michaelzevin/model_selection/spins/data/gw_events/ \
-p mchirp q chieff z \
-w pdet_designnetwork \
-S \
-b 0.5 0.2 0.3 \
--name ${model0}_${nobs} \
-s gaussian \

EOF
done
