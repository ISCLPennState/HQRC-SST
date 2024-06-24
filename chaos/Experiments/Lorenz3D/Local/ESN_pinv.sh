#!/bin/bash
#export OMP_NUM_THREADS=12

cd ../../../Methods

for UNITS in 200 300 500 1000
do
for BETA in 1e-4 1e-5 1e-6 1e-7
do
python RUN.py esn \
--mode all \
--display_output 1 \
--system_name SST \
--write_to_log 1 \
--N 427 \
--N_used 427 \
--RDIM 5 \
--noise_level 0 \
--scaler Standard \
--n_nodes $UNITS \
--degree 10 \
--radius 0.9 \
--sigma_input 1 \
--reg $BETA \
--dynamics_length 40 \
--it_pred_length 300 \
--n_tests 3 \
--solver pinv \
--number_of_epochs 1000000 \
--learning_rate 0.001 \
--reference_train_time 10 \
--buffer_train_time 0.5
done
done




