#!/bin/bash
#export OMP_NUM_THREADS=12
cd ../../../Methods

for RDIM in 5
do
for SS in 80 100 120 200 500 1000 2000
do
for SL in 16
do
for KP in 1.0
do
for L in 1 2 3
do
    python RUN.py rnn_statefull \
    --mode all \
    --display_output 1 \
    --system_name SST \
    --write_to_log 1 \
    --N 427 \
    --N_used 427 \
    --RDIM $RDIM \
    --noise_level 0 \
    --rnn_cell_type lstm \
    --unitary_cplex 1 \
    --unitary_capacity 2 \
    --reg 0.0 \
    --scaler standard \
    --initializer xavier \
    --sequence_length $SL \
    --dropout_keep_prob $KP \
    --zoneout_keep_prob $KP \
    --hidden_state_propagation_length 70 \
    --prediction_length 4 \
    --rnn_activation_str tanh \
    --rnn_num_layers $L \
    --rnn_size_layers $SS \
    --subsample 1 \
    --batch_size 32 \
    --max_epochs 100 \
    --num_rounds 5 \
    --overfitting_patience 20 \
    --training_min_epochs 1 \
    --learning_rate 0.001 \
    --train_val_ratio 0.8 \
    --it_pred_length 300 \
    --n_tests 3 \
    --reference_train_time 1 \
    --buffer_train_time 0.2 \
    --retrain 0
done
done
done
done
done



