train_epochs=100
learning_rate=0.001
llama_layers=32

master_port=12350
num_process=1
batch_size=1
d_model=32
d_ff=128

# conda activate tlm
source ~/miniconda3/bin/activate tlm
cd ~/Projects/timellm_data_display/Timellm_package

accelerate launch --use_deepspeed --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port \
    --num_machines 1 --dynamo_backend "no" validate.py \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --factor 3 \
    --enc_in 2 \
    --dec_in 7 \
    --c_out 7 \
    --d_model $d_model \
    --d_ff $d_ff \
    --batch_size $batch_size \
    --learning_rate $learning_rate \
    --llm_layers $llama_layers \
    --train_epochs $train_epochs
