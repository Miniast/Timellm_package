train_epochs=100
learning_rate=0.001
llama_layers=32

master_port=12345
num_process=1
batch_size=2
d_model=32
d_ff=128

# conda activate tlm

accelerate launch --use_deepspeed --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port train.py \
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