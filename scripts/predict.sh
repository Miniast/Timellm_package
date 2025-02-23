model_name=TimeLLM
train_epochs=100
learning_rate=0.001
llama_layers=32

master_port=12345
num_process=2
batch_size=3
d_model=32
d_ff=128

accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_capacity_predict.py \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs