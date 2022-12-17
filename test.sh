modes=('final_code' 'full_code')
folds=(1 2 3 4 5 6 7 8 9 10)
datasets=('FlakeFlagger' 'IdoFT')
for dataset in ${dasets[@]}
do
  for fold in ${folds[@]}
  do
    for mode in ${modes[@]}
    do
      python main.py \
      --fold ${fold} \
      --do_train False \
      --do_test True \
      --mode ${mode} \
      --dataset ${dataset} \
      --n_epochs 1 \
      --learning_rate 1e-5 \
      --batch_size 32 \
      --max_len 400 \
      --model_path 'microsoft/codebert-base' \
      --save_path './experiments' \
      --metric_key 'f1' \
      --data_dir './data/'+ ${dataset} + '/' + ${fold} + '/'
    done
  done
done
