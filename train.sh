#! /bin/bash

modes=('final_code' 'full_code')
folds=(1 2 3 4)
datasets=('FlakeFlagger' 'IdoFT')

for dataset in ${datasets[@]}
do
  for fold in ${folds[@]}
  do
    for mode in ${modes[@]}
    do
      python main.py \
      --fold ${fold} \
      --do_train True \
      --do_test False \
      --mode ${mode} \
      --dataset ${dataset} \
      --n_epochs 1 \
      --learning_rate 1e-5 \
      --batch_size 16 \
      --max_len 400 \
      --model_path 'microsoft/codebert-base' \
      --save_path './experiments' \
      --metric_key 'f1' \
      --data_dir './data/'${dataset}'/fold'${fold}'/'
    done
  done
done

