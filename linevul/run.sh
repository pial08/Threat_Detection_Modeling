python linevul_main.py \
  --model_name=12heads_linevul_model.bin \
  --output_dir=./saved_models \
  --model_type=roberta \
  --tokenizer_name=microsoft/codebert-base \
  --model_name_or_path=microsoft/codebert-base \
  --do_train \
  --do_test \
  --do_sorting_by_line_scores \
  --learning_rate=6e-6 \
  --weight_decay=0.9 \
  --epochs=25 \
  --effort_at_top_k=0.9 \
  --top_k_recall_by_lines=0.01 \
  --top_k_recall_by_pred_prob=0.2 \
  --reasoning_method=all \
  --train_data_file=../data/D2A_Dataset/train.csv \
  --eval_data_file=../data/D2A_Dataset/val.csv \
  --test_data_file=../data/D2A_Dataset/test.csv \
  --block_size 512 \
  --train_batch_size 16 \
  --eval_batch_size 16 


  # best learning rate 5e-6/6e-6