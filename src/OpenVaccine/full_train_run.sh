#!/bin/bash
set -e  # exit on first error

export N_FOLDS=5
export GPU_ID=1


#######################################
# 1_train_seq
#######################################

# DEFINE VARIABLES
export DATA_PATH=/mnt/data
export PROJECT_PATH=/mnt/data/experiments_train/1_train_seq
export ADD_BPPS=0
export ADD_LOOP_STRUC=0

# 0 - PRETRAIN
python pretrain.py --gpu_id $GPU_ID --kmer_aggregation --nmute 0 --epochs 200 --nlayers 5 \
--batch_size 96 --kmers 5 --lr_scale 0.1 --data_path DATA_PATH \
--project_path $PROJECT_PATH --workers 2 --fold 0 --weight_decay 0.1 \
--dropout 0.1 --nclass 5 --ntoken 15 --nhead 32 --ninp 256 --nhid 1024 --warmup_steps 600 \
--add_bpps $ADD_BPPS --add_loop_struc $ADD_LOOP_STRUC

# 1 - TRAIN
for ((i=0; i<$N_FOLDS; i++)); do
python train.py --gpu_id $GPU_ID --kmer_aggregation --nmute 0 --epochs 75 --nlayers 5 \
--batch_size 32 --kmers 5 --lr_scale 0.1 --data_path DATA_PATH --workers 16 \
--project_path $PROJECT_PATH --dropout 0.1 --nclass 5 --ntoken 15 --nhead 32 --ninp 256 --nhid 1024 \
--warmup_steps 600 --fold $i --weight_decay 0.1 --nfolds $N_FOLDS --error_alpha 0.5 --noise_filter 0.25 \
--add_bpps $ADD_BPPS --add_loop_struc $ADD_LOOP_STRUC
done

# 2 - PREDICT (pseudo label generation)
python predict.py --gpu_id $GPU_ID --kmer_aggregation --nmute 0 --nlayers 5 --batch_size 256 \
--kmers 5 --lr_scale 0.1 --data_path DATA_PATH --project_path $PROJECT_PATH \
--dropout 0.1 --nclass 5 --ntoken 15 --nhead 32 --ninp 256 --nhid 1024 --warmup_steps 600 \
 --fold 0 --weight_decay 1e-4 --nfolds $N_FOLDS --pl_generation \
--add_bpps $ADD_BPPS --add_loop_struc $ADD_LOOP_STRUC

# 3 - TRAIN PL (fine-tuning with pseudo labels)
for ((i=0; i<$N_FOLDS; i++)); do
python train_pl.py --gpu_id $GPU_ID --kmer_aggregation --nmute 0 --epochs 75 --nlayers 5 \
--batch_size 16 --kmers 5 --lr_scale 0.1 --data_path DATA_PATH --workers 16 \
--project_path $PROJECT_PATH --dropout 0.1 --nclass 5 --ntoken 15 --nhead 32 --ninp 256 --nhid 1024 \
--warmup_steps 600 --fold $i --weight_decay 0.1 --nfolds $N_FOLDS --error_alpha 0.5 --noise_filter 0.25 \
--add_bpps $ADD_BPPS --add_loop_struc $ADD_LOOP_STRUC
done



#######################################
# 2_train_bpp
#######################################

# DEFINE VARIABLES
export PROJECT_PATH=/mnt/data/experiments_train/2_train_bpp
export ADD_BPPS=1
export ADD_LOOP_STRUC=0

# 0 - PRETRAIN
python pretrain.py --gpu_id $GPU_ID --kmer_aggregation --nmute 0 --epochs 200 --nlayers 5 \
--batch_size 96 --kmers 5 --lr_scale 0.1 --data_path DATA_PATH \
--project_path $PROJECT_PATH --workers 2 --fold 0 --weight_decay 0.1 \
--dropout 0.1 --nclass 5 --ntoken 15 --nhead 32 --ninp 256 --nhid 1024 --warmup_steps 600 \
--add_bpps $ADD_BPPS --add_loop_struc $ADD_LOOP_STRUC

# 1 - TRAIN
for ((i=0; i<$N_FOLDS; i++)); do
python train.py --gpu_id $GPU_ID --kmer_aggregation --nmute 0 --epochs 75 --nlayers 5 \
--batch_size 32 --kmers 5 --lr_scale 0.1 --data_path DATA_PATH --workers 16 \
--project_path $PROJECT_PATH --dropout 0.1 --nclass 5 --ntoken 15 --nhead 32 --ninp 256 --nhid 1024 \
--warmup_steps 600 --fold $i --weight_decay 0.1 --nfolds $N_FOLDS --error_alpha 0.5 --noise_filter 0.25 \
--add_bpps $ADD_BPPS --add_loop_struc $ADD_LOOP_STRUC
done

# 2 - PREDICT (pseudo label generation)
python predict.py --gpu_id $GPU_ID --kmer_aggregation --nmute 0 --nlayers 5 --batch_size 256 \
--kmers 5 --lr_scale 0.1 --data_path DATA_PATH --project_path $PROJECT_PATH \
--dropout 0.1 --nclass 5 --ntoken 15 --nhead 32 --ninp 256 --nhid 1024 --warmup_steps 600 \
 --fold 0 --weight_decay 1e-4 --nfolds $N_FOLDS --pl_generation \
--add_bpps $ADD_BPPS --add_loop_struc $ADD_LOOP_STRUC

# 3 - TRAIN PL (fine-tuning with pseudo labels)
for ((i=0; i<$N_FOLDS; i++)); do
python train_pl.py --gpu_id $GPU_ID --kmer_aggregation --nmute 0 --epochs 75 --nlayers 5 \
--batch_size 16 --kmers 5 --lr_scale 0.1 --data_path DATA_PATH --workers 16 \
--project_path $PROJECT_PATH --dropout 0.1 --nclass 5 --ntoken 15 --nhead 32 --ninp 256 --nhid 1024 \
--warmup_steps 600 --fold $i --weight_decay 0.1 --nfolds $N_FOLDS --error_alpha 0.5 --noise_filter 0.25 \
--add_bpps $ADD_BPPS --add_loop_struc $ADD_LOOP_STRUC
done



#######################################
# 3_train_struc
#######################################

# DEFINE VARIABLES
export PROJECT_PATH=/mnt/data/experiments_train/3_train_struc
export ADD_BPPS=0
export ADD_LOOP_STRUC=1

# 0 - PRETRAIN
python pretrain.py --gpu_id $GPU_ID --kmer_aggregation --nmute 0 --epochs 200 --nlayers 5 \
--batch_size 96 --kmers 5 --lr_scale 0.1 --data_path DATA_PATH \
--project_path $PROJECT_PATH --workers 2 --fold 0 --weight_decay 0.1 \
--dropout 0.1 --nclass 5 --ntoken 15 --nhead 32 --ninp 256 --nhid 1024 --warmup_steps 600 \
--add_bpps $ADD_BPPS --add_loop_struc $ADD_LOOP_STRUC

# 1 - TRAIN
for ((i=0; i<$N_FOLDS; i++)); do
python train.py --gpu_id $GPU_ID --kmer_aggregation --nmute 0 --epochs 75 --nlayers 5 \
--batch_size 32 --kmers 5 --lr_scale 0.1 --data_path DATA_PATH --workers 16 \
--project_path $PROJECT_PATH --dropout 0.1 --nclass 5 --ntoken 15 --nhead 32 --ninp 256 --nhid 1024 \
--warmup_steps 600 --fold $i --weight_decay 0.1 --nfolds $N_FOLDS --error_alpha 0.5 --noise_filter 0.25 \
--add_bpps $ADD_BPPS --add_loop_struc $ADD_LOOP_STRUC
done

# 2 - PREDICT (pseudo label generation)
python predict.py --gpu_id $GPU_ID --kmer_aggregation --nmute 0 --nlayers 5 --batch_size 256 \
--kmers 5 --lr_scale 0.1 --data_path DATA_PATH --project_path $PROJECT_PATH \
--dropout 0.1 --nclass 5 --ntoken 15 --nhead 32 --ninp 256 --nhid 1024 --warmup_steps 600 \
 --fold 0 --weight_decay 1e-4 --nfolds $N_FOLDS --pl_generation \
--add_bpps $ADD_BPPS --add_loop_struc $ADD_LOOP_STRUC

# 3 - TRAIN PL (fine-tuning with pseudo labels)
for ((i=0; i<$N_FOLDS; i++)); do
python train_pl.py --gpu_id $GPU_ID --kmer_aggregation --nmute 0 --epochs 75 --nlayers 5 \
--batch_size 16 --kmers 5 --lr_scale 0.1 --data_path DATA_PATH --workers 16 \
--project_path $PROJECT_PATH --dropout 0.1 --nclass 5 --ntoken 15 --nhead 32 --ninp 256 --nhid 1024 \
--warmup_steps 600 --fold $i --weight_decay 0.1 --nfolds $N_FOLDS --error_alpha 0.5 --noise_filter 0.25 \
--add_bpps $ADD_BPPS --add_loop_struc $ADD_LOOP_STRUC
done



#######################################
# 4_train_default
#######################################

# DEFINE VARIABLES
export PROJECT_PATH=/mnt/data/experiments_train/4_train_default
export ADD_BPPS=1
export ADD_LOOP_STRUC=1

# 0 - PRETRAIN
python pretrain.py --gpu_id $GPU_ID --kmer_aggregation --nmute 0 --epochs 200 --nlayers 5 \
--batch_size 96 --kmers 5 --lr_scale 0.1 --data_path DATA_PATH \
--project_path $PROJECT_PATH --workers 2 --fold 0 --weight_decay 0.1 \
--dropout 0.1 --nclass 5 --ntoken 15 --nhead 32 --ninp 256 --nhid 1024 --warmup_steps 600 \
--add_bpps $ADD_BPPS --add_loop_struc $ADD_LOOP_STRUC

# 1 - TRAIN
for ((i=0; i<$N_FOLDS; i++)); do
python train.py --gpu_id $GPU_ID --kmer_aggregation --nmute 0 --epochs 75 --nlayers 5 \
--batch_size 32 --kmers 5 --lr_scale 0.1 --data_path DATA_PATH --workers 16 \
--project_path $PROJECT_PATH --dropout 0.1 --nclass 5 --ntoken 15 --nhead 32 --ninp 256 --nhid 1024 \
--warmup_steps 600 --fold $i --weight_decay 0.1 --nfolds $N_FOLDS --error_alpha 0.5 --noise_filter 0.25 \
--add_bpps $ADD_BPPS --add_loop_struc $ADD_LOOP_STRUC
done

# 2 - PREDICT (pseudo label generation)
python predict.py --gpu_id $GPU_ID --kmer_aggregation --nmute 0 --nlayers 5 --batch_size 256 \
--kmers 5 --lr_scale 0.1 --data_path DATA_PATH --project_path $PROJECT_PATH \
--dropout 0.1 --nclass 5 --ntoken 15 --nhead 32 --ninp 256 --nhid 1024 --warmup_steps 600 \
 --fold 0 --weight_decay 1e-4 --nfolds $N_FOLDS --pl_generation \
--add_bpps $ADD_BPPS --add_loop_struc $ADD_LOOP_STRUC

# 3 - TRAIN PL (fine-tuning with pseudo labels)
for ((i=0; i<$N_FOLDS; i++)); do
python train_pl.py --gpu_id $GPU_ID --kmer_aggregation --nmute 0 --epochs 75 --nlayers 5 \
--batch_size 16 --kmers 5 --lr_scale 0.1 --data_path DATA_PATH --workers 16 \
--project_path $PROJECT_PATH --dropout 0.1 --nclass 5 --ntoken 15 --nhead 32 --ninp 256 --nhid 1024 \
--warmup_steps 600 --fold $i --weight_decay 0.1 --nfolds $N_FOLDS --error_alpha 0.5 --noise_filter 0.25 \
--add_bpps $ADD_BPPS --add_loop_struc $ADD_LOOP_STRUC
done