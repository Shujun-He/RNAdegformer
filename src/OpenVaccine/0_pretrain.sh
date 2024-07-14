python pretrain.py --gpu_id 0 --kmer_aggregation --nmute 0 --epochs 200 --nlayers 5 \
--batch_size 96 --kmers 5 --lr_scale 0.1 --data_path /mnt/data/ \
--project_path /mnt/data/experiments_train --workers 2 \
--dropout 0.1 --nclass 5 --ntoken 15 --nhead 32 --ninp 256 --nhid 1024 --warmup_steps 600 \
--fold 0 --weight_decay 0.1
