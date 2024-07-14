# Code to reproduce results for the openvaccine dataset


## Data setup
First download all necessary data ```bash download_data.sh```, which does the following:

Download base dataset: https://www.kaggle.com/c/stanford-covid-vaccine/data

Download 12x augmented dataset: https://www.kaggle.com/shujun717/openvaccine-12x-dataset

Unzip them to the same directory, so that the directory contains

├── train.json                   
├── test.json                    
├── sample_submission.csv                     
├── bpps                    
├── post_deadline_files                   
└── openvaccine_12x_dataset


## Training Pipeline
1. **Load data**
   - As instructed in ``download_data.sh`` into a ``data_path``
2. **Create new directory** 
   - to store all weights and predictions, e.g. ``project_path``
3. **Pretrain model** 
   - with ``pretrain.sh`` for 1 fold, specify paths from above
   - pretrain weights will be stored in ``project_path/weights_pretrain``
   - best checkpoint wights will be stored in ``project_path/weights_pretrain_best``
4. **Train model** 
   - with ``run.sh`` for 10 folds
   - pretrain weights will be stored in ``project_path/weights_train``
   - best checkpoint wights will be stored in ``project_path/weights_train_best``
5. **Create pseudo labels** 
   - with trained model with ``predict_pl.sh`` 
   - pseudo labels (pl) will be stored in ``project_path/pseudo_labels``
6. **Fine-tune model**
   - based on pseudo labels with ``train_pl.sh`` for 10 folds
   - pl train weights will be stored in ``project_path/weights_pl``
   - best checkpoint wights will be stored in ``project_path/weights_pl_best``
7. **Obtain predictions**
   - with pl and ground truth fine-tuned weights with ``predict.sh``
   - predictions will be stored in ``project_path/predictions``
