# Code to reproduce results for the openvaccine dataset

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

Here I include the hypeparameters that give the best single model.

1. pretrain with all available sequences: ```./pretrain.sh```

2. train on targets: ```./run.sh```

3. ```python get_best_weights.py``` to extract best weights for each fold based on validation performance

4. to make predictions and generate a csv file (submission.csv) for submission on kaggle: ```./predict.sh``` then you can make a submission at https://www.kaggle.com/c/stanford-covid-vaccine/submissions
