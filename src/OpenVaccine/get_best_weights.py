# Get best weights from final training step (pl fine tuning)
# Note: this is now enabled per default in train_pl.py

import numpy as np
import os
import pandas as pd

from Functions import mkdir

current_project_path = '/mnt/data/experiments_train'
current_n_folds = 5


def get_best_weights_from_fold(fold, csv_file, weights_path, des, top=1, copy=True):
    # csv_file='log_fold{}.csv'.format(fold)

    history = pd.read_csv(csv_file)
    scores = np.asarray(-history.val_loss)
    top_epochs = scores.argsort()[-top:][::-1]
    print(f"Best score fold {fold}", scores[top_epochs])
    mkdir(weights_path, des)

    if copy:
        for i in range(top):
            epoch = history.epoch[top_epochs[i]]
            weights_path = f"{weights_path}/epoch{epoch}.ckpt"
            os.system('cp {} {}/fold{}top{}.ckpt'.format(weights_path, des, fold, i + 1))

    return scores[top_epochs[0]]


def get_best_weights_train_pl(project_path, current_n_folds, copy=False):
    scores = []
    for i in range(current_n_folds):
        scores.append(get_best_weights_from_fold(fold=i, csv_file=os.path.join(project_path, f"logs_pl/log_fold{i}.csv"),
                                                 weights_path=os.path.join(project_path, f"weights_pl/checkpoints_fold{i}_pl"),
                                                 des=os.path.join(project_path, 'weights_pl_best'), copy=copy))

    print(f"Mean cv score", -np.mean(scores))

    df = pd.DataFrame(scores, columns=["cv_score_fold"])
    df.to_csv(os.path.join(project_path, "best_scores.csv"))

    with open(os.path.join(project_path, 'mean_cv_score.txt'), 'w+') as f:
        f.write("Mean cv score: " + str(-np.mean(scores)))


def get_scores(project_path, current_n_folds):
    get_best_weights_train_pl(project_path, current_n_folds, copy=False)


if __name__ == "__main__":
    # get_best_weights_train_pl(current_project_path, current_n_folds)
    get_scores(current_project_path, current_n_folds)
