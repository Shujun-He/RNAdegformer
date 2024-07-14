# Get scores of weights with corresponding logs

import numpy as np
import os
import pandas as pd

current_project_path = '/mnt/data/experiments_train/logs_train'
current_n_folds = 10


def get_train_scores(project_path='/mnt/data/experiments_train/logs_train', n_folds=10, top=1):
    best_scores = []
    for fold in range(n_folds):
        csv_file_path = os.path.join(project_path, f"log_fold{fold}.csv")
        history = pd.read_csv(csv_file_path)
        scores = np.asarray(-history.val_loss)
        top_epochs = scores.argsort()[-top:][::-1]
        print(f"Best score fold {fold}", scores[top_epochs])
        best_scores.append(scores[top_epochs[0]])

    df = pd.DataFrame(best_scores, columns=["cv_score_fold"])
    df.to_csv(os.path.join(project_path, "best_scores.csv"))

    with open(os.path.join(project_path, 'mean_cv_score.txt'), 'w+') as f:
        f.write("Mean cv score: " + str(-np.mean(best_scores)))
    print(f"Mean cv score", -np.mean(best_scores))


if __name__ == "__main__":
    get_train_scores(current_project_path, current_n_folds)
