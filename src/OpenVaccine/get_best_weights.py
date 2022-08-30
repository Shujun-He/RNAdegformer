import numpy as np
import os
import pandas as pd



def get_best_weights_from_fold(fold,csv_file,weights_path,des,top=1):
    #csv_file='log_fold{}.csv'.format(fold)

    history=pd.read_csv(csv_file)
    scores=np.asarray(-history.val_loss)
    top_epochs=scores.argsort()[-top:][::-1]
    print(scores[top_epochs])
    os.system(f'mkdir {des}')

    for i in range(top):
        #weights_path='checkpoints_fold{}/epoch{}.ckpt'.format(fold,history.epoch[top_epochs[i]])
        epoch=history.epoch[top_epochs[i]]
        weights_path=f"{weights_path}/epoch{epoch}.ckpt"
        #print(weights_path)
        os.system('cp {} {}/fold{}top{}.ckpt'.format(weights_path,des,fold,i+1))

    return scores[top_epochs[0]]

scores=[]
for i in range(10):
    scores.append(get_best_weights_from_fold(i,csv_file=f"logs/log_pl_fold{i}.csv",weights_path=f"weights/checkpoints_fold{i}_pl",des='best_pl_weights'))

with open('cv.txt','w+') as f:
    f.write(str(-np.mean(scores)))
