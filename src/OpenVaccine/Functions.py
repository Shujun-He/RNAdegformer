import torch
import os
import shutil
from sklearn import metrics
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import Metrics
import pandas as pd
import pickle
from sklearn.model_selection import KFold, StratifiedKFold


def get_distance_mask(L, add_dm=True):
    m = np.zeros((3, L, L))

    if add_dm:
        # Only compute distance mask if bpps is added as well
        for i in range(L):
            for j in range(L):
                for k in range(3):
                    if abs(i - j) > 0:
                        m[k, i, j] = 1 / abs(i - j) ** (k + 1)
    return m


def aug_data(df, aug_df):
    target_df = df.copy()
    new_df = aug_df[aug_df['id'].isin(target_df['id'])]
    ids = new_df.id.to_list()
    indices = []
    for id in df.id:
        indices.append(ids.index(id))
    indices = np.asarray(indices)
    # del new_df['index']
    # new_df=new_df[indices]
    # print(df[0])
    # new_df=new_df.reindex(indices)
    # print(new_df.head())
    # print(target_df.head())
    # exit()
    del target_df['structure']
    del target_df['predicted_loop_type']
    new_df = new_df.merge(target_df, on=['id', 'sequence'], how='left')

    df['cnt'] = df['id'].map(new_df[['id', 'cnt']].set_index('id').to_dict()['cnt'])
    df['log_gamma'] = 100
    df['score'] = 1.0
    df = new_df[df.columns]
    return df, indices


def get_alt_structures(df):
    """
    columns in the order of 'sequence', 'structure', 'predicted_loop_type'
    """
    # pred_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']

    # train = pd.read_json(path, lines=True)

    folders = ['nupack', 'rnastructure', 'vienna_2',
               'contrafold_2', ]

    token2int = {x: i for i, x in enumerate('ACGU().BEHIMSX')}

    def preprocess_inputs(df, cols):
        return np.transpose(
            np.array(
                df[cols]
                .applymap(lambda seq: [token2int[x] for x in seq])
                .values
                .tolist()
            ),
            (0, 2, 1)
        )

    inputs = []
    for folder in folders:
        columns = ['sequence', folder, folder + '_loop']
        inputs.append(preprocess_inputs(df, columns))
    inputs = np.asarray(inputs)
    # train_inputs = preprocess_inputs(train)
    # train_labels = np.array(train[pred_cols].values.tolist()).transpose((0, 2, 1))
    return inputs


def get_alt_structures_50C(df):
    """
    columns in the order of 'sequence', 'structure', 'predicted_loop_type'
    """
    # pred_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']

    # train = pd.read_json(path, lines=True)

    folders = ['eternafold', 'nupack', 'rnastructure', 'vienna_2',
               'contrafold_2', ]

    token2int = {x: i for i, x in enumerate('ACGU().BEHIMSX')}

    def preprocess_inputs(df, cols):
        return np.transpose(
            np.array(
                df[cols]
                .applymap(lambda seq: [token2int[x] for x in seq])
                .values
                .tolist()
            ),
            (0, 2, 1)
        )

    inputs = []
    for folder in folders:
        columns = ['sequence', folder, folder + '_loop']
        inputs.append(preprocess_inputs(df, columns))
    inputs = np.asarray(inputs)
    # train_inputs = preprocess_inputs(train)
    # train_labels = np.array(train[pred_cols].values.tolist()).transpose((0, 2, 1))
    return inputs


def MCRMSE(y_pred, y_true):
    colwise_mse = torch.mean(torch.square(y_true - y_pred), axis=1)
    MCRMSE = torch.mean(torch.sqrt(colwise_mse), axis=1)
    return MCRMSE


def weighted_MCRMSE(y_pred, y_true, ew):
    colwise_mse = torch.mean(ew * torch.square(y_true - y_pred), axis=1)
    MCRMSE = torch.mean(torch.sqrt(colwise_mse), axis=1)
    return MCRMSE


def get_errors(df, cols=['reactivity_error', 'deg_error_Mg_pH10', 'deg_error_pH10',
                         'deg_error_Mg_50C', 'deg_error_50C']):
    return np.transpose(
        np.array(
            df[cols]
            .values
            .tolist()
        ),
        (0, 2, 1)
    )


def get_data(train):
    pred_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']

    # train = pd.read_json(path, lines=True)

    token2int = {x: i for i, x in enumerate('ACGU().BEHIMSX')}

    def preprocess_inputs(df, cols=['sequence', 'structure', 'predicted_loop_type']):
        return np.transpose(
            np.array(
                df[cols]
                .applymap(lambda seq: [token2int[x] for x in seq])
                .values
                .tolist()
            ),
            (0, 2, 1)
        )

    train_inputs = preprocess_inputs(train)
    train_labels = np.array(train[pred_cols].values.tolist()).transpose((0, 2, 1))
    return train_inputs, train_labels


def get_test_data(path):
    pred_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']

    test = pd.read_json(path, lines=True)

    token2int = {x: i for i, x in enumerate('ACGU().BEHIMSX')}

    def preprocess_inputs(df, cols=['sequence', 'structure', 'predicted_loop_type']):
        return np.transpose(
            np.array(
                df[cols]
                .applymap(lambda seq: [token2int[x] for x in seq])
                .values
                .tolist()
            ),
            (0, 2, 1)
        )

    test_inputs = preprocess_inputs(test)
    return test_inputs


def get_train_val_indices(df, fold, SEED=2020, nfolds=5):
    splits = StratifiedKFold(n_splits=nfolds, random_state=SEED, shuffle=True)
    splits = list(splits.split(df.sequence, df.SN_filter))
    train_indices = splits[fold][0]
    val_indices = splits[fold][1]
    val_indices = val_indices[np.asarray(df.signal_to_noise)[val_indices] > 1]
    return train_indices, val_indices


def get_train_val_indices_PL(data, fold, SEED=2020, nfolds=5):
    splits = KFold(n_splits=nfolds, random_state=SEED, shuffle=True)
    splits = list(splits.split(data))
    train_indices = splits[fold][0]
    val_indices = splits[fold][1]
    # val_indices=val_indices[np.asarray(df.signal_to_noise)[val_indices]>1]
    return train_indices, val_indices


def get_best_weights_from_fold(opts, logs_folder, weights_folder, top=5, rm_weights=False):
    csv_file = os.path.join(opts.project_path, logs_folder, f'log_fold{opts.fold}.csv')

    history = pd.read_csv(csv_file)
    if len(history) < top:
        top = len(history)
    scores = np.asarray(-history.val_loss)
    top_epochs = scores.argsort()[-top:][::-1]
    # print(scores[top_epochs])
    mkdir(opts.project_path, f"{weights_folder}_best")
    success = True

    for i in range(top):
        try:
            if logs_folder.endswith("_pl"):
                weights_path = os.path.join(opts.project_path, weights_folder,
                                            f"checkpoints_fold{opts.fold}_pl/epoch{history.epoch[top_epochs[i]]}.ckpt")
            else:
                weights_path = os.path.join(opts.project_path, weights_folder,
                                            f"checkpoints_fold{opts.fold}/epoch{history.epoch[top_epochs[i]]}.ckpt")
            shutil.copy(weights_path, f"{opts.project_path}/{weights_folder}_best/fold{opts.fold}top{i + 1}.ckpt")
        except Exception as e:
            print("An error occurred:", repr(e))
            success = False

    if rm_weights and success:
        os.system(f'rm -r {opts.project_path}/{weights_folder}')


def mutate_dna_sequence(sequence, nmute=15):
    perm = torch.randperm(sequence.size(1))
    to_mutate = perm[:nmute]
    mutation = torch.randint(4, size=(sequence.size(0), nmute))
    sequence[:, to_mutate] = mutation
    return sequence


def mutate_rna_input(sequence, nmute=.15):
    nmute = int(sequence.shape[1] * nmute)
    perm = torch.randperm(sequence.size(1))
    to_mutate = perm[:nmute]
    sequence_mutation = torch.randint(4, size=(sequence.size(0), nmute))
    structure_mutation = torch.randint(4, 7, size=(sequence.size(0), nmute))
    d_mutation = torch.randint(7, 14, size=(sequence.size(0), nmute))
    mutated = sequence.clone()
    mutated[:, to_mutate, 0] = sequence_mutation
    mutated[:, to_mutate, 1] = structure_mutation
    mutated[:, to_mutate, 2] = d_mutation
    return mutated


def mask_rna_input(sequence, nmute=.15):
    nmute = int(sequence.shape[1] * nmute)
    perm = torch.randperm(sequence.size(1))
    to_mutate = perm[:nmute]
    masked = sequence.clone()
    masked[:, to_mutate, :] = 14
    return masked


def get_MLM_mask(sequence, nmask=12):
    mask = np.zeros(sequence.shape, dtype='bool')
    to_mask = np.random.choice(len(sequence[0]), size=(nmask), replace=False)
    mask[:, to_mask] = True
    return mask


def get_complementary_sequence(sequence):
    complementary_sequence = sequence.copy()
    complementary_sequence[sequence == 0] = 1
    complementary_sequence[sequence == 1] = 0
    complementary_sequence[sequence == 2] = 3
    complementary_sequence[sequence == 3] = 2
    complementary_sequence = complementary_sequence[:, ::-1]
    return complementary_sequence


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
    return lr


def mkdir(path: str, folder: str):
    full_path = os.path.join(path, folder)
    if not os.path.isdir(full_path):
        os.makedirs(full_path, exist_ok=True)


def save_weights(model, optimizer, epoch, folder):
    if not os.path.isdir(folder):
        os.makedirs(folder, exist_ok=True)
    torch.save(model.state_dict(), folder + '/epoch{}.ckpt'.format(epoch + 1))


def validate(model, device, dataset, batch_size=64):
    batches = len(dataset)
    total = 0
    predictions = []
    # outputs=[]
    ground_truths = []
    loss = 0
    val_acc = 0
    criterion = MCRMSE
    label_indices = [0, 1, 3]
    model.eval()
    with torch.no_grad():
        for data in tqdm(dataset):
            X = data['data'].to(device, )
            Y = data['labels'].to(device)
            bpps = data['bpp'].to(device)
            # ew=data['ew'].to(device).float()
            # directions=data['directions']
            # directions=directions.reshape(len(directions),1)*np.ones(X.shape)
            # directions=torch.Tensor(directions).to(device).long()
            outputs = []
            for i in range(X.shape[1]):
                outputs.append(model(X[:, i], bpps[:, i])[:, :68])
                # outputs.append(model(X[:,i],bpps[:,5+i]+distance_mask)[:,:68])
            outputs = torch.stack(outputs, 0).mean(0)
            # print(outputs.shape)
            # exit()

            del X
            loss += criterion(outputs, Y[:, :68]).mean()
            # output=softmax(output)
            # classification_predictions = torch.argmax(output,dim=1).squeeze()
            # val_acc+=Metrics.accuracy(classification_predictions.cpu().numpy(),Y.cpu().numpy())
            #     predictions.append(pred.cpu().numpy())
            # for vector in output:
            #     outputs.append(vector.cpu().numpy())
            # for label in Y.cpu().numpy():
            #     ground_truths.append(label)
            # del output
    # ground_truths=np.asarray(ground_truths,dtype='int')
    torch.cuda.empty_cache()
    val_loss = (loss / batches).cpu()
    print("val los:", val_loss.item())
    return val_loss


def revalidate(model, device, dataset, batch_size=64):
    batches = len(dataset)
    total = 0
    predictions = []
    # outputs=[]
    ground_truths = []
    loss = 0
    val_acc = 0
    criterion = MCRMSE
    label_indices = [0, 1, 3]
    with torch.no_grad():
        for data in tqdm(dataset):
            X = data['data'].to(device, ).long()[:, :]
            Y = data['labels'].to(device).float()
            bpps = data['bpp'].to(device).float()
            # directions=data['directions']
            # directions=directions.reshape(len(directions),1)*np.ones(X.shape)
            # directions=torch.Tensor(directions).to(device).long()
            output = model(X, bpps)[:, :68, ]
            del X
            loss += criterion(output, Y).mean()
            # output=softmax(output)
            classification_predictions = torch.argmax(output, dim=1).squeeze()
            val_acc += Metrics.accuracy(classification_predictions.cpu().numpy(), Y.cpu().numpy())
            #     predictions.append(pred.cpu().numpy())
            # for vector in output:
            #     outputs.append(vector.cpu().numpy())
            # for label in Y.cpu().numpy():
            #     ground_truths.append(label)
            del output
    # ground_truths=np.asarray(ground_truths,dtype='int')
    torch.cuda.empty_cache()
    val_loss = (loss / batches).cpu()
    print(val_loss.item())
    return val_loss


def predict(model, device, dataset, batch_size=64):
    batches = int(len(dataset.val_indices) / batch_size) + 1
    model.train(False)
    total = 0
    ground_truths = dataset.labels[dataset.val_indices]
    predictions = []
    attention_weights = []
    loss = 0
    criterion = nn.CrossEntropyLoss()
    dataset.switch_mode(training=False)
    dataset.update_batchsize(batch_size)
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            data = dataset[i]
            X = data['data'].to(device, ).long()
            Y = data['labels'].to(device, dtype=torch.int64)
            output, _, _, aw = model(X, None)
            del X
            loss += criterion(output, Y)
            classification_predictions = torch.argmax(output, dim=1).squeeze()
            for pred in output:
                predictions.append(pred.cpu().numpy())
            for weight in aw:
                attention_weights.append(weight.cpu().numpy())

            del output
    torch.cuda.empty_cache()
    val_loss = (loss / batches).cpu()
    predictions = np.asarray(predictions)
    attention_weights = np.asarray(attention_weights)
    binary_predictions = predictions.copy()
    binary_predictions[binary_predictions == 2] = 1
    binary_ground_truths = ground_truths.copy()
    binary_ground_truths[binary_ground_truths == 2] = 1
    return predictions, attention_weights, np.asarray(dataset.data[dataset.val_indices])
