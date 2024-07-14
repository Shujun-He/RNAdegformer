# 3. Semi-supervised learning: Train on pseudo labels (= "pl")

import os

import numpy as np
import torch
import torch.nn as nn
import time
import json
from Functions import *
from Dataset import *
from Network import *
from LrScheduler import *
import Metrics
from Logger import CSVLogger
import argparse
from ranger import Ranger

from Functions import mkdir
from evaluate import get_train_scores

try:
    # from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")
from torchvision import transforms, utils
from sklearn.model_selection import train_test_split, KFold


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0', help='which gpu to use')
    parser.add_argument('--data_path', type=str, default='/mnt/data/training_data', help='data path')
    parser.add_argument('--project_path', type=str, default='/mnt/data/experiments_train',
                        help='project path for weights and logs')
    parser.add_argument('--epochs', type=int, default=150, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=24, help='size of each batch during training')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight dacay used in optimizer')
    parser.add_argument('--ntoken', type=int, default=4,
                        help='number of tokens to represent DNA nucleotides (should always be 4)')
    parser.add_argument('--nclass', type=int, default=2, help='number of classes from the linear decoder')
    parser.add_argument('--ninp', type=int, default=512, help='ninp for transformer encoder')
    parser.add_argument('--nhead', type=int, default=8, help='nhead for transformer encoder')
    parser.add_argument('--nhid', type=int, default=2048, help='nhid for transformer encoder')
    parser.add_argument('--nlayers', type=int, default=6, help='nlayers for transformer encoder')
    parser.add_argument('--save_freq', type=int, default=1, help='saving checkpoints per save_freq epochs')
    parser.add_argument('--dropout', type=float, default=.1, help='transformer dropout')
    parser.add_argument('--warmup_steps', type=int, default=3200, help='training schedule warmup steps')
    parser.add_argument('--lr_scale', type=float, default=0.1, help='learning rate scale')
    parser.add_argument('--nmute', type=int, default=18, help='number of mutations during training')
    parser.add_argument('--kmers', type=int, nargs='+', default=[2, 3, 4, 5, 6], help='k-mers to be aggregated')
    # parser.add_argument('--kmer_aggregation', type=bool, default=True, help='k-mers to be aggregated')
    parser.add_argument('--kmer_aggregation', dest='kmer_aggregation', action='store_true')
    parser.add_argument('--add_bpps', dest='add_bpps', type=int, help="use bpps as feature [0=False, 1=True]")
    parser.add_argument('--add_loop_struc', dest='add_loop_struc', type=int,
                        help="use structure + loop as features [0=False, 1=True]")
    parser.add_argument('--no_kmer_aggregation', dest='kmer_aggregation', action='store_false')
    parser.set_defaults(kmer_aggregation=True)
    parser.add_argument('--nfolds', type=int, default=5, help='number of cross validation folds')
    parser.add_argument('--fold', type=int, default=0, help='which fold to train')
    parser.add_argument('--val_freq', type=int, default=1, help='which fold to train')
    parser.add_argument('--stride', type=int, default=1, help='stride used in k-mer convolution')
    parser.add_argument('--viral_loss_weight', type=int, default=1, help='stride used in k-mer convolution')
    parser.add_argument('--workers', type=int, default=1, help='number of workers for dataloader')
    parser.add_argument('--error_beta', type=float, default=5, help='some weighting factor for pl stds')
    parser.add_argument('--error_alpha', type=float, default=0, help='some weighting factor for pl stds')
    parser.add_argument('--noise_filter', type=float, default=0.25, help='number of workers for dataloader')
    opts = parser.parse_args()
    return opts


def train_fold():
    # get arguments
    opts = get_args()

    print("Fold nr:", opts.fold)

    # gpu selection
    print("Devices found for training:", [(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())])
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Device in use for training:", device)

    # instantiate datasets
    json_path = os.path.join(opts.data_path, 'data/train.json')

    json_df = pd.read_json(json_path, lines=True)
    json_df = json_df[json_df.signal_to_noise > opts.noise_filter]  #: filter for good signal-to-noise sequence
    ids = np.asarray(json_df.id.to_list())

    error_weights = get_errors(json_df)
    error_weights = opts.error_alpha + np.exp(-error_weights * opts.error_beta)
    train_indices, val_indices = get_train_val_indices(json_df, opts.fold, SEED=2020, nfolds=opts.nfolds)

    _, labels = get_data(json_df)
    sequences = np.asarray(json_df.sequence)
    train_seqs = list(sequences[train_indices])
    val_seqs = sequences[val_indices]
    train_labels = labels[train_indices]
    val_labels = labels[val_indices]
    train_ids = ids[train_indices]
    val_ids = ids[val_indices]
    train_ew = error_weights[train_indices]
    val_ew = error_weights[val_indices]

    train_labels = np.pad(train_labels, ((0, 0), (0, 23), (0, 0)), constant_values=0)
    train_ew = np.pad(train_ew, ((0, 0), (0, 23), (0, 0)), constant_values=0)

    n_train = len(train_labels)

    # train_ids=train_ids+long_df.id.to_list()

    test_json_path = os.path.join(opts.data_path, 'data/test.json')
    test = pd.read_json(test_json_path, lines=True)

    # aug_test=test
    # dataloader
    pseudo_labels_path = f'{opts.project_path}/pseudo_labels/pseudo_labels.p'
    with open(pseudo_labels_path, 'rb') as file:
        long_preds, long_stds, short_preds, short_stds = pickle.load(file)

    # first dimension: different sequences, second dimension: sequence itself (cut at 91 for test set)
    # So what I want is: (N, 107/130, 5)
    short_preds = short_preds[:, :91]  # Where is 91 coming from? Number of bases relevant for prediction, cf. kaggle
    short_stds = short_stds[:, :91]
    long_preds = long_preds[:, :91]
    long_stds = long_stds[:, :91]
    short_stds[:, 68:] = 0

    ls_indices = test.seq_length == 130
    long_data = test[ls_indices]
    long_ids = np.asarray(long_data.id.to_list())
    long_sequences = np.asarray(long_data.sequence.to_list())
    # long_indices,_=get_train_val_indices_PL(long_sequences,opts.fold,SEED=2020,nfolds=opts.nfolds)
    # long_sequences=long_sequences[long_indices]
    # long_preds=long_preds[long_indices]
    # long_stds=long_stds[long_indices]
    # long_ids=long_ids[long_indices]
    long_stds = opts.error_alpha + np.exp(-5 * opts.error_beta * long_stds)

    ss_indices = test.seq_length == 107
    short_data = test[ss_indices]
    short_ids = np.asarray(short_data.id)
    short_sequences = np.asarray(short_data.sequence)
    # short_indices,_=get_train_val_indices_PL(short_sequences,opts.fold,SEED=2020,nfolds=opts.nfolds)
    # short_sequences=short_sequences[short_indices]
    # short_preds=short_preds[short_indices]
    # short_stds=short_stds[short_indices]
    # short_ids=short_ids[short_indices]
    short_stds = opts.error_alpha + np.exp(-5 * opts.error_beta * short_stds)

    train_seqs = np.concatenate(
        [train_seqs, short_sequences, long_sequences])  # last two arrays are from test set, training on all data
    train_labels = np.concatenate([train_labels, short_preds, long_preds], 0)
    train_ids = np.concatenate([train_ids, short_ids, long_ids])
    train_ew = np.concatenate([train_ew, short_stds, long_stds])
    # print(train_labels.shape)  # (5665, 91, 5)
    # print(train_ids.shape)  # (5665,)
    # print(train_ew.shape)  # (5665, 91, 5)
    # exit()

    # Set up data loaders
    # train_inputs=np.stack([train_inputs],0)
    # val_inputs=np.stack([val_inputs,val_inputs2],0)
    pl_dataset = RNADataset(train_seqs[n_train:], train_labels[n_train:], train_ids[n_train:], train_ew[n_train:],
                            opts.data_path, pad=True, k=opts.kmers[0], add_bpps=opts.add_bpps,
                            add_loop_struc=opts.add_loop_struc)
    finetune_dataset = RNADataset(train_seqs[:n_train], train_labels[:n_train, :68], train_ids[:n_train],
                                  train_ew[:n_train, :68], opts.data_path, k=opts.kmers[0], add_bpps=opts.add_bpps,
                                  add_loop_struc=opts.add_loop_struc)
    val_dataset = RNADataset(val_seqs, val_labels, val_ids, val_ew, opts.data_path, training=False, k=opts.kmers[0],
                             add_bpps=opts.add_bpps, add_loop_struc=opts.add_loop_struc)
    pl_dataloader = DataLoader(pl_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.workers)
    finetune_dataloader = DataLoader(finetune_dataset, batch_size=opts.batch_size // 2, shuffle=True,
                                     num_workers=opts.workers)
    val_dataloader = DataLoader(val_dataset, batch_size=opts.batch_size * 2, shuffle=False, num_workers=opts.workers)

    # print(dataset.data.shape)
    # print(dataset.bpps[0].shape)
    # exit()

    # Checkpointing
    mkdir(opts.project_path, "weights_pl")
    checkpoints_folder = os.path.join(opts.project_path, 'weights_pl/checkpoints_fold{}_pl'.format(opts.fold))

    # Logging
    logs_folder = "logs_pl"

    mkdir(opts.project_path, logs_folder)
    logs_path = os.path.join(opts.project_path, logs_folder)

    # Store CLI arguments as json for documentation
    with open(os.path.join(logs_path, "train_pl_config.json"), 'wt') as f:
        json.dump(vars(opts), f, indent=4)

    csv_file = os.path.join(logs_path, 'log_fold{}.csv'.format(opts.fold))
    columns = ['epoch', 'train_loss', 'val_loss']
    logger = CSVLogger(columns, csv_file)

    csv_file_train = os.path.join(logs_path, 'train_log_fold{}.csv'.format(opts.fold))
    logger_train = CSVLogger(['epoch', 'train_loss'], csv_file_train)

    # build model and logger
    model = RNADegformer(opts.ntoken, opts.nclass, opts.ninp, opts.nhead, opts.nhid,
                         opts.nlayers, opts.kmer_aggregation, kmers=opts.kmers, stride=opts.stride,
                         dropout=opts.dropout).to(device)
    optimizer = Ranger(model.parameters(), weight_decay=opts.weight_decay)
    criterion = weighted_MCRMSE
    # lr_schedule=lr_AIAYN(optimizer,opts.ninp,opts.warmup_steps,opts.lr_scale)

    # Mixed precision initialization
    opt_level = 'O1'
    # model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
    model = nn.DataParallel(model)

    # model.load_state_dict(torch.load(f'{opts.project_path}/best_weights/fold{opts.fold}top1.ckpt'))

    # Load best weights from pre-training
    weights_trained_best = os.path.join(opts.project_path, "weights_train_best", f"fold{opts.fold}top1.ckpt")
    model.load_state_dict(torch.load(weights_trained_best))

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('Total number of parameters: {}'.format(pytorch_total_params))

    # distance_mask=get_distance_mask(107)
    # distance_mask=torch.tensor(distance_mask).float().to(device).reshape(1,107,107)
    # print("Starting training for fold {}/{}".format(opts.fold,opts.nfolds))
    # training loop
    cos_epoch = int(opts.epochs * 0.75) - 1
    # cos_epoch=0
    # cos_epoch=-1
    lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                             (opts.epochs - cos_epoch) * len(finetune_dataloader))

    starting_time = time.time()
    print("Starting pseudo-label based fine tuning:", time.strftime("%H:%M:%S", time.localtime()))

    for epoch in range(opts.epochs):
        model.train(True)
        t = time.time()
        total_loss = 0
        optimizer.zero_grad()
        train_preds = []
        ground_truths = []
        step = 0
        if epoch > cos_epoch:
            dataloader = finetune_dataloader
        else:
            dataloader = pl_dataloader

        for data in dataloader:
            # for step in range(1):
            step += 1
            # lr=lr_schedule.step()
            lr = get_lr(optimizer)
            # print(lr)
            src = data['data'].to(device)
            labels = data['labels']
            bpps = data['bpp'].to(device)

            # print(bpps.shape)
            # exit()
            # bpp_selection=np.random.randint(bpps.shape[1])
            # bpps=bpps[:,bpp_selection]
            # src=src[:,bpp_selection]

            # print(bpps.shape)
            # print(src.shape)
            # exit()

            # print(bpps.shape)
            # exit()
            # src=mutate_rna_input(src,opts.nmute)
            # src=src.long()[:,np.random.randint(2)]
            labels = labels.to(device)  # .float()
            src_mask = data['src_mask'].to(device)
            # exit()

            output = model(src, bpps, src_mask)

            ew = data['ew'].to(device)

            # print(output.shape)
            # print(labels.shape)
            loss = criterion(output[:, :labels.shape[1]], labels, ew).mean()

            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #    scaled_loss.backward()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss
            print("Epoch [{}/{}], Step [{}/{}] Loss: {:.3f} Lr:{:.6f} Time: {:.1f}"
                  .format(epoch + 1, opts.epochs, step, len(dataloader), total_loss / (step + 1), lr,
                          time.time() - t), end='\r', flush=True)  # total_loss/(step+1)
            # break
            if epoch > cos_epoch:
                lr_schedule.step()
        print('')

        train_loss = total_loss / (step + 1)
        to_train_log = [epoch + 1, train_loss]
        logger_train.log(to_train_log)

        # recon_acc=np.sum(recon_preds==true_seqs)/len(recon_preds)
        torch.cuda.empty_cache()
        if (epoch + 1) % opts.val_freq == 0 and epoch > cos_epoch:
            # if (epoch+1)%opts.val_freq==0:
            val_loss = validate(model, device, val_dataloader, batch_size=opts.batch_size)
            to_log = [epoch + 1, train_loss, val_loss, ]
            logger.log(to_log)

        if (epoch + 1) % opts.save_freq == 0:
            save_weights(model, optimizer, epoch, checkpoints_folder)

        # if epoch == cos_epoch:
        #     print('yes')

    print("Pseudo-label based fine tuning finished:", time.strftime("%H:%M:%S", time.localtime()))
    training_minutes = (time.time() - starting_time) / 60
    print(f"Pseudo-label based fine tuning time fold {opts.fold} (min.):", training_minutes)

    with open(os.path.join(opts.project_path, logs_folder, f"training_time_fold{opts.fold}.txt"), "w") as file:
        file.write(f"Training time fold {opts.fold} (min.): \n{training_minutes}\n")

    get_best_weights_from_fold(opts, logs_folder=logs_folder, weights_folder="weights_pl", top=1)

    if opts.fold == opts.nfolds - 1:
        get_train_scores(project_path=os.path.join(opts.project_path, logs_folder), n_folds=opts.nfolds)


if __name__ == '__main__':
    train_fold()

# for i in range(3,10):
# ngrams=np.arange(2,i)
# print(ngrams)
# train_fold(0,ngrams)
# # train_fold(0,[2,3,4])
