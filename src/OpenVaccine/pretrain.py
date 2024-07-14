# 1. Unsupervised learning: Mutated/masked inputs pre-training

import os
import torch
import torch.nn as nn
import time
import json
import numpy as np
from Functions import *
from Dataset import *
from Network import *
from LrScheduler import *
import Metrics
from Logger import CSVLogger
import argparse
from ranger import Ranger
from torchvision import transforms, utils
from sklearn.model_selection import train_test_split, KFold
from tqdm import tqdm


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
    parser.add_argument('--save_freq', type=int, default=10, help='saving checkpoints per save_freq epochs')
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
    train_ids = json_df.id.to_list()

    json_path = os.path.join(opts.data_path, 'data/test.json')
    test = pd.read_json(json_path, lines=True)

    # aug_test=test
    # dataloader

    # ls_indices=test.seq_length==130
    data = test  # [ls_indices]  # pretraining on test data? -> only for pre-training, using afaik public test set
    # (which is a curated selection of train set)

    train_data, val_data = train_test_split(data, test_size=0.1, random_state=2022)

    ids = np.asarray(train_data.id.to_list())
    training_dataset = RNADataset(train_data.sequence.to_list(), np.zeros(len(train_data)), ids,
                                  np.arange(len(train_data)), opts.data_path, pad=True, k=opts.kmers[0],
                                  add_bpps=opts.add_bpps, add_loop_struc=opts.add_loop_struc)
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size,
                                     shuffle=True, num_workers=opts.workers)

    val_ids = np.asarray(val_data.id.to_list())
    # val_dataset = RNADataset(val_data.sequence.to_list(), np.zeros(len(val_data)), val_ids, np.arange(len(val_data)),
    #                          opts.data_path, pad=True, k=opts.kmers[0], add_bpps=opts.add_bpps, add_loop_struc=opts.add_loop_struc)
    val_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size,
                                shuffle=True, num_workers=opts.workers)
    # Checkpointing
    mkdir(opts.project_path, "weights_pretrain")
    checkpoints_folder = os.path.join(opts.project_path, 'weights_pretrain/checkpoints_fold{}'.format(opts.fold))

    # Logs
    mkdir(opts.project_path, "logs_pretrain")
    logs_path = os.path.join(opts.project_path, "logs_pretrain")

    # Store CLI arguments as json for documentation
    with open(os.path.join(logs_path, "pretrain_config.json"), 'wt') as f:
        json.dump(vars(opts), f, indent=4)

    csv_file = os.path.join(logs_path, 'log_fold{}.csv'.format(opts.fold))
    columns = ['epoch', 'train_loss', 'val_loss']
    logger = CSVLogger(columns, csv_file)

    csv_file_train = os.path.join(logs_path, 'train_log_fold{}.csv'.format(opts.fold))
    logger_train = CSVLogger(['epoch', 'train_loss'], csv_file_train)

    # build model and logger
    model = RNADegformer(opts.ntoken, opts.nclass, opts.ninp, opts.nhead, opts.nhid,
                         opts.nlayers, opts.kmer_aggregation, kmers=opts.kmers, stride=opts.stride,
                         dropout=opts.dropout, pretrain=True).to(device)
    optimizer = Ranger(model.parameters(), weight_decay=opts.weight_decay)
    # optimizer=torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    # lr_schedule=lr_AIAYN(optimizer,opts.ninp,opts.warmup_steps,opts.lr_scale)

    # Mixed precision initialization
    opt_level = 'O1'
    # model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
    model = nn.DataParallel(model)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('Total number of parameters: {}'.format(pytorch_total_params))

    # training loop
    cos_epoch = int(opts.epochs * 0.75)
    total_steps = len(training_dataloader)  # +len(short_dataloader)
    lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, (opts.epochs - cos_epoch) * (total_steps))
    best_loss = 100_000

    starting_time = time.time()
    print("Starting pre-training:", time.strftime("%H:%M:%S", time.localtime()))

    for epoch in range(opts.epochs):
        model.train(True)
        t = time.time()
        total_loss = 0
        optimizer.zero_grad()
        train_preds = []
        ground_truths = []
        step = 0
        for data in training_dataloader:
            # for step in range(1):
            step += 1
            lr = get_lr(optimizer)
            src = data['data']
            labels = data['labels']
            bpps = data['bpp'].to(device)
            src_mask = data['src_mask'].to(device)

            if np.random.uniform() > 0.5:
                masked = mutate_rna_input(src)
            else:
                masked = mask_rna_input(src)

            src = src.to(device).long()
            masked = masked.to(device).long()

            output = model(masked, bpps, src_mask)

            mask_selection = src[:, :, 0] != 14

            loss = (criterion(output[0][mask_selection].reshape(-1, 4), src[:, :, 0][mask_selection].reshape(-1)) + \
                    criterion(output[1][mask_selection].reshape(-1, 3), src[:, :, 1][mask_selection].reshape(-1) - 4) + \
                    criterion(output[2][mask_selection].reshape(-1, 7), src[:, :, 2][mask_selection].reshape(-1) - 7))

            # print(loss)
            # exit()

            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #    scaled_loss.backward()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss
            if epoch > cos_epoch:
                lr_schedule.step()
            print("Epoch [{}/{}], Step [{}/{}] Loss: {:.3f} Lr:{:.6f} Time: {:.1f}"
                  .format(epoch + 1, opts.epochs, step, total_steps, total_loss / (step + 1), lr, time.time() - t),
                  end='\r', flush=True)  # total_loss/(step+1)

        print('')

        train_loss = total_loss / (step + 1)
        to_train_log = [epoch + 1, train_loss]
        logger_train.log(to_train_log)

        if (epoch + 1) % opts.save_freq == 0:
            val_loss = []
            for _ in tqdm(range(5)):
                for data in val_dataloader:
                    # for step in range(1):
                    src = data['data']
                    labels = data['labels']
                    bpps = data['bpp'].to(device)
                    src_mask = data['src_mask'].to(device)

                    if np.random.uniform() > 0.5:
                        masked = mutate_rna_input(src)
                    else:
                        masked = mask_rna_input(src)

                    src = src.to(device).long()
                    masked = masked.to(device).long()

                    with torch.no_grad():
                        output = model(masked, bpps, src_mask)

                    mask_selection = src[:, :, 0] != 14

                    loss = (criterion(output[0][mask_selection].reshape(-1, 4),
                                      src[:, :, 0][mask_selection].reshape(-1)) +
                            criterion(output[1][mask_selection].reshape(-1, 3),
                                      src[:, :, 1][mask_selection].reshape(-1) - 4) +
                            criterion(output[2][mask_selection].reshape(-1, 7),
                                      src[:, :, 2][mask_selection].reshape(-1) - 7))
                    val_loss.append(loss.item())

            val_loss = np.mean(val_loss)
            train_loss = total_loss / (step + 1)
            torch.cuda.empty_cache()
            to_log = [epoch + 1, train_loss, val_loss]
            logger.log(to_log)

            if val_loss < best_loss:
                print(f"new best_loss found at epoch {epoch + 1}: {val_loss}")
                best_loss = val_loss
                save_weights(model, optimizer, epoch, checkpoints_folder)

    logs_folder = "logs_pretrain"

    print("Pre-training finished:", time.strftime("%H:%M:%S", time.localtime()))
    training_minutes = (time.time() - starting_time) / 60
    print(f"Training time fold {opts.fold} (min.):", training_minutes)

    with open(os.path.join(opts.project_path, logs_folder, f"training_time_fold{opts.fold}.txt"), "w") as file:
        file.write(f"Pre-training time fold {opts.fold} (min.): \n{training_minutes}\n")

    get_best_weights_from_fold(opts, logs_folder=logs_folder, weights_folder="weights_pretrain")


if __name__ == '__main__':
    train_fold()
