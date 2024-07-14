# CAUTION: PATHS NOT ADAPTED HERE

import os
import torch
import torch.nn as nn
import time
from Functions import *
from Dataset import *
from Network import *
from LrScheduler import *
import Metrics
from Logger import CSVLogger
import argparse

try:
    # from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0,1', help='which gpu to use')
    parser.add_argument('--path', type=str,
                        default='/home/exx/Documents/NucleicTransformer/OpenVaccine/post_deadline_files',
                        help='path of csv file with DNA sequences and labels')
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
    parser.add_argument('--no_kmer_aggregation', dest='kmer_aggregation', action='store_false')
    parser.set_defaults(kmer_aggregation=True)
    parser.add_argument('--nfolds', type=int, default=5, help='number of cross validation folds')
    parser.add_argument('--fold', type=int, default=0, help='which fold to train')
    parser.add_argument('--val_freq', type=int, default=1, help='which fold to train')
    opts = parser.parse_args()
    return opts


opts = get_args()
# gpu selection
os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_id
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# lr=0

# json_path=os.path.join(opts.path,'data/train.json')
# data,labels=get_data(json_path)
# exit()


# logger=CSVLogger(columns,csv_file)

# build model and logger
fold_models = []
folds = np.arange(opts.nfolds)
for fold in folds:
    MODELS = []
    for i in range(5):
        model = RNADegformer(opts.ntoken, opts.nclass, opts.ninp, opts.nhead, opts.nhid,
                             opts.nlayers, opts.kmer_aggregation, kmers=opts.kmers,
                             dropout=opts.dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=opts.weight_decay)
        criterion = nn.CrossEntropyLoss(reduction='none')
        lr_schedule = lr_AIAYN(optimizer, opts.ninp, opts.warmup_steps, opts.lr_scale)
        # Initialization
        opt_level = 'O1'
        # model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
        model = nn.DataParallel(model)

        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print('Total number of parameters: {}'.format(pytorch_total_params))

        model.load_state_dict(torch.load("best_weights/fold{}top{}.ckpt".format(fold, i + 1)))
        # model.load_state_dict(torch.load("checkpoints_fold0/epoch{}.ckpt".format(i)))
        model.eval()
        MODELS.append(model)

    dict = MODELS[0].module.state_dict()
    for key in dict:
        for i in range(1, len(MODELS)):
            dict[key] = dict[key] + MODELS[i].module.state_dict()[key]

        dict[key] = dict[key] / float(len(MODELS))

    MODELS[0].module.load_state_dict(dict)
    avg_model = MODELS[0]
    fold_models.append(avg_model)

# alt_input=

# json_path=os.path.join(opts.path,'/home/exx/Documents/NucleicTransformer/OpenVaccine/post_deadline_files/new_sequences.csv')
test = pd.read_csv(os.path.join(opts.path, 'new_sequences.csv'))

test_dataset = RNADataset233(test, opts.path)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

ids = []
preds = []
with torch.no_grad():
    for batch in tqdm(test_dataloader):
        sequence = batch['src'].to(device)
        bpps = batch['bpp'].float().to(device)
        # print(bpps.shape)
        # print(sequence.shape)
        # exit()
        avg_preds = []
        outputs = []
        # for i in range(sequence.shape[1]):
        temp = []
        for model in fold_models:
            # outputs.append(model(sequence[:,i],bpps[:,i]))
            temp.append(model(sequence, bpps))

        temp = torch.stack(temp, 0).mean(0)
        outputs.append(temp)

        outputs = torch.stack(outputs, 1).squeeze().cpu().numpy()  # .mean(0)
        # exit()
        preds.append(outputs)

with open('predictions_233.p', 'wb+') as f:
    pickle.dump(preds, f)
