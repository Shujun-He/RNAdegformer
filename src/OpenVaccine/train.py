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
from ranger import Ranger
try:
    #from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")
from torchvision import transforms, utils
#from Mutation import *
from sklearn.model_selection import train_test_split, KFold

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0',  help='which gpu to use')
    parser.add_argument('--path', type=str, default='../', help='path of csv file with DNA sequences and labels')
    parser.add_argument('--epochs', type=int, default=150, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=24, help='size of each batch during training')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight dacay used in optimizer')
    parser.add_argument('--ntoken', type=int, default=4, help='number of tokens to represent DNA nucleotides (should always be 4)')
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
    parser.add_argument('--kmers', type=int, nargs='+', default=[2,3,4,5,6], help='k-mers to be aggregated')
    #parser.add_argument('--kmer_aggregation', type=bool, default=True, help='k-mers to be aggregated')
    parser.add_argument('--kmer_aggregation', dest='kmer_aggregation', action='store_true')
    parser.add_argument('--no_kmer_aggregation', dest='kmer_aggregation', action='store_false')
    parser.set_defaults(kmer_aggregation=True)
    parser.add_argument('--nfolds', type=int, default=5, help='number of cross validation folds')
    parser.add_argument('--fold', type=int, default=0, help='which fold to train')
    parser.add_argument('--val_freq', type=int, default=1, help='which fold to train')
    parser.add_argument('--stride', type=int, default=1, help='stride used in k-mer convolution')
    parser.add_argument('--viral_loss_weight', type=int, default=1, help='stride used in k-mer convolution')
    parser.add_argument('--workers', type=int, default=1, help='number of workers for dataloader')
    parser.add_argument('--error_beta', type=float, default=5, help='number of workers for dataloader')
    parser.add_argument('--error_alpha', type=float, default=0, help='number of workers for dataloader')
    parser.add_argument('--noise_filter', type=float, default=0.25, help='number of workers for dataloader')
    opts = parser.parse_args()
    return opts

def train_fold():
    #get arguments
    opts=get_args()

    #gpu selection
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #instantiate datasets
    json_path=os.path.join(opts.path,'train.json')

    json=pd.read_json(json_path,lines=True)
    json=json[json.signal_to_noise > opts.noise_filter]
    ids=np.asarray(json.id.to_list())


    error_weights=get_errors(json)
    error_weights=opts.error_alpha+np.exp(-error_weights*opts.error_beta)
    train_indices,val_indices=get_train_val_indices(json,opts.fold,SEED=2020,nfolds=opts.nfolds)

    _,labels=get_data(json)
    sequences=np.asarray(json.sequence)
    train_seqs=sequences[train_indices]
    val_seqs=sequences[val_indices]
    train_labels=labels[train_indices]
    val_labels=labels[val_indices]
    train_ids=ids[train_indices]
    val_ids=ids[val_indices]
    train_ew=error_weights[train_indices]
    val_ew=error_weights[val_indices]

    #train_inputs=np.stack([train_inputs],0)
    #val_inputs=np.stack([val_inputs,val_inputs2],0)
    dataset=RNADataset(train_seqs,train_labels,train_ids, train_ew, opts.path)
    val_dataset=RNADataset(val_seqs,val_labels, val_ids, val_ew, opts.path, training=False)
    dataloader = DataLoader(dataset, batch_size=opts.batch_size,
                            shuffle=True, num_workers=opts.workers)
    val_dataloader = DataLoader(val_dataset, batch_size=opts.batch_size*2,
                            shuffle=False, num_workers=opts.workers)

    # print(dataset.data.shape)
    # print(dataset.bpps[0].shape)
    # exit()
    #checkpointing
    checkpoints_folder='checkpoints_fold{}'.format((opts.fold))
    csv_file='log_fold{}.csv'.format((opts.fold))
    columns=['epoch','train_loss',
             'val_loss']
    logger=CSVLogger(columns,csv_file)

    #build model and logger
    model=RNADegformer(opts.ntoken, opts.nclass, opts.ninp, opts.nhead, opts.nhid,
                           opts.nlayers, opts.kmer_aggregation, kmers=opts.kmers,stride=opts.stride,
                           dropout=opts.dropout).to(device)
    optimizer=Ranger(model.parameters(), weight_decay=opts.weight_decay)
    criterion=weighted_MCRMSE
    #lr_schedule=lr_AIAYN(optimizer,opts.ninp,opts.warmup_steps,opts.lr_scale)

    # Mixed precision initialization
    opt_level = 'O1'
    #model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
    model = nn.DataParallel(model)
    #pretrained_df=pd.read_csv('pretrain.csv')
    #print(pretrained_df.epoch[-1])
    model.load_state_dict(torch.load('pretrain_weights/epoch0.ckpt'))

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('Total number of paramters: {}'.format(pytorch_total_params))


    #distance_mask=get_distance_mask(107)
    #distance_mask=torch.tensor(distance_mask).float().to(device).reshape(1,107,107)
    #print("Starting training for fold {}/{}".format(opts.fold,opts.nfolds))
    #training loop
    cos_epoch=int(opts.epochs*0.75)-1
    lr_schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,(opts.epochs-cos_epoch)*len(dataloader))
    for epoch in range(opts.epochs):
        model.train(True)
        t=time.time()
        total_loss=0
        optimizer.zero_grad()
        train_preds=[]
        ground_truths=[]
        step=0
        for data in dataloader:
        #for step in range(1):
            step+=1
            #lr=lr_schedule.step()
            lr=get_lr(optimizer)
            #print(lr)
            src=data['data'].to(device)
            labels=data['labels']
            bpps=data['bpp'].to(device)
            #print(bpps.shape[1])
            # bpp_selection=np.random.randint(bpps.shape[1])
            # bpps=bpps[:,bpp_selection]
            # src=src[:,bpp_selection]

            # print(bpps.shape)
            # print(src.shape)
            # exit()

            # print(bpps.shape)
            # exit()
            #src=mutate_rna_input(src,opts.nmute)
            #src=src.long()[:,np.random.randint(2)]
            labels=labels.to(device)#.float()
            output=model(src,bpps)
            ew=data['ew'].to(device)
            #print(output.shape)
            #print(labels.shape)
            loss=criterion(output[:,:68],labels,ew).mean()

            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #    scaled_loss.backward()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()
            total_loss+=loss
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.3f} Lr:{:.6f} Time: {:.1f}"
                           .format(epoch+1, opts.epochs, step+1, len(dataloader), total_loss/(step+1) , lr,time.time()-t),end='\r',flush=True) #total_loss/(step+1)
            #break
            if epoch > cos_epoch:
                lr_schedule.step()
        print('')
        train_loss=total_loss/(step+1)
        #recon_acc=np.sum(recon_preds==true_seqs)/len(recon_preds)
        torch.cuda.empty_cache()
        if (epoch+1)%opts.val_freq==0 and epoch > cos_epoch:
        #if (epoch+1)%opts.val_freq==0:
            val_loss=validate(model,device,val_dataloader,batch_size=opts.batch_size)
            to_log=[epoch+1,train_loss,val_loss,]
            logger.log(to_log)


        if (epoch+1)%opts.save_freq==0:
            save_weights(model,optimizer,epoch,checkpoints_folder)

        # if epoch == cos_epoch:
        #     print('yes')


    get_best_weights_from_fold(opts.fold)

if __name__ == '__main__':
    train_fold()


# for i in range(3,10):
    # ngrams=np.arange(2,i)
    # print(ngrams)
    # train_fold(0,ngrams)
# # train_fold(0,[2,3,4])
