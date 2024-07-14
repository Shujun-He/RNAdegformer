import pickle
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
from Functions import *
import matplotlib.pyplot as plt

tokens = 'ACGU().BEHIMSX'


# eterna,'nupack','rnastructure','vienna_2','contrafold_2',
class RNADataset(Dataset):
    def __init__(self, seqs, labels, ids, ew, bpp_path, transform=None, training=True, pad=False, k=5,
                 add_bpps=False, add_loop_struc=False):
        if not add_bpps:
            print("WARNING: Using dummy data for bpps")
        if not add_loop_struc:
            print("WARNING: Using dummy data for loop + structure")

        self.transform = transform
        self.seqs = seqs  # .transpose(1,0,2,3)
        # print(self.data.shape)
        self.data = []
        self.labels = labels.astype('float32')
        self.bpp_path = bpp_path
        self.ids = ids
        self.training = training
        self.bpps = []
        self.add_bpps = add_bpps
        dm = get_distance_mask(len(seqs[-1]), add_dm=self.add_bpps)  # .reshape(1,bpps.shape[-1],bpps.shape[-1])
        self.dms = np.asarray([dm for i in range(12)])
        # print(dm.shape)
        # exit()
        self.lengths = []
        for i, id in tqdm(enumerate(self.ids)):
            bpps = np.load(os.path.join(self.bpp_path, 'data/train_test_bpps', id + '_bpp.npy'))
            seq_length = bpps.shape[1]

            if add_bpps:
                bpps = np.load(os.path.join(self.bpp_path, 'data/train_test_bpps', id + '_bpp.npy'))
            else:
                bpps = np.load(os.path.join(self.bpp_path, 'data/dummy_bpps', f'dummy_{seq_length}_bpp.npy'))

            if pad:
                bpps = np.pad(bpps, ([0, 0], [0, 130 - bpps.shape[1]], [0, 130 - bpps.shape[2]]), constant_values=0)
            # print(bpps.shape)
            # exit()
            # dms=np.asarray([dm for i in range(bpps.shape[0])])
            # bpps=np.concatenate([bpps.reshape(bpps.shape[0],1,bpps.shape[1],bpps.shape[2]),dms],1)

            if add_loop_struc:
                with open(os.path.join(self.bpp_path, 'data/train_test_bpps', id + '_struc.p'), 'rb') as f:
                    structures = pickle.load(f)
                with open(os.path.join(self.bpp_path, 'data/train_test_bpps', id + '_loop.p'), 'rb') as f:
                    loops = pickle.load(f)
            else:
                with open(os.path.join(self.bpp_path, 'data/dummy_bpps', f'dummy_{seq_length}_struc.p'), 'rb') as f:
                    structures = pickle.load(f)
                with open(os.path.join(self.bpp_path, 'data/dummy_bpps', f'dummy_{seq_length}_loop.p'), 'rb') as f:
                    loops = pickle.load(f)

            seq = self.seqs[i]
            self.lengths.append(len(seq))
            # print(seq)
            # exit()
            input = []

            for j in range(bpps.shape[0]):
                input_seq = np.asarray([tokens.index(s) for s in seq])
                # input_seq=np.pad(input_seq,[0,130-bpps.shape[1]])
                input_structure = np.asarray([tokens.index(s) for s in structures[j]])
                input_loop = np.asarray([tokens.index(s) for s in loops[j]])
                input.append(np.stack([input_seq, input_structure, input_loop], -1))
            input = np.asarray(input).astype('int')
            # print(input.shape)
            if pad:
                input = np.pad(input, ([0, 0], [0, 130 - input.shape[1]], [0, 0]), constant_values=14)
            # print(input.shape)
            # exit()
            # print(input.shape)
            self.data.append(input)
            # exit()
            # print(np.stack([input_seq,input_structure,input_loop],-1).shape)
            # exit()
            # plt.subplot(1,4,1)
            # for _ in range(4):
            #     plt.subplot(1,4,_+1)
            #     plt.imshow(bpps[0,_])
            # plt.show()
            # exit()
            self.bpps.append(np.clip(bpps, 0, 1).astype('float32'))
            # if i >200:
            #     break
        self.data = np.asarray(self.data)
        self.lengths = np.asarray(self.lengths)
        # print(self.lengths)
        # exit()
        self.ew = ew
        self.k = k
        # if pad:
        self.src_masks = [self.generate_src_mask(self.lengths[i], self.data.shape[-2], self.k) for i in
                          range(len(self.data))]
        self.pad = pad

    def generate_src_mask(self, L1, L2, k):
        mask = np.ones((k, L2), dtype='int8')
        for i in range(k):
            mask[i, L1 + i + 1 - k:] = 0
        return mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # sample = {'data': self.data[idx], 'labels': self.labels[idx]}
        if self.training:
            bpp_selection = np.random.randint(self.bpps[idx].shape[0])
            bpps = self.bpps[idx][bpp_selection]
            bpps = np.concatenate([bpps.reshape(1, bpps.shape[0], bpps.shape[1]), self.dms[0]], 0)
            bpps = bpps.astype('float32')
            # print(self.bpps[idx].shape[0])
            # src_mask=
            # print(src_mask.shape)
            # exit()
            # if self.pad:
            sample = {'data': self.data[idx][bpp_selection], 'labels': self.labels[idx], 'bpp': bpps,
                      'ew': self.ew[idx], 'id': self.ids[idx], 'src_mask': self.src_masks[idx]}
            # else:
            #     sample = {'data': self.data[idx][bpp_selection], 'labels': self.labels[idx], 'bpp': self.bpps[idx][bpp_selection],
            #     'ew': self.ew[idx],'id':self.ids[idx]}
        else:
            bpps = self.bpps[idx]
            bpps = np.concatenate([bpps.reshape(bpps.shape[0], 1, bpps.shape[1], bpps.shape[2]), self.dms], 1)
            bpps = bpps.astype('float32')
            sample = {'data': self.data[idx], 'labels': self.labels[idx], 'bpp': bpps,
                      'ew': self.ew[idx], 'id': self.ids[idx]}
        # if self.transform:
        #     sample=self.transform(sample)
        return sample


class RNADataset233(Dataset):
    def __init__(self, df, path):
        self.df = df
        self.path = path
        self.tokens = 'ACGU().BEHIMSX'

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sequence = np.array([self.tokens.index(s) for s in self.df['sequence'][idx]])
        structure = np.array([self.tokens.index(s) for s in self.df['structure'][idx]])
        loop = np.array([self.tokens.index(s) for s in self.df['structure'][idx]])

        src = np.stack([sequence, structure, loop], -1)

        bpp = np.load(os.path.join(self.path, 'new_sequences_bpps', f'{idx}.npy'))
        dm = get_distance_mask(len(sequence), add_dm=self.add_bpps)

        bpp = np.concatenate([bpp.reshape(1, *bpp.shape), dm], 0)

        return {'src': src, 'bpp': bpp}
