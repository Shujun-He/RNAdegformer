import os
import pickle
import numpy as np

data_path = "/mnt/data/experiments_train/data"


def create_dummy_data(n=107, n_algorithms=12):
    structures = ['X' * n for _ in range(n_algorithms)]
    loops = ['.' * n for _ in range(n_algorithms)]  # or also make it A so it gets 0 encoded?
    bpps = np.zeros((n_algorithms, n, n))

    dummy_path = os.path.join(data_path, "dummy_bpps")

    with open(os.path.join(dummy_path, f"dummy_{n}_loop.p"), 'wb') as f:  # [sic] the name is wrong, as also files in train-test_bpps are flipped
        pickle.dump(structures, f)

    with open(os.path.join(dummy_path, f"dummy_{n}_struc.p"), 'wb') as f:  # [sic] too!
        pickle.dump(loops, f)

    np.save(os.path.join(dummy_path, f"dummy_{n}_bpp.npy"), bpps)
    print("Dummy data created for seq length:", n, "in", dummy_path)


if __name__ == '__main__':
    create_dummy_data(107)
    create_dummy_data(130)