import os
import argparse
import shutil


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", '--project_path', type=str, default='/mnt/data/experiments_train',
                        help='project path for weights and logs')
    parser.add_argument("-f", "--force_delete", type=bool, default=False, help="Force delete without asking for "
                                                                         "confirmation.")
    opts = parser.parse_args()
    return opts


def _delete_folder_helper(rm_folders, opts):
    try:
        for folder in rm_folders:
            shutil.rmtree(os.path.join(opts.project_path, folder))
        print("Folders deleted.")
    except Exception as e:
        print("Deleting all folders failed. Parts of files still might be lost!")
        print(e)


def cleanup():
    opts = get_args()
    folders = ["weights_pretrain", "weights_train", "weights_pl"]
    rm_folders = []

    for folder in folders:
        weights_path = os.path.join(opts.project_path, folder)
        weights_best_path = os.path.join(opts.project_path, folder + "_best")

        # First check if best weights are there
        if os.path.exists(weights_best_path) and len(os.listdir(weights_best_path)) > 0:
            print("Identified non-empty folder:", weights_best_path)
            if os.path.exists(weights_path):
                print("  WILL DELETE:", weights_path)
                rm_folders.append(folder)

    # Remove train weights
    if opts.force_delete:
        _delete_folder_helper(rm_folders, opts)
    elif len(rm_folders) > 0:
        if input("Are you sure you want to permanently delete the folders identified above? [Y/N]") == "Y":
            _delete_folder_helper(rm_folders, opts)
        else:
            print("Aborted.")
    else:
        print("No folders to delete found. Project already cleaned or check project path")


if __name__ == '__main__':
    cleanup()
