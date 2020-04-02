from __future__ import print_function

import os
import wget
import zipfile


def get_filenames(path):
    if not os.path.exists(path):
        os.makedirs(path)

    image_paths = []
    for filename in os.listdir(path):
#        view_id = filename.split('_')[1].split('.')[0]
#        # ignore images with markers (0) and upward views (5)
#        if not(view_id == '0' or view_id == '5'):
        image_paths.append(os.path.join(path, filename))
    #print(image_paths)
    # 90% train images and 10% test images
    n_train_samples = int(len(image_paths) * 0.9)
    train_filenames = image_paths[:n_train_samples]
    test_filenames = image_paths[n_train_samples:]

    return train_filenames, test_filenames
