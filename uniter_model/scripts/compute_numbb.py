"""
compute adaptive number of bounding boxes
"""
import argparse
import glob
import json
from os.path import basename
import multiprocessing as mp

import numpy as np
from tqdm import tqdm
from cytoolz import curry


def _compute_nbb(img_dump, conf_th, max_bb, min_bb):
    num_bb = max(min_bb, (img_dump['conf'] > conf_th).sum())
    num_bb = min(max_bb, num_bb)
    return int(num_bb)


@curry
def _compute_item(conf_th, max_bb, min_bb, fname):
    name = basename(fname)
    try:
        nbb = _compute_nbb(np.load(fname, allow_pickle=True),
                           conf_th, max_bb, min_bb)
    except OSError:
        # some corrupted files in conceptual caption
        nbb = None
    return name, nbb


def _compute_all_nbb(img_dir, conf_th, max_bb, min_bb, nproc):
    files = glob.glob(f'{img_dir}/*.npz')
    with mp.Pool(nproc) as pool:
        fname2nbb = dict(
            pool.imap_unordered(_compute_item(conf_th, max_bb, min_bb),
                                tqdm(files), chunksize=2048))

    return fname2nbb


def main(opts):
    n2bb = _compute_all_nbb(opts.img_dir, opts.conf_th,
                            opts.max_bb, opts.min_bb,
                            opts.nproc)
    with open(f'{opts.img_dir}/'
              f'nbb_th{opts.conf_th}_max{opts.max_bb}_min{opts.min_bb}.json',
              'w') as f:
        json.dump(n2bb, f)
    corrupts = [f for f, n in n2bb.items() if n is None]
    if corrupts:
        with open(f'{opts.img_dir}/corrupted.json', 'w') as f:
            json.dump(corrupts, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", default=None, type=str,
                        help="The input images.")
    parser.add_argument('--conf_th', type=float, default=0.2,
                        help='threshold for dynamic bounding boxes '
                             '(-1 for fixed)')
    parser.add_argument('--max_bb', type=int, default=100,
                        help='max number of bounding boxes')
    parser.add_argument('--min_bb', type=int, default=10,
                        help='min number of bounding boxes')
    parser.add_argument('--nproc', type=int,
                        help='number of cores used')
    args = parser.parse_args()
    main(args)
