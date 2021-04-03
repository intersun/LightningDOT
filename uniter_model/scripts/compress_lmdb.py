"""
compress processed LMDB
"""
import argparse
import io
import multiprocessing as mp

import numpy as np
import lmdb
from tqdm import tqdm

import msgpack
import msgpack_numpy
msgpack_numpy.patch()


def compress_dump(item):
    key, dump = item
    img_dump = {k.decode('utf-8'): v for k, v in msgpack.loads(dump).items()}
    with io.BytesIO() as writer:
        np.savez_compressed(writer, **img_dump, allow_pickle=True)
        return key, writer.getvalue()


def main(opts):
    if opts.db[-1] == '/':
        opts.db = opts.db[:-1]
    out_name = f'{opts.db}_compressed'
    env = lmdb.open(opts.db, readonly=True)
    txn = env.begin()
    out_env = lmdb.open(out_name, map_size=1024**4)
    out_txn = out_env.begin(write=True)
    with mp.Pool(opts.nproc) as pool, tqdm(total=txn.stat()['entries']) as pbar:
        for i, (key, value) in enumerate(
                pool.imap_unordered(compress_dump, txn.cursor(),
                                    chunksize=128)):
            out_txn.put(key=key, value=value)
            if i % 1000 == 0:
                out_txn.commit()
                out_txn = out_env.begin(write=True)
            pbar.update(1)
        out_txn.commit()
        out_env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=None, type=str,
                        help="processed LMDB")
    parser.add_argument('--nproc', type=int,
                        help='number of cores used')
    args = parser.parse_args()
    main(args)
