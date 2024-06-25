import json

# import joblib
import numpy as np
import dill
import pickle
import csv
import os
import subprocess
from typing import Sequence, Mapping
import os.path as osp
import glob
from tqdm import tqdm
import warnings
import h5py
from ..utils.multiproc import map_async_with_thread


def load_json(fn):
    with open(fn, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: dict, fn):
    with open(fn, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4)


def load_jsonl(fn, as_generator=False):
    with open(fn, "r", encoding="utf-8") as f:
        # FIXME: as_generator not working
        # if as_generator:
        #     for line in f:
        #         yield json.loads(line)
        # else:
        return [json.loads(line) for line in f]


def save_jsonl(obj: Sequence[dict], fn):
    with open(fn, "w", encoding="utf-8") as f:
        for line in obj:
            json.dump(line, f)
            f.write("\n")


def load_pickle(fn):
    # return joblib.load(fn)
    with open(fn, "rb") as f:
        obj = dill.load(f)
    return obj


def save_pickle(obj, fn):
    # return joblib.dump(obj, fn, protocol=pickle.HIGHEST_PROTOCOL)
    with open(fn, "wb") as f:
        dill.dump(obj, f, protocol=dill.HIGHEST_PROTOCOL)


def load_csv(fn, delimiter=",", has_header=True):
    fr = open(fn, "r")
    read_csv = csv.reader(fr, delimiter=delimiter)

    ret_list = []

    for idx, x in enumerate(read_csv):
        if has_header and idx == 0:
            header = x
            continue
        if has_header:
            ret_list += [{k: v for k, v in zip(header, x)}]
        else:
            ret_list += [x]

    return ret_list


def save_csv(obj, fn, delimiter=",", has_header=True):
    fn = open(fn, "w")
    writer = csv.writer(fn, delimiter=delimiter, escapechar='\\')
    if has_header:
        assert isinstance(obj[0], dict), "obj should be a list of dict"
        keys = list(obj[0].keys())
        writer.writerow(keys)

    for row in obj:
        if isinstance(row, dict):
            row = ['"' + str(row[k]) + '"' if isinstance(row[k], str) else str(row[k]) for k in keys]
        writer.writerow(row)

    fn.close()


def save_hdf5(obj, fn, **kwargs):
    import h5py

    if isinstance(fn, str):
        with h5py.File(fn, "a") as f:
            save_hdf5_recursive(obj, f, **kwargs)
    elif isinstance(fn, h5py.File):
        save_hdf5_recursive(obj, fn, **kwargs)
    else:
        raise NotImplementedError(f"{type(obj)} to hdf5 not implemented")


def save_hdf5_recursive(kv, cur_handler, **kwargs):
    """convenient saving hierarchical data recursively to hdf5"""
    if isinstance(kv, Sequence):
        kv = {str(idx): v for idx, v in enumerate(kv)}
    for k, v in kv.items():
        if k in cur_handler:
            warnings.warn(f"{k} already exists in {cur_handler}")
        else:
            if isinstance(v, np.ndarray):
                cur_handler.create_dataset(k, data=v, **kwargs)
            elif isinstance(v, Mapping):
                next_handler = cur_handler.create_group(k)
                save_hdf5_recursive(v, next_handler)


def load_hdf5(fn):
    import h5py

    if isinstance(fn, str):
        with h5py.File(fn, "r") as f:
            return load_hdf5_recursive(f)
    elif isinstance(fn, h5py.Group) or isinstance(fn, h5py.Dataset):
        return load_hdf5_recursive(fn)
    else:
        raise NotImplementedError(f"{type(fn)} from hdf5 not implemented")


def load_hdf5_recursive(cur_handler):
    """convenient saving hierarchical data recursively to hdf5"""
    if isinstance(cur_handler, h5py.Group):
        ret_dict = dict()
        for k in cur_handler.keys():
            ret_dict[k] = load_hdf5_recursive(cur_handler[k])
        return ret_dict
    elif isinstance(cur_handler, h5py.Dataset):
        return np.array(cur_handler)
    else:
        raise NotImplementedError(f"{type(cur_handler)} from hdf5 not implemented")


class LargeHDF5Cache:
    """to deal with IO comflict during large scale prediction,
    we mock the behavior of single hdf5 and build hierarchical cache according to hash_key
    this class is only for saving large hdf5 with parallel workers
    """

    def __init__(self, hdf5_path, **kwargs):
        self.hdf5_path = hdf5_path
        self.kwargs = kwargs
        self.tmp_dir = hdf5_path + ".tmp"

        os.makedirs(osp.dirname(self.hdf5_path), exist_ok=True)
        os.makedirs(self.tmp_dir, exist_ok=True)

    def key_exists(self, hash_id):
        finish_flag = osp.join(self.tmp_dir, hash_id + ".hdf5.finish")
        return osp.exists(finish_flag)

    def cache_save(self, save_dict):
        """save_dict should be like {hash_id: ...}"""
        hash_id = list(save_dict.keys())[0]
        tmp_file = osp.join(self.tmp_dir, hash_id + ".hdf5")
        subprocess.run(f"rm -rf {tmp_file}", shell=True)  # in case of corrupted tmp file without .finish flag
        save_hdf5(save_dict, tmp_file, **self.kwargs)
        subprocess.run(f"touch {tmp_file}.finish", shell=True)

    def final_save(self):
        from torch.utils.data import DataLoader

        tmp_files = glob.glob(osp.join(self.tmp_dir, "*.hdf5"))
        result_handle = h5py.File(self.hdf5_path, "a")
        loader = DataLoader(tmp_files, batch_size=1, collate_fn=lambda x: load_hdf5(x[0]), num_workers=8, prefetch_factor=6)
        for ret_dict in tqdm(loader, desc=f"merging to {self.hdf5_path}"):
            save_hdf5(ret_dict, result_handle)
        result_handle.close()
        subprocess.run(f"rm -rf {self.tmp_dir}", shell=True)
