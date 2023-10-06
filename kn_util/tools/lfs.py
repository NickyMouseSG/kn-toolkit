import sys
import os

sys.path.insert(0, os.getcwd())
import subprocess
import argparse
import os.path as osp
from ..utils.download import Downloader, get_headers
from ..utils.git_utils import get_origin_url
from ..basic import map_async
from functools import partial
import re

HF_DOWNLOAD_TEMPLATE = "https://huggingface.co/{org}/{repo}/resolve/main/{path}"


def parse_args():
    parser = argparse.ArgumentParser(description='tools for lfs')
    parser.add_argument("command", type=str, help="The command to run")
    return parser


def run_cmd(cmd, return_output=False):
    # print('Running: {}'.format(cmd))
    if return_output:
        return subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True).stdout
    else:
        subprocess.run(cmd, shell=True, check=True)


def lfs_list_files(include=None):
    cmd = "git lfs ls-files"
    if include:
        cmd += " --include=\"{}\"".format(args.include)
    paths = run_cmd(cmd, return_output=True).splitlines()
    paths = [_.split(" ")[-1].strip() for _ in paths]
    return paths


def pull(args):
    paths = lfs_list_files(include=args.include)

    print(f"=> Found {len(paths)} files to fetch")

    for idx in range(0, len(paths), args.chunk):
        print(f"=> Fetching chunk {idx//args.chunk} of {len(paths)//args.chunk}")
        cmd = "git lfs fetch --include=\"{}\"".format(",".join(paths[idx:idx + 100]))
        run_cmd(cmd)

    run_cmd("git lfs checkout")


def track(args):
    # args = parse_args()
    cmd = "find ./ -name '*' -type f -not -path './.git*'"

    paths = run_cmd(cmd, return_output=True)
    print(paths)
    cont = input("Continue? (y/n)")
    if cont != "y":
        exit(0)

    paths = paths.splitlines()

    for path in paths:
        cmd = "git lfs track \"{}\"".format(path)
        run_cmd(cmd)


def _parse_repo_url(url):
    """parse org, repo from url
    url like https://huggingface.co/TheBloke/stable-vicuna-13B-GGUF
    """
    items = [_ for _ in url.split("/") if _ != ""]
    org, repo = items[-2:]
    if items[-3] == "datasets":
        org = "datasets/" + org

    return org, repo


def download(args):
    # clone the repo
    paths = lfs_list_files(include=args.include)
    print(f"=> Found {len(paths)} files to download")

    url = get_origin_url()
    org, repo = _parse_repo_url(url)
    headers = get_headers(from_hf=True)

    url_path_pairs = []

    for path in paths:
        url = args.template.format(org=org, repo=repo, path=path)
        url_path_pairs += [(url, path)]

    def _download_fn(url_path_pair):
        url, path = url_path_pair
        finish_flag = osp.dirname(path) + "/." + osp.basename(path) + ".finish"

        if osp.exists(finish_flag):
            print(f"=> {path} already downloaded")
            return
        if osp.exists(path):
            os.remove(path)

        Downloader.async_sharded_download(url=url, headers=headers, verbose=True, out=path)
        subprocess.run(f"touch {finish_flag}", shell=True)

    for pair in url_path_pairs:
        print(pair)
        _download_fn(pair)


if __name__ == "__main__":
    parser = parse_args()
    command = parser.parse_known_args()[0].command

    if command == "pull":
        parser.add_argument("--chunk", type=int, help="The chunk number to fetch", default=100)
        parser.add_argument("--include", type=str, help="The partial path to fetch, split by ,", default=None)
        args = parser.parse_args()
        pull(args)
    elif command == "track":
        args = parser.parse_args()
        track(args)
    elif command == "download":
        parser.add_argument("--include", type=str, help="The partial path to fetch, split by ,", default=None)
        parser.add_argument("--template", type=str, help="The chunk number to fetch", default=HF_DOWNLOAD_TEMPLATE)
        args = parser.parse_args()
        download(args)
