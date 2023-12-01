import asyncio
import httpx
import requests
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm
from huggingface_hub.utils._headers import build_hf_headers
from transformers.utils.hub import http_user_agent
from contextlib import nullcontext
import io
import os.path as osp
from functools import partial
import aiofiles
import json
import subprocess
import tempfile
from ..utils.system import run_cmd

import nest_asyncio

nest_asyncio.apply()

# https://www.iamhippo.com/2021-08/1546.html
USER_AGENT_LIST = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.164 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36"
]


def is_vailable(url):
    ret = get_response_with_redirects(url)
    if ret.status_code == 200:
        return True


def get_response_with_redirects(url, verbose=False, headers=None):
    with httpx.Client(follow_redirects=False, timeout=None) as client:
        response = client.head(url, headers=headers)
        while response.status_code in (301, 302):
            if verbose:
                print("Redirected to:", response.headers['Location'])
            response = client.head(response.headers['Location'])

        if response.status_code == 200:
            return response
        else:
            with client.stream('GET', url=url, headers=headers) as r:
                return r


# async download
def get_headers(from_hf=False):
    # only for huggingface
    user_agent_header = http_user_agent()
    if from_hf:
        headers = build_hf_headers(user_agent=user_agent_header, token="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")
    else:
        headers = {"User-Agent": user_agent_header}
    return headers


def _get_byte_length(obj):
    return (obj.bit_length() + 7) // 8


class Downloader:

    @staticmethod
    def get_output_path(url):
        out = url.split("/")[-1]
        return out


class AsyncDownloader(Downloader):

    @classmethod
    async def merge_shard_files(cls, shard_paths, chunk_size=1024**3, out=None, pbar=None):

        async with aiofiles.open(out, 'wb') as f:
            for shard_path in shard_paths:
                async with aiofiles.open(shard_path, 'rb') as f_shard:
                    while True:
                        chunk = await f_shard.read(chunk_size)
                        if not chunk:
                            break

                        await f.write(chunk)
                        if pbar:
                            pbar.update(len(chunk))

                # Replace with asyncio-compatible command execution or delegate to a thread
                await cls._async_run_cmd(f"> {shard_path} && rm {shard_path}")

    @classmethod
    def _read_chunk_progress(cls, out):
        progress_file = osp.join(osp.dirname(out), "." + osp.basename(out) + ".progress")
        numbers = []
        if osp.exists(progress_file):
            with open(progress_file, 'rb') as f:
                for _ in range(cls.num_shards):
                    chunk = f.read(cls.record_size)
                    numbers.append(int.from_bytes(chunk, 'big'))
        else:
            with open(progress_file, 'wb') as f:
                for _ in range(cls.num_shards):
                    f.write((0).to_bytes(cls.record_size, 'big'))
            return [0] * cls.num_shards
        return numbers

    @classmethod
    async def _write_chunk_progress(cls, out, written_bytes, shard_id):
        progress_file = osp.join(osp.dirname(out), "." + osp.basename(out) + ".progress")
        async with aiofiles.open(progress_file, 'rb+') as f:
            await f.seek(shard_id * cls.record_size)
            binary_data = written_bytes.to_bytes(_get_byte_length(written_bytes), 'big')
            await f.write(binary_data)

    @staticmethod
    def get_shard_path(out, s_pos, e_pos):
        return osp.join(osp.dirname(out), "." + osp.basename(out) + f".tmp{s_pos}-{e_pos}")

    @staticmethod
    async def write_shard_async(out, s_pos, shard_path, chunk_size=1024 * 100):
        async with aiofiles.open(out, "rb+") as f:
            await f.seek(s_pos)
            async with aiofiles.open(shard_path, "rb") as buffer:
                while chunk := await buffer.read(chunk_size):
                    await f.write(chunk)

    @staticmethod
    async def _async_run_cmd(cmd):
        process = await asyncio.create_subprocess_shell(cmd, stdout=asyncio.subprocess.PIPE, shell=True)

    @classmethod
    async def _async_range_download(
        cls,
        url,
        s_pos,
        e_pos,
        client,
        chunk_size,
        out=None,
        headers=None,
        pbar=None,
        max_retries=5,
    ):
        #! should initiate separate file handler for each coroutine, or speed will be slow
        to_buffer = (out is None)

        headers = headers or {}
        retries = 0
        written_bytes = 0

        if to_buffer:
            # byteio
            buffer = io.BytesIO()
            buffer.seek(0)
        else:
            # file path
            out = cls.get_shard_path(out, s_pos, e_pos)
            buffer = open(out, "rb+") if osp.exists(out) else open(out, "wb+")
            buffer.seek(0, 2)  # seek to end
            skip_bytes = buffer.tell()
            written_bytes += skip_bytes
            if pbar:
                pbar.update(skip_bytes)  # update progress bar

            if written_bytes == e_pos - s_pos + 1:
                return

        while retries < max_retries:
            try:
                headers["Range"] = f"bytes={s_pos + written_bytes}-{e_pos}"
                async with client.stream('GET', url=url, headers=headers) as r:
                    async for chunk in r.aiter_bytes():
                        if chunk:  # prevent keep-alive chunks
                            chunk_size = len(chunk)
                            buffer.write(chunk)
                            written_bytes += chunk_size

                            if pbar:
                                pbar.update(chunk_size)
                break  # Break the retry loop if download is successful
            except httpx.NetworkError:
                retries += 1
                if retries < max_retries:
                    print(
                        f"=> Network error at {written_bytes}/{e_pos-s_pos + 1}, retrying ({retries}/{max_retries})...")
                else:
                    print("=> Max retries reached. Download failed.")
                    if not to_buffer:
                        buffer.close()
                    raise

        if not to_buffer:
            buffer.close()
        else:
            return buffer

    @staticmethod
    def _calc_divisional_range(filesize, chuck=10):
        step = filesize // chuck
        ranges = [[i, i + step - 1] for i in range(0, filesize, step)]
        ranges[-1][-1] = filesize - 1
        return ranges

    @classmethod
    def download(cls, url, **kwargs):
        return asyncio.run(cls._async_sharded_download(url, **kwargs))

    @classmethod
    async def _async_sharded_download(cls,
                                      url,
                                      out=None,
                                      chunk_size=1024 * 100,
                                      num_shards=10,
                                      headers=None,
                                      proxy=None,
                                      low_memory=False,
                                      verbose=True):
        to_buffer = (out is None)
        if isinstance(out, io.BufferedRandom):
            out = out.name

        if out == "auto":
            out = cls.get_output_path(url)

        if proxy == "auto":
            proxy = "127.0.0.1:8091"

        res = get_response_with_redirects(url, headers=headers)

        if res.headers.get("Accept-Ranges", None) != "bytes":
            print("File does not support range download, use direct download")
            return

        # get filesize
        url = res.url
        filesize = int(res.headers["Content-Length"])
        divisional_ranges = cls._calc_divisional_range(filesize, num_shards)

        transport = httpx.AsyncHTTPTransport(retries=5)
        proxy = httpx.Proxy(url=f"http://{proxy}") if proxy else None
        client = httpx.AsyncClient(transport=transport, timeout=None, proxies=proxy)


        pbar = tqdm_asyncio(total=filesize,
                            dynamic_ncols=True,
                            desc=f"Downloading",
                            unit="B",
                            unit_scale=True,
                            smoothing=0.1,
                            miniters=1,
                            ascii=True) if verbose else None
        context = pbar if verbose else nullcontext()

        loop = asyncio.get_event_loop()

        async def download_shard(s_pos, e_pos):
            return await cls._async_range_download(url=url,
                                                   s_pos=s_pos,
                                                   e_pos=e_pos,
                                                   client=client,
                                                   chunk_size=chunk_size,
                                                   out=out,
                                                   headers=headers,
                                                   pbar=pbar)

        with context:
            # https://zhuanlan.zhihu.com/p/575243634
            tasks = asyncio.gather(
                *[download_shard(s_pos, e_pos) for idx, (s_pos, e_pos) in enumerate(divisional_ranges)])
            result = loop.run_until_complete(tasks)

        # loop.close()

        if to_buffer:
            ret_buffer = io.BytesIO()
            for buffer in result:
                ret_buffer.write(buffer.getvalue())
            return ret_buffer
        else:
            shard_paths = [
                cls.get_shard_path(out, s_pos, e_pos) for idx, (s_pos, e_pos) in enumerate(divisional_ranges)
            ]
            with open(out, "wb") as f:
                pass
            from time import time
            st = time()
            if low_memory:
                # merge shard files to out

                pbar = tqdm_asyncio(total=filesize,
                                    dynamic_ncols=True,
                                    desc=f"Merging",
                                    unit="B",
                                    unit_scale=True,
                                    smoothing=0.1,
                                    miniters=1,
                                    ascii=True) if verbose else None
                context = pbar if verbose else nullcontext()
                with context:
                    await cls.merge_shard_files(shard_paths, chunk_size=1024**3, out=out, pbar=pbar)
            else:
                cmd = f"cat {' '.join(shard_paths)} > {out} && rm {' '.join(shard_paths)}"
                # print(cmd)
                await cls._async_run_cmd(cmd)

            if verbose:
                print("=> Merging time:", time() - st)


class CommandDownloader(Downloader):

    @classmethod
    def download_axel(cls,
                      url,
                      out=None,
                      headers=None,
                      proxy=None,
                      num_shards=None,
                      timeout=5,
                      retries=3,
                      verbose=True):
        if out == "auto":
            out = cls.get_output_path(url)

        if proxy == "auto":
            proxy = "127.0.0.1:8091"

        axel_args = ""
        if proxy:
            axel_args += f" --proxy {proxy}"

        if headers:
            for k, v in headers.items():
                axel_args += f" --header \"{k}:{v}\""

        if num_shards:
            axel_args += f" --num-connections {num_shards}"

        axel_args += f" --max-redirect {retries} --timeout {timeout}"

        cmd = f"axel {axel_args} '{url}' -o '{out}'"
        if out is not None:
            run_cmd(cmd, verbose=verbose)
        else:
            with tempfile.NamedTemporaryFile() as f, io.BytesIO() as buffer:
                run_cmd(cmd, verbose=verbose, out=f.name)
                buffer.write(f.read())
            return buffer

    @classmethod
    def download_wget(cls, url, out=None, headers=None, proxy=None, timeout=5, retries=3, verbose=True):
        if out == "auto":
            out = cls.get_output_path(url)

        if proxy == "auto":
            proxy = "127.0.0.1:8091"

        wget_args = ""
        if proxy:
            wget_args += f" --proxy=on --proxy http://{proxy}"

        wget_args += f" --tries {retries} --timeout {timeout} --no-check-certificate --continue"

        if headers:
            for k, v in headers.items():
                wget_args += f" --header \"{k}:{v}\""


        if out is not None:
            cmd = f"wget {wget_args} '{url}' -O '{out}'"
            run_cmd(cmd, verbose=verbose)
        else:
            with tempfile.NamedTemporaryFile() as f, io.BytesIO() as buffer:
                cmd = f"wget {wget_args} '{url}' -O '{f.name}'"
                run_cmd(cmd, verbose=verbose)
                buffer.write(f.read())
            buffer.seek(0)
            return buffer


class SimpleDownloader(Downloader):

    @classmethod
    def download(cls, url, out=None, chunk_size=1024 * 100, headers=None, proxy=None, verbose=True):
        if out == "auto":
            out = cls.get_output_path(url)

        to_buffer = (out is None)

        # resolve redirect
        res = get_response_with_redirects(url, headers=headers)
        url = res.url
        filesize = int(res.headers["Content-Length"])

        pbar = tqdm(total=filesize,
                    dynamic_ncols=True,
                    desc=f"Downloading",
                    unit="B",
                    unit_scale=True,
                    smoothing=0.1,
                    miniters=1,
                    ascii=True) if verbose else None

        context = pbar if verbose else nullcontext()

        if to_buffer:
            buffer = io.BytesIO()
        else:
            # buffer = open(out, "rb+") if osp.exists(out) else open(out, "wb+")
            if osp.exists(out) and osp.getsize(out) == filesize:
                return
            buffer = open(out, "wb")
            # buffer.seek(0, 2)  # seek to end
            # skip_bytes = buffer.tell()

        with context:
            proxy = httpx.Proxy(url=f"http://{proxy}") if proxy else None
            client = httpx.Client(timeout=None, proxies=proxy)
            if chunk_size is None:
                with client.stream('GET', url=url, headers=headers) as r:
                        for chunk in r.iter_bytes(chunk_size=chunk_size):
                            if chunk:
                                buffer.write(chunk)
                                if pbar:
                                    pbar.update(len(chunk))
            else:
                with client.get(url=url, headers=headers) as r:
                    buffer.write(r.content)

        if not to_buffer:
            buffer.close()
        else:
            return buffer
