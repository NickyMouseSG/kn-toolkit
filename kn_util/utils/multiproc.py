import time
from pathos.multiprocessing import Pool
import pathos.multiprocessing as mp

from tqdm import tqdm
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, wait
from threading import Semaphore
from multiprocessing.pool import ThreadPool as ThreadPoolVanilla
from queue import Queue
from .rich import get_rich_progress_mofn
import copy


def _run_sequential(iterable, func, desc="", verbose=True):
    pbar = tqdm(total=len(iterable), desc=desc, disable=not verbose)
    ret = []
    for it in iterable:
        ret.append(func(it))
        pbar.update(1)
    pbar.close()
    return ret


def map_async_with_coroutine(iterable, func, desc="", wrap_func=True):
    pbar = tqdm(total=len(iterable), desc=desc)

    async def func_async(item):
        loop = asyncio.get_running_loop()
        if wrap_func:
            result = await loop.run_in_executor(None, func, item)  # 在默认执行器中运行非异步函数
        else:
            result = await func(item)
        pbar.update(1)
        return result

    async def _map_async_with_coroutine(iterable):
        tasks = [func_async(item) for item in iterable]
        results = await asyncio.gather(*tasks)
        pbar.close()
        return results

    return asyncio.run(_map_async_with_coroutine(iterable))


def map_async(
    iterable,
    func,
    num_process=30,
    chunksize=1,
    desc: object = "",
    test_flag=False,
    verbose=True,
):
    """while test_flag=True, run sequentially"""
    if test_flag:
        return _run_sequential(iterable, func, desc=desc, verbose=verbose)
    else:
        p = Pool(num_process)
        # ret = []
        # for it in tqdm(iterable, desc=desc):
        #     ret.append(p.apply_async(func, args=(it,)))
        ret = p.map_async(
            func=func,
            iterable=iterable,
            chunksize=chunksize,
        )
        total = ret._number_left

        progress = get_rich_progress_mofn(disable=not verbose)
        progress.start()
        task_id = progress.add_task(description=desc, total=total)

        while ret._number_left > 0:
            if verbose:
                progress.update(task_id, completed=total - ret._number_left)
                progress.refresh()
            time.sleep(0.1)

        progress.update(task_id, completed=total)
        progress.stop()

        return ret.get()


def map_async_with_thread(
    iterable,
    func,
    num_thread=30,
    desc="",
    verbose=True,
    test_flag=False,
):
    if test_flag:
        return _run_sequential(iterable, func, desc=desc)

    with ThreadPoolExecutor(num_thread) as executor:
        progress = get_rich_progress_mofn(disable=not verbose)
        progress.start()
        task_id = progress.add_task(description=desc, total=len(iterable))

        results = []

        not_done = set()

        for i, x in enumerate(iterable):
            future = executor.submit(func, x)
            future.index = i
            not_done.add(future)

        results = {}

        while len(not_done) > 0:
            done, not_done = wait(not_done, return_when="FIRST_COMPLETED")
            for future in done:
                results[future.index] = future.result()
            progress.update(task_id, advance=len(done))
            progress.refresh()

        progress.stop()

        return [results[i] for i in range(len(iterable))]


def map_async_with_shard(
    iterable,
    func=lambda x: x,
    loader=lambda x: x,
    num_process=32,
    num_thread=32,
    max_semaphore=32,
    max_retries=5,
    shard_size=1000,
    verbose=True,
    is_ready=lambda x: True,
):
    """
    producer-consumer model, split iterable to shards.
    For each shard, use thread to load data and process data iteratively
    (Here, func should be compute-bound, and loader should be io-bound)

    Q1: Why only split iterable to shards when using process pool?
    A1: Frequently switching between processes is much more expensive than switching between threads.
    """
    iterable = list(iterable)  # make sure it is shardable

    def _process_shard(shard):
        semaphore = Semaphore(max_semaphore)
        task_queue = Queue()
        failed_queue = Queue()

        for item in shard:
            task_queue.put((0, item))

        def locked_iterable():
            while not task_queue.empty():
                semaphore.acquire()
                yield task_queue.get()

        def wrapped_loader(*args, **kwargs):
            try:
                return True, item, loader(*args, **kwargs)
            except Exception as e:
                return False, item, None

        def deal_with_error(retry_cnt):
            if retry_cnt >= max_retries:
                failed_queue.put(item)
            else:
                task_queue.put((retry_cnt + 1, item))
            semaphore.release()

        with ThreadPoolVanilla(num_thread) as thread_pool:
            for retry_cnt, success, item, output in thread_pool.imap_unordered(
                lambda x: (x[0], *wrapped_loader(x[1])),
                locked_iterable(),
            ):
                if not success:
                    deal_with_error(retry_cnt)
                    continue

                try:
                    ret = func(output)
                except:
                    deal_with_error(retry_cnt)
                    continue

                if not is_ready(ret):
                    deal_with_error(retry_cnt)
                    continue

                semaphore.release()

        return failed_queue.qsize()

    iterable_shards = [iterable[i : i + shard_size] for shard_idx, i in enumerate(range(0, len(iterable), shard_size))]
    if verbose:
        print(f"Total {len(iterable_shards)} shards")

    map_async(
        iterable=iterable_shards,
        func=_process_shard,
        num_process=num_process,
        verbose=verbose,
    )


def pathos_wait(not_done, timeout=0.5):
    done = set()
    time.sleep(timeout)
    _not_done = copy.copy(not_done)
    for future in _not_done:
        if future.ready():
            not_done.remove(future)
            done.add(future)
            continue
    return done, not_done
