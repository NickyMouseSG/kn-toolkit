"""
Modified from https://github.com/m-bain/frozen-in-time/blob/22a91d78405ec6032fdf521ae1ff5573358e632f/base/base_dataset.py
"""

import random
import io
try:
    import torch
except:
    pass

# import av
import cv2
import decord
import imageio
from decord import VideoReader
import numpy as np
import math
from loguru import logger
from einops import rearrange


def pts_to_secs(pts: int, time_base: float, start_pts: int) -> float:
    """
    Converts a present time with the given time base and start_pts offset to seconds.

    Returns:
        time_in_seconds (float): The corresponding time in seconds.

    https://github.com/facebookresearch/pytorchvideo/blob/main/pytorchvideo/data/utils.py#L54-L64
    """
    if pts == math.inf:
        return math.inf

    return int(pts - start_pts) * time_base


def get_pyav_video_duration(video_reader):
    video_stream = video_reader.streams.video[0]
    video_duration = pts_to_secs(video_stream.duration, video_stream.time_base, video_stream.start_time)
    return float(video_duration)


def fill_temporal_param(duration, num_frames=None, fps=None):
    if num_frames is None:
        assert fps is not None
        assert duration is not None
        num_frames = int(duration * fps)
    elif fps is None:
        assert num_frames is not None
        assert duration is not None
        fps = num_frames / duration
    # duration should always be given
    # elif duration is None:
    #     assert fps is not None
    #     assert num_frames is not None
    #     duration = num_frames / fps

    return num_frames, fps, duration


def get_frame_indices(
    num_frames,
    vlen,
    sample="rand",
    offset_from_start=None,
    max_num_frames=-1,
):

    assert sample in ["rand", "middle", "start"]
    acc_samples = min(num_frames, vlen)
    # split the video into `acc_samples` intervals, and sample from each interval.
    intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
    ranges = []
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1] - 1))

    if sample == "rand":
        try:
            frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
        except:
            frame_indices = np.random.permutation(vlen)[:acc_samples]
            frame_indices.sort()
            frame_indices = list(frame_indices)
    elif sample == "start":
        offset_from_start = offset_from_start if offset_from_start is not None else 0
        frame_indices = [np.minimum(x[0] + offset_from_start, x[1]) for x in ranges]
    elif sample == "middle":
        frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
    else:
        raise NotImplementedError

    return frame_indices


# def read_frames_av(video_path, num_frames, sample="rand", fix_start=None, max_num_frames=-1):
#     reader = av.open(video_path)
#     frames = [torch.from_numpy(f.to_rgb().to_ndarray()) for f in reader.decode(video=0)]
#     vlen = len(frames)
#     duration = get_pyav_video_duration(reader)
#     fps = vlen / float(duration)
#     frame_indices = get_frame_indices(num_frames, vlen, sample=sample, fix_start=fix_start, input_fps=fps, max_num_frames=max_num_frames)
#     frames = torch.stack([frames[idx] for idx in frame_indices])  # (T, H, W, C), torch.uint8
#     frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8
#     return frames, frame_indices, duration


def read_frames_gif(
    video_path,
    num_frames,
    sample="rand",
    fix_start=None,
    max_num_frames=-1,
):
    gif = imageio.get_reader(video_path)
    vlen = len(gif)
    frame_indices = get_frame_indices(num_frames, vlen, sample=sample, fix_start=fix_start, max_num_frames=max_num_frames)
    frames = []
    for index, frame in enumerate(gif):
        # for index in frame_idxs:
        if index in frame_indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            frame = torch.from_numpy(frame).byte()
            # # (H x W x C) to (C x H x W)
            frame = frame.permute(2, 0, 1)
            frames.append(frame)
    frames = torch.stack(frames)  # .float() / 255
    return frames, frame_indices, None


def read_frames_decord(
    video_path,
    num_frames=None,
    frame_size=(-1, -1),
    fps=None,
    sample="rand",
    offset_from_start=None,
    truncate_secs=None,
    output_format="tchw",
    bridge="native",
):
    decord.bridge.set_bridge(bridge)
    video_reader = VideoReader(
        video_path,
        num_threads=1,
        width=frame_size[0],
        height=frame_size[1],
    )
    vlen = len(video_reader)
    fps_orig = video_reader.get_avg_fps()
    duration = vlen / float(fps_orig)
    if num_frames is None and fps is None:
        num_frames = vlen

    num_frames, fps, duration = fill_temporal_param(duration=duration, num_frames=num_frames, fps=fps)

    # only truncate if duration is longer than truncate
    if truncate_secs is not None and duration > truncate_secs:
        duration = truncate_secs
        vlen = int(truncate_secs * float(fps))

    frame_indices = get_frame_indices(
        num_frames,
        vlen,
        sample=sample,
        offset_from_start=offset_from_start,
    )

    meta = {
        "fps": fps,
        "vlen": vlen,
        "duration": duration,
        "frame_indices": frame_indices,
    }

    frames = video_reader.get_batch(frame_indices).asnumpy()  # (T, H, W, C)
    output_format = " ".join(output_format)
    frames = rearrange(frames, f"t h w c -> {output_format}")

    return frames, meta


VIDEO_READER_FUNCS = {
    # "av": read_frames_av,
    "decord": read_frames_decord,
    "gif": read_frames_gif,
}


def validate_bytes(video_bytes):
    try:
        _ = read_frames_decord(io.BytesIO(video_bytes), num_frames=2, frame_size=(32, 32))
        return True
    except:
        pass

    return False
