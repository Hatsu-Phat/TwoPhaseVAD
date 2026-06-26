from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, List
import cv2
import numpy as np

VIDEO_EXTS = {'.mp4', '.avi', '.mkv', '.mov'}


def find_video_path(video_root: str | Path, rel_path: str) -> Optional[Path]:
    video_root = Path(video_root)
    rel_path = rel_path.replace('\\', '/')
    direct = video_root / rel_path
    if direct.exists():
        return direct

    alt = video_root / 'videos' / rel_path
    if alt.exists():
        return alt

    matches = list(video_root.rglob(Path(rel_path).name))
    return matches[0] if matches else None


def get_video_info(video_path: str | Path) -> Tuple[float, int, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f'Cannot open video: {video_path}')
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if total <= 0:
        raise RuntimeError(f'Invalid total frame count for {video_path}: {total}')
    return fps, total, w, h


def temporal_segments(total_frames: int, num_segments: int = 32) -> List[Tuple[int, int]]:
    if total_frames <= 0:
        raise ValueError('total_frames must be positive')
    boundaries = np.floor(np.linspace(0, total_frames, num_segments + 1)).astype(int)
    out = []
    for i in range(num_segments):
        s = int(boundaries[i])
        e = int(boundaries[i + 1]) - 1
        if e < s:
            e = s
        s = max(0, min(s, total_frames - 1))
        e = max(0, min(e, total_frames - 1))
        out.append((s, e))
    return out


def read_frame_rgb(cap: cv2.VideoCapture, idx: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def read_clip_rgb(video_path: str | Path, start: int, end: int, clip_len: int, image_size: int | None = None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f'Cannot open video: {video_path}')
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start = max(0, min(int(start), max(total - 1, 0)))
    end = max(start, min(int(end), max(total - 1, 0)))
    if end - start + 1 >= clip_len:
        idxs = np.linspace(start, end, clip_len).round().astype(int)
    else:
        idxs = list(range(start, end + 1))
        while len(idxs) < clip_len:
            idxs.append(idxs[-1])
        idxs = np.array(idxs[:clip_len], dtype=int)

    frames = []
    last = None
    for idx in idxs:
        frame = read_frame_rgb(cap, int(idx))
        if frame is None:
            frame = last.copy() if last is not None else np.zeros((224, 224, 3), dtype=np.uint8)
        last = frame.copy()
        if image_size is not None:
            frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        frames.append(frame)
    cap.release()
    return frames


def write_clip_mp4(frames_rgb, out_path: str | Path, fps: float = 15.0):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not frames_rgb:
        raise ValueError('No frames to write')
    h, w = frames_rgb[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps) if fps > 0 else 15.0, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f'Cannot open VideoWriter: {out_path}')
    for frame_rgb in frames_rgb:
        if frame_rgb.shape[:2] != (h, w):
            frame_rgb = cv2.resize(frame_rgb, (w, h))
        writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    writer.release()
