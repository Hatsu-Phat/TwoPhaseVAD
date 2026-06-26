from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np

NORMAL_NAMES = {'Normal', 'Normal_Videos_event', 'Training_Normal_Videos_Anomaly', 'Testing_Normal_Videos'}


def is_normal_relpath(rel_path: str) -> bool:
    p = Path(rel_path.replace('\\', '/'))
    first = p.parts[0] if len(p.parts) > 1 else ''
    stem = p.stem
    return first in NORMAL_NAMES or stem.startswith('Normal_Videos') or stem.startswith('NormalVideos')


def class_name_from_relpath(rel_path: str) -> str:
    p = Path(rel_path.replace('\\', '/'))
    if len(p.parts) > 1:
        return p.parts[0]
    if p.stem.startswith('Normal_Videos'):
        return 'Normal_Videos_event'
    return 'Unknown'


def read_split_file(path: str | Path) -> List[str]:
    items = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for raw in f:
            line = raw.strip().replace('\\', '/')
            if not line:
                continue
            if not line.lower().endswith(('.mp4', '.avi', '.mkv', '.mov')):
                continue
            items.append(line)
    return items


def parse_temporal_annotations(annotation_file: str | Path | None) -> Dict[str, Dict[str, Any]]:
    if annotation_file is None:
        return {}
    path = Path(annotation_file)
    if not path.exists():
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for raw in f:
            parts = raw.strip().split()
            if len(parts) < 6:
                continue
            video = parts[0]
            event = parts[1]
            try:
                nums = [int(x) for x in parts[2:6]]
            except ValueError:
                continue
            ranges = []
            for a, b in [(nums[0], nums[1]), (nums[2], nums[3])]:
                if a > 0 and b > 0:
                    if b < a:
                        a, b = b, a
                    ranges.append((a, b))
            out[video] = {'event': event, 'ranges': ranges}
    return out


def segment_labels_from_ranges(total_frames: int, ranges_1based: List[Tuple[int, int]], num_segments: int = 32) -> np.ndarray:
    labels = np.zeros(num_segments, dtype=np.int64)
    if total_frames <= 0 or not ranges_1based:
        return labels
    boundaries = np.floor(np.linspace(0, total_frames, num_segments + 1)).astype(int)
    for i in range(num_segments):
        seg_s = int(boundaries[i]) + 1
        seg_e = int(boundaries[i + 1])
        if seg_e < seg_s:
            seg_e = seg_s
        for a, b in ranges_1based:
            if max(seg_s, a) <= min(seg_e, b):
                labels[i] = 1
                break
    return labels


def frame_labels_from_ranges(total_frames: int, ranges_1based: List[Tuple[int, int]]) -> np.ndarray:
    labels = np.zeros(total_frames, dtype=np.int64)
    for a, b in ranges_1based:
        s = max(0, a - 1)
        e = min(total_frames - 1, b - 1)
        if e >= s:
            labels[s:e + 1] = 1
    return labels
