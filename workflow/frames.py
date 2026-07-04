"""Real Visual Capture frame extraction via ffmpeg."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageFilter

from workflow.utils import run_cmd

_TRIGGER_PRIORITY = {"visual_cue": 3, "scene": 2, "interval": 1}
_CLUSTER_SECONDS = 1.0


@dataclass
class FrameTranscript:
    """Minimal transcript shape for frame extraction (visual-cue boost)."""

    segments: list[dict]
    text: str


def load_transcript_for_frames(output_dir: Path) -> FrameTranscript:
    path = output_dir / "transcript.json"
    if not path.is_file():
        return FrameTranscript(segments=[], text="")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return FrameTranscript(segments=[], text="")
    if not isinstance(data, dict):
        return FrameTranscript(segments=[], text="")
    segments = data.get("segments", [])
    if not isinstance(segments, list):
        segments = []
    valid_segments = [
        segment
        for segment in segments
        if isinstance(segment, dict) and segment.get("start") is not None
    ]
    text = " ".join(str(segment.get("text", "")) for segment in valid_segments)
    return FrameTranscript(segments=valid_segments, text=text)


def frames_bundle_valid(output_dir: Path) -> bool:
    """True when frames.json exists and every listed JPEG is present."""
    meta_path = output_dir / "frames.json"
    if not meta_path.is_file():
        return False
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if not isinstance(data, list):
        return False
    if not data:
        return True
    for entry in data:
        if not isinstance(entry, dict):
            return False
        path = entry.get("path")
        if not path or not Path(str(path)).is_file():
            return False
    return True


def visual_cue_timestamps(segments: list[dict], cues: list[str]) -> list[float]:
    lowered = [cue.lower() for cue in cues]
    times: list[float] = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        start = segment.get("start")
        if start is None:
            continue
        try:
            ts = float(start)
        except (TypeError, ValueError):
            continue
        text = str(segment.get("text", "")).lower()
        if any(cue in text for cue in lowered):
            times.append(ts)
    return times


def interval_timestamps(duration: float, interval_seconds: float) -> list[float]:
    if interval_seconds <= 0 or duration <= 0:
        return []
    times: list[float] = []
    t = 0.0
    while t < duration:
        times.append(t)
        t += interval_seconds
    return times


def gap_fill_interval_timestamps(
    keyframe_times: list[float],
    duration: float,
    *,
    gap_seconds: float,
    interval_seconds: float,
) -> list[float]:
    """Add fallback samples only in long gaps without scene or visual-cue captures."""
    if interval_seconds <= 0 or duration <= 0:
        return []
    keyframes = sorted({ts for ts in keyframe_times if 0 <= ts <= duration})
    if not keyframes:
        return interval_timestamps(duration, interval_seconds)

    times: list[float] = []
    if keyframes[0] > gap_seconds:
        t = 0.0
        while t < keyframes[0]:
            times.append(t)
            t += interval_seconds

    for start, end in zip(keyframes, keyframes[1:], strict=False):
        if end - start > gap_seconds:
            t = start + interval_seconds
            while t < end - interval_seconds / 2:
                times.append(t)
                t += interval_seconds

    if duration - keyframes[-1] > gap_seconds:
        t = keyframes[-1] + interval_seconds
        while t < duration:
            times.append(t)
            t += interval_seconds

    return times


def _subsample_evenly(items: list[tuple[float, str]], limit: int) -> list[tuple[float, str]]:
    if len(items) <= limit:
        return items
    if limit <= 0:
        return []
    if limit == 1:
        return [items[0]]
    last = len(items) - 1
    return [items[round(index * last / (limit - 1))] for index in range(limit)]


def merge_capture_timestamps(
    *,
    duration: float,
    scene_times: list[float],
    interval_times: list[float],
    visual_cue_times: list[float],
    max_frames: int,
    cluster_seconds: float = 2.0,
    min_spacing_seconds: float = 30.0,
) -> list[tuple[float, str]]:
    """Build a capture schedule with spacing so static segments are not oversampled."""
    candidates: list[tuple[float, str]] = []
    for ts in scene_times:
        if 0 <= ts <= duration:
            candidates.append((ts, "scene"))
    for ts in visual_cue_times:
        if 0 <= ts <= duration:
            candidates.append((ts, "visual_cue"))
    for ts in interval_times:
        if 0 <= ts <= duration:
            candidates.append((ts, "interval"))

    candidates.sort(key=lambda item: item[0])
    clustered: list[tuple[float, str]] = []
    for ts, trigger in candidates:
        if clustered and ts - clustered[-1][0] < cluster_seconds:
            prev_ts, prev_trigger = clustered[-1]
            if trigger == "visual_cue" and prev_trigger == "visual_cue":
                clustered.append((ts, trigger))
                continue
            if _TRIGGER_PRIORITY[trigger] > _TRIGGER_PRIORITY[prev_trigger]:
                clustered[-1] = (min(prev_ts, ts), trigger)
            continue
        clustered.append((ts, trigger))

    spaced: list[tuple[float, str]] = []
    for ts, trigger in clustered:
        if trigger in ("scene", "visual_cue"):
            if spaced and ts - spaced[-1][0] < cluster_seconds:
                if trigger == "visual_cue" and spaced[-1][1] == "visual_cue":
                    spaced.append((ts, trigger))
                    continue
                if _TRIGGER_PRIORITY[trigger] > _TRIGGER_PRIORITY[spaced[-1][1]]:
                    spaced[-1] = (min(spaced[-1][0], ts), trigger)
                continue
            spaced.append((ts, trigger))
            continue
        if spaced and ts - spaced[-1][0] < min_spacing_seconds:
            continue
        spaced.append((ts, trigger))

    spaced.sort(key=lambda item: item[0])
    if len(spaced) <= max_frames:
        return spaced

    cues = [item for item in spaced if item[1] == "visual_cue"]
    others = [item for item in spaced if item[1] != "visual_cue"]
    if len(cues) >= max_frames:
        return _subsample_evenly(cues, max_frames)

    budget = max_frames - len(cues)
    return sorted(cues + _subsample_evenly(others, budget))


def average_hash(path: Path, *, hash_size: int = 8) -> int:
    """Perceptual average hash on decoded pixels (works across JPEG recompression)."""
    with Image.open(path) as img:
        gray = img.convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS)
        pixels = list(gray.get_flattened_data())
    return _bits_from_grayscale_pixels(pixels)


def layout_hash(path: Path, *, hash_size: int = 8, blur_radius: float = 4.0) -> int:
    """Blur-tolerant hash for layout comparison (ignores Teams speaker highlights)."""
    with Image.open(path) as img:
        gray = img.convert("L").resize((64, 64), Image.Resampling.LANCZOS)
        if blur_radius > 0:
            gray = gray.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        gray = gray.resize((hash_size, hash_size), Image.Resampling.LANCZOS)
        pixels = list(gray.get_flattened_data())
    return _bits_from_grayscale_pixels(pixels)


def _bits_from_grayscale_pixels(pixels: list[int]) -> int:
    avg = sum(pixels) / len(pixels)
    bits = 0
    for index, value in enumerate(pixels):
        if value >= avg:
            bits |= 1 << index
    return bits


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def dedupe_near_duplicate_frames(
    timestamps: list[float],
    hashes: dict[int, int],
    *,
    threshold: int = 5,
    dedup_window_seconds: float = 20.0,
    global_dedup: bool = False,
    triggers: list[str] | None = None,
    protected_triggers: frozenset[str] = frozenset({"visual_cue"}),
) -> list[int]:
    """Drop near-duplicate layouts; global mode ignores speaker-highlight-only changes."""
    kept_indices: list[int] = []
    for index, ts in enumerate(timestamps):
        if triggers is not None and triggers[index] in protected_triggers:
            kept_indices.append(index)
            continue
        value = hashes[index]
        is_duplicate = False
        for prior in kept_indices:
            delta = ts - timestamps[prior]
            if not global_dedup and delta > dedup_window_seconds:
                continue
            if hamming_distance(value, hashes[prior]) <= threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            kept_indices.append(index)
    return kept_indices


class FfmpegFrameExtractor:
    def __init__(
        self,
        config: dict,
        *,
        run_cmd_fn: Callable[..., object] = run_cmd,
        probe_duration_fn: Callable[[Path], float] | None = None,
        scene_times_fn: Callable[[Path, float], list[float]] | None = None,
        layout_hash_fn: Callable[[Path], int] | None = None,
    ) -> None:
        frames_cfg = config.get("frames", {})
        self._scene_threshold = float(frames_cfg.get("scene_threshold", 0.35))
        self._interval_seconds = float(frames_cfg.get("interval_seconds", 25))
        self._gap_fill_seconds = float(frames_cfg.get("gap_fill_seconds", 120.0))
        self._max_frames = int(frames_cfg.get("max_frames", 40))
        self._jpeg_quality = str(frames_cfg.get("jpeg_quality", 3))
        self._max_width = int(frames_cfg.get("max_width", 1280))
        self._cluster_seconds = float(frames_cfg.get("cluster_seconds", 2.0))
        self._min_spacing_seconds = float(frames_cfg.get("min_spacing_seconds", 30.0))
        self._dedup_threshold = int(frames_cfg.get("dedup_hash_threshold", 5))
        self._dedup_window_seconds = float(frames_cfg.get("dedup_window_seconds", 20.0))
        self._dedup_global = bool(frames_cfg.get("dedup_global", True))
        self._layout_blur = float(frames_cfg.get("dedup_layout_blur", 4.0))
        self._visual_cues = [str(cue) for cue in config.get("visual_cues", [])]
        self._run_cmd = run_cmd_fn
        self._probe_duration = probe_duration_fn or self._probe_duration_ffprobe
        self._scene_times = scene_times_fn or self._detect_scene_times
        blur = self._layout_blur
        self._layout_hash = layout_hash_fn or (lambda path: layout_hash(path, blur_radius=blur))

    def extract(self, recording: Path, transcript: object, output_dir: Path) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        frames_dir = output_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "frames.json").unlink(missing_ok=True)
        for stale in frames_dir.glob("frame_*.jpg"):
            stale.unlink()

        duration = self._probe_duration(recording)
        segments = getattr(transcript, "segments", [])
        scene_times = self._scene_times(recording, self._scene_threshold)
        cue_times = visual_cue_timestamps(segments, self._visual_cues)
        keyframe_times = sorted(set(scene_times + cue_times))
        interval_times = gap_fill_interval_timestamps(
            keyframe_times,
            duration,
            gap_seconds=self._gap_fill_seconds,
            interval_seconds=self._interval_seconds,
        )
        schedule = merge_capture_timestamps(
            duration=duration,
            scene_times=scene_times,
            interval_times=interval_times,
            visual_cue_times=cue_times,
            max_frames=self._max_frames,
            cluster_seconds=self._cluster_seconds,
            min_spacing_seconds=self._min_spacing_seconds,
        )

        metadata: list[dict] = []
        for index, (timestamp, trigger) in enumerate(schedule, start=1):
            frame_path = frames_dir / f"frame_{index:04d}.jpg"
            self._extract_frame(recording, timestamp, frame_path)
            metadata.append(
                {
                    "timestamp": round(timestamp, 3),
                    "path": str(frame_path),
                    "trigger": trigger,
                }
            )

        deduped_meta = self._dedupe_metadata(metadata)
        for entry in metadata:
            if entry not in deduped_meta:
                Path(entry["path"]).unlink(missing_ok=True)

        deduped_meta = self._renumber_frames(frames_dir, deduped_meta)
        self._prune_orphan_frames(frames_dir, deduped_meta)
        self._write_frames_json(output_dir / "frames.json", deduped_meta)
        return [Path(entry["path"]) for entry in deduped_meta]

    def _extract_frame(self, recording: Path, timestamp: float, frame_path: Path) -> None:
        vf = f"scale={self._max_width}:-2"
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            str(recording),
            "-frames:v",
            "1",
            "-vf",
            vf,
            "-q:v",
            self._jpeg_quality,
            str(frame_path),
        ]
        try:
            self._run_cmd(cmd)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Frame extraction failed at {timestamp:.1f}s for {recording.name}. "
                "Ensure ffmpeg is installed and the file is a valid video."
            ) from exc

    def _dedupe_metadata(self, metadata: list[dict]) -> list[dict]:
        if len(metadata) <= 1:
            return metadata
        hashes = {i: self._layout_hash(Path(entry["path"])) for i, entry in enumerate(metadata)}
        timestamps = [float(entry["timestamp"]) for entry in metadata]
        triggers = [str(entry["trigger"]) for entry in metadata]
        kept_indices = dedupe_near_duplicate_frames(
            timestamps,
            hashes,
            threshold=self._dedup_threshold,
            dedup_window_seconds=self._dedup_window_seconds,
            global_dedup=self._dedup_global,
            triggers=triggers,
        )
        return [metadata[i] for i in kept_indices]

    @staticmethod
    def _renumber_frames(frames_dir: Path, metadata: list[dict]) -> list[dict]:
        """Rename kept frames sequentially after dedup gaps (frame_0001, frame_0002, ...)."""
        renumbered: list[dict] = []
        for index, entry in enumerate(metadata, start=1):
            old_path = Path(entry["path"])
            new_path = frames_dir / f"frame_{index:04d}.jpg"
            if old_path != new_path:
                if new_path.exists():
                    new_path.unlink()
                old_path.rename(new_path)
            renumbered.append({**entry, "path": str(new_path)})
        return renumbered

    @staticmethod
    def _prune_orphan_frames(frames_dir: Path, metadata: list[dict]) -> None:
        kept = {Path(entry["path"]).name for entry in metadata}
        for path in frames_dir.glob("frame_*.jpg"):
            if path.name not in kept:
                path.unlink(missing_ok=True)

    def _probe_duration_ffprobe(self, recording: Path) -> float:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(recording),
        ]
        result = self._run_cmd(cmd)
        stdout = getattr(result, "stdout", "") or ""
        try:
            return max(0.0, float(stdout.strip()))
        except ValueError as exc:
            raise RuntimeError(f"Could not read duration for {recording.name}") from exc

    def _detect_scene_times(self, recording: Path, threshold: float) -> list[float]:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-i",
            str(recording),
            "-filter:v",
            f"select='gt(scene,{threshold})',showinfo",
            "-f",
            "null",
            "-",
        ]
        result = self._run_cmd(cmd)
        stderr = getattr(result, "stderr", "") or ""
        times = [float(match) for match in re.findall(r"pts_time:([\d.]+)", stderr)]
        return sorted(set(times))

    @staticmethod
    def _write_frames_json(path: Path, metadata: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)
