"""
Reusable retrieval + localization engine over the embedding store.

Used by the web app and by any future notebook/experiment. Loads the compact
clip/frame indices (built by build_index.py) into RAM for instant search, and
memmaps the patch-token store for on-demand localization heatmaps.

Retrieval strategies (mirroring the CAV-MAE Sync paper):
    fast          - cosine on mean-pooled clip vectors (single matmul)
    diagonal_mean - mean cosine over the 16 time-aligned frame pairs (re-rank)
    diagonal_max  - max  cosine over the 16 time-aligned frame pairs (re-rank)
    mean / max    - over all 16x16 frame pairs (re-rank)
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

import cavmae_common as C


class RetrievalEngine:
    def __init__(self, emb_dir: Path = C.EMB_DIR,
                 meta_path: Path = C.DATA_DIR / "segments_metadata.json"):
        self.emb_dir = Path(emb_dir)
        idx = self.emb_dir / "index"
        self.clip_audio = np.load(idx / "clip_audio.npy")      # (N,768)
        self.clip_video = np.load(idx / "clip_video.npy")
        self.frame_audio = np.load(idx / "frame_audio.npy")    # (N,16,768)
        self.frame_video = np.load(idx / "frame_video.npy")
        self.seg_ids = json.loads((self.emb_dir / "seg_ids.json").read_text())
        self.id2idx = {s: i for i, s in enumerate(self.seg_ids)}

        meta = json.loads(Path(meta_path).read_text())
        self.prov = {s["video_id"]: s for s in meta["segments"]}
        self.series_reference = meta.get("series_reference", {})
        self.label_map = meta.get("label_map", {})

        # patch tokens (lazy mmap, only opened if localization is used)
        self._patch_video = None
        self._patch_audio = None
        self._cls_audio = None
        self._cls_video = None

    # -- lazy stores ---------------------------------------------------------
    @property
    def patch_video(self):
        if self._patch_video is None:
            self._patch_video = np.load(self.emb_dir / "patch_video.npy", mmap_mode="r")
        return self._patch_video

    @property
    def patch_audio(self):
        if self._patch_audio is None:
            self._patch_audio = np.load(self.emb_dir / "patch_audio.npy", mmap_mode="r")
        return self._patch_audio

    @property
    def cls_audio(self):
        if self._cls_audio is None:
            self._cls_audio = np.load(self.emb_dir / "cls_audio.npy", mmap_mode="r")
        return self._cls_audio

    @property
    def cls_video(self):
        if self._cls_video is None:
            self._cls_video = np.load(self.emb_dir / "cls_video.npy", mmap_mode="r")
        return self._cls_video

    # -- provenance ----------------------------------------------------------
    def provenance(self, seg_id: str) -> dict:
        return self.prov.get(seg_id, {"video_id": seg_id})

    def is_silent(self, idx: int) -> bool:
        return bool(self.prov.get(self.seg_ids[idx], {}).get("silent_source", False))

    # -- search --------------------------------------------------------------
    def search(self, seg_idx: int, direction: str = "audio2video",
               strategy: str = "fast", top_k: int = 24,
               exclude_self_parent: bool = True,
               exclude_silent_audio: bool = True) -> list[dict]:
        """Rank corpus segments against the query segment.

        direction:
            audio2video - query the segment's AUDIO against all VIDEO
            video2audio - query the segment's VIDEO against all AUDIO
        """
        if direction == "audio2video":
            q_clip, db_clip = self.clip_audio[seg_idx], self.clip_video
            q_frame, db_frame = self.frame_audio[seg_idx], self.frame_video
            query_is_silent = self.is_silent(seg_idx)
        else:
            q_clip, db_clip = self.clip_video[seg_idx], self.clip_audio
            q_frame, db_frame = self.frame_video[seg_idx], self.frame_audio
            query_is_silent = False

        scores = db_clip @ q_clip                      # (N,) fast cosine
        cand = np.argsort(-scores)

        # optional filters
        mask_out = set()
        if exclude_self_parent:
            parent = self.prov.get(self.seg_ids[seg_idx], {}).get("parent_video_id")
            if parent:
                mask_out = {i for i, s in enumerate(self.seg_ids)
                            if self.prov.get(s, {}).get("parent_video_id") == parent}
        else:
            mask_out = {seg_idx}

        # for audio->video, silent DB clips have meaningless audio->video match only
        # matters when target modality is audio; skip silent targets in video2audio
        silent_target = (direction == "video2audio" and exclude_silent_audio)

        # take a generous candidate pool, then re-rank if needed
        pool = [i for i in cand if i not in mask_out]
        if silent_target:
            pool = [i for i in pool if not self.is_silent(i)]

        if strategy != "fast":
            pool_head = pool[: max(top_k * 8, 200)]
            reranked = [(i, self._pair_score(q_frame, db_frame[i], strategy))
                        for i in pool_head]
            reranked.sort(key=lambda t: -t[1])
            pool_scored = reranked + [(i, float(scores[i])) for i in pool[len(pool_head):]]
        else:
            pool_scored = [(i, float(scores[i])) for i in pool]

        results = []
        for i, sc in pool_scored[:top_k]:
            p = self.provenance(self.seg_ids[i])
            results.append({
                "seg_id": self.seg_ids[i],
                "index": int(i),
                "score": float(sc),
                "series_id": p.get("series_id"),
                "series_title": p.get("series_title"),
                "title": p.get("title"),
                "year": p.get("year"),
                "media_type": p.get("media_type"),
                "start_sec": p.get("start_sec"),
                "end_sec": p.get("end_sec"),
                "parent_filename": p.get("parent_filename"),
                "silent_source": p.get("silent_source", False),
            })
        return {
            "query_silent": query_is_silent,
            "direction": direction,
            "strategy": strategy,
            "results": results,
        }

    @staticmethod
    def _pair_score(qf: np.ndarray, df: np.ndarray, strategy: str) -> float:
        # qf, df: (16, 768) already L2-normalised
        if strategy in ("diagonal_mean", "diagonal_max"):
            diag = (qf * df).sum(-1)               # (16,)
            return float(diag.mean() if strategy == "diagonal_mean" else diag.max())
        sim = qf @ df.T                            # (16,16)
        return float(sim.mean() if strategy == "mean" else sim.max())

    # -- localization --------------------------------------------------------
    def localize(self, seg_idx: int, frame_idx: int) -> np.ndarray:
        """Return a 14x14 sound-source heatmap for one frame.

        Cosine similarity between the frame's audio summary (cls_audio) and each
        of the 196 visual patch tokens, reshaped to the 14x14 spatial grid.
        """
        pv = np.asarray(self.patch_video[seg_idx, frame_idx], dtype=np.float32)  # (196,768)
        av = np.asarray(self.cls_audio[seg_idx, frame_idx], dtype=np.float32)    # (768,)
        pv = pv / np.clip(np.linalg.norm(pv, axis=-1, keepdims=True), 1e-8, None)
        av = av / np.clip(np.linalg.norm(av), 1e-8, None)
        sim = pv @ av                                    # (196,)
        return sim.reshape(14, 14)

    def localize_audio(self, seg_idx: int, frame_idx: int) -> np.ndarray:
        """Return an 8x26 (freq x time) grounding heatmap for one frame.

        Symmetric twin of `localize`: cosine similarity between the frame's
        visual summary (cls_video) and each of the 208 audio patch tokens,
        reshaped to the 8 frequency x 26 time patch grid of the 4 s window.
        Answers "which parts of the sound are grounded in what's on screen".
        """
        pa = np.asarray(self.patch_audio[seg_idx, frame_idx], dtype=np.float32)  # (208,768)
        vv = np.asarray(self.cls_video[seg_idx, frame_idx], dtype=np.float32)    # (768,)
        pa = pa / np.clip(np.linalg.norm(pa, axis=-1, keepdims=True), 1e-8, None)
        vv = vv / np.clip(np.linalg.norm(vv), 1e-8, None)
        sim = pa @ vv                                    # (208,)
        return sim.reshape(8, 26)                         # (freq, time)


@lru_cache(maxsize=1)
def get_engine() -> RetrievalEngine:
    return RetrievalEngine()
