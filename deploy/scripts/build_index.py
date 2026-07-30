"""
Build compact retrieval indices from the embedding store.

For a fast interactive search engine we reduce each segment's 16 per-frame
cls embeddings to a single L2-normalised clip vector (mean pooling), so a
whole-corpus search is one matrix multiply. The full per-frame embeddings
remain in the store for precise diagonal re-ranking and localization.

Outputs under embeddings/index/:
    clip_audio.npy   (N, 768) fp32  L2-normalised mean-pooled audio
    clip_video.npy   (N, 768) fp32  L2-normalised mean-pooled video
    frame_audio.npy  (N, 16, 768) fp32  L2-normalised per-frame audio
    frame_video.npy  (N, 16, 768) fp32  L2-normalised per-frame video
    index_meta.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import cavmae_common as C


def l2(x, axis=-1):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(n, 1e-8, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir", default=str(C.EMB_DIR))
    args = ap.parse_args()
    emb_dir = Path(args.emb_dir)

    arrays, meta, seg_ids = C.open_store(emb_dir, mmap_mode="r")
    n = len(seg_ids)
    out = emb_dir / "index"
    out.mkdir(exist_ok=True)

    ca = np.asarray(arrays["cls_audio"], dtype=np.float32)   # (N,16,768)
    cv = np.asarray(arrays["cls_video"], dtype=np.float32)

    frame_audio = l2(ca)                 # per-frame normalised
    frame_video = l2(cv)
    clip_audio = l2(ca.mean(axis=1))     # mean-pool then normalise
    clip_video = l2(cv.mean(axis=1))

    np.save(out / "clip_audio.npy", clip_audio)
    np.save(out / "clip_video.npy", clip_video)
    np.save(out / "frame_audio.npy", frame_audio.astype(np.float32))
    np.save(out / "frame_video.npy", frame_video.astype(np.float32))

    (out / "index_meta.json").write_text(json.dumps({
        "n_segments": n,
        "dim": int(clip_audio.shape[1]),
        "total_frame": int(frame_audio.shape[1]),
        "normalised": True,
        "pooling": "mean",
    }, indent=2))
    print(f"[build_index] wrote clip + frame indices for {n} segments to {out}")


if __name__ == "__main__":
    main()
