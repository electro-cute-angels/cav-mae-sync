"""
Extract CAV-MAE Sync embeddings for every segment in cavmae_data/full.json
and write them into the on-disk store under ./embeddings/.

Saved per segment (16 frames each):
    cls_audio   (16, 768)        fp32   -> retrieval / search engine
    cls_video   (16, 768)        fp32   -> retrieval / search engine
    patch_audio (16, 208, 768)   fp16   -> fine-grained audio experiments
    patch_video (16, 196, 768)   fp16   -> localization heatmaps (14x14 grid)

The store is memmapped and written sequentially, so the job is resumable:
re-run with --resume to continue from the last completed segment.

Usage:
    python scripts/extract_embeddings.py --batch-size 12 --num-workers 12
    python scripts/extract_embeddings.py --resume
    python scripts/extract_embeddings.py --no-patches      # cls only (2 GB)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import cavmae_common as C


def collate(batch):
    fbanks, images, labels, vids, fidx = zip(*batch)
    return (torch.stack(fbanks), torch.stack(images), list(vids))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir", default=str(C.EMB_DIR))
    ap.add_argument("--batch-size", type=int, default=12, help="segments per batch")
    ap.add_argument("--num-workers", type=int, default=12)
    ap.add_argument("--data-parallel", action="store_true",
                    help="use all visible GPUs via DataParallel")
    ap.add_argument("--no-patches", dest="save_patches", action="store_false",
                    help="store only cls embeddings")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="debug: first N segments")
    args = ap.parse_args()

    emb_dir = Path(args.emb_dir)
    seg_ids = C.load_segment_order()
    if args.limit:
        seg_ids = seg_ids[: args.limit]
    n = len(seg_ids)

    device = "cuda"
    model = C.build_model(device=device)
    if args.data_parallel and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"[extract] DataParallel over {torch.cuda.device_count()} GPUs")

    progress_path = emb_dir / "progress.json"

    # --- open or create store -------------------------------------------------
    if args.resume and (emb_dir / "meta.json").exists():
        arrays, meta, stored_ids = C.open_store(emb_dir, mmap_mode="r+")
        assert stored_ids[:n] == seg_ids, "segment order changed; cannot resume"
        cursor = json.loads(progress_path.read_text())["done"] if progress_path.exists() else 0
        print(f"[extract] resuming at segment {cursor}/{n}")
    else:
        spec = C.StoreSpec(n_segments=n)
        arrays = C.create_store(emb_dir, spec, seg_ids, save_patches=args.save_patches)
        cursor = 0
        progress_path.write_text(json.dumps({"done": 0, "total": n}))
        print(f"[extract] created store for {n} segments (patches={args.save_patches})")

    # --- dataset / loader over the remaining range ---------------------------
    ds = C.build_dataset(efficient=True)
    if args.limit:
        ds = Subset(ds, list(range(args.limit)))
    remaining = Subset(ds, list(range(cursor, n)))
    loader = DataLoader(remaining, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate,
                        pin_memory=True, persistent_workers=args.num_workers > 0)

    t0 = time.time()
    done = cursor
    with tqdm(total=n, initial=cursor, desc="segments") as bar:
        for fbanks, images, vids in loader:
            out = C.forward_features(model, fbanks, images, device=device)
            b = len(vids)
            sl = slice(done, done + b)
            arrays["cls_audio"][sl] = out["cls_audio"]
            arrays["cls_video"][sl] = out["cls_video"]
            if "patch_audio" in arrays:
                arrays["patch_audio"][sl] = out["patch_audio"]
                arrays["patch_video"][sl] = out["patch_video"]
            done += b
            progress_path.write_text(json.dumps({"done": done, "total": n}))
            bar.update(b)
            bar.set_postfix(seg_s=f"{(done - cursor) / (time.time() - t0):.1f}")

    for a in arrays.values():
        a.flush()
    print(f"[extract] complete: {done}/{n} segments in {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
