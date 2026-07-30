"""
Reusable foundation for all CAV-MAE Sync experiments on the corpus.

This module centralises everything that every downstream experiment
(embedding extraction, retrieval search engine, localization, future
probing / fine-tuning) needs so that behaviour stays consistent:

  * exact model configuration decoded from the released checkpoint
  * exact audio / image preprocessing (re-uses the repo dataloader)
  * a single `build_model()` entry point
  * a single `forward_features()` that returns the 4 embedding tensors
    grouped per segment: (cls_audio, cls_video, patch_audio, patch_video)
  * helpers to read/write the on-disk embedding store

Nothing here is specific to one experiment; import it everywhere.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_SRC = PROJECT_ROOT / "external" / "cav-mae-sync" / "src"
CKPT_PATH = PROJECT_ROOT / "models" / "cav_mae_sync.pth"
DATA_DIR = PROJECT_ROOT / "cavmae_data"
DATAFILE = DATA_DIR / "datafiles" / "full.json"
LABEL_CSV = DATA_DIR / "class_labels_indices.csv"
EMB_DIR = PROJECT_ROOT / "embeddings"

# Make the repo importable (models, dataloader_sync live under src/)
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))


def _patch_torchaudio_load() -> None:
    """Newer torchaudio routes load() through TorchCodec (not installed).

    The repo dataloader calls ``torchaudio.load``; our audio is plain 16 kHz
    mono WAV, so back it with soundfile and return a (channels, samples)
    float32 tensor exactly like the old torchaudio backend did.
    """
    import torchaudio

    try:  # if a working backend already exists, leave it alone
        torchaudio.load(str(DATA_DIR / "audio" / "__nonexistent__.wav"))
    except FileNotFoundError:
        return
    except Exception:
        pass

    import soundfile as sf

    def _sf_load(filepath, *a, **k):
        data, sr = sf.read(str(filepath), dtype="float32", always_2d=True)
        return torch.from_numpy(data.T).contiguous(), sr

    torchaudio.load = _sf_load


_patch_torchaudio_load()


# --------------------------------------------------------------------------
# Model / preprocessing configuration (decoded from cav_mae_sync.pth)
# --------------------------------------------------------------------------
# register_tokens weight is (16, 768) -> num_register_tokens = 16 // 2 = 8
# pos_embed_a is (1, 208, 768) -> audio_length = 208 / 8 * 16 = 416 (4s window)
# cls_token_a present -> cls_token=True ; no contrastive_head_* keys -> False
MODEL_CONFIG = dict(
    img_size=224,
    audio_length=416,          # 4 second audio window (target_length)
    patch_size=16,
    embed_dim=768,
    modality_specific_depth=11,
    num_heads=12,
    num_register_tokens=8,
    cls_token=True,
    global_local_losses=False,  # loss-only flag, adds no params
    total_frame=16,
    contrastive_heads=False,
)

# Normalisation stats shipped with the released retrieval config.
NORM_MEAN = -5.081
NORM_STD = 4.4849
TARGET_LENGTH = MODEL_CONFIG["audio_length"]   # 416
TOTAL_FRAME = MODEL_CONFIG["total_frame"]       # 16
NUM_MEL_BINS = 128
IM_RES = 224
FBANK_FULL_LEN = 1024      # full per-segment spectrogram length (~10.24 s @ 10 ms)

# Token counts produced per frame by forward_feat (after dropping cls+registers)
N_AUDIO_PATCHES = 208          # 416/16 * 128/16
N_VIDEO_PATCHES = 196          # 14 x 14
EMBED_DIM = MODEL_CONFIG["embed_dim"]


def audio_conf(mode: str = "retrieval") -> dict:
    """Audio/image config dict consumed by the repo AudiosetDataset."""
    return {
        "num_mel_bins": NUM_MEL_BINS,
        "target_length": TARGET_LENGTH,
        "freqm": 0,
        "timem": 0,
        "mixup": 0,
        "dataset": "corpus",
        "mode": mode,
        "mean": NORM_MEAN,
        "std": NORM_STD,
        "noise": False,
        "label_smooth": 0,
        "im_res": IM_RES,
        "frame_use": TOTAL_FRAME // 2,
        "total_frame": TOTAL_FRAME,
        "skip_norm": False,
    }


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def build_model(ckpt_path: os.PathLike | str = CKPT_PATH,
                device: str | torch.device = "cuda") -> torch.nn.Module:
    """Instantiate CAV-MAE Sync and load the pretrained checkpoint (strict)."""
    import models  # from repo src, made importable above

    model = models.CAVMAESync(
        img_size=MODEL_CONFIG["img_size"],
        audio_length=MODEL_CONFIG["audio_length"],
        patch_size=MODEL_CONFIG["patch_size"],
        embed_dim=MODEL_CONFIG["embed_dim"],
        modality_specific_depth=MODEL_CONFIG["modality_specific_depth"],
        num_heads=MODEL_CONFIG["num_heads"],
        num_register_tokens=MODEL_CONFIG["num_register_tokens"],
        cls_token=MODEL_CONFIG["cls_token"],
        global_local_losses=MODEL_CONFIG["global_local_losses"],
        total_frame=MODEL_CONFIG["total_frame"],
        contrastive_heads=MODEL_CONFIG["contrastive_heads"],
    )

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.state_dict() if hasattr(ckpt, "state_dict") else ckpt
    # strip DataParallel 'module.' prefix
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    # tolerate decoder / loss-only params that forward_feat never touches
    missing = [m for m in missing if not m.startswith(("decoder", "mask_token"))]
    if missing:
        print(f"[build_model] WARNING missing keys (non-decoder): {missing[:10]}")
    model.eval().to(device)
    return model


@torch.no_grad()
def forward_features(model: torch.nn.Module,
                     fbanks: torch.Tensor,
                     images: torch.Tensor,
                     device: str | torch.device = "cuda"):
    """Run forward_feat on a batch of segments.

    Args:
        fbanks:  (B, 16, target_length, 128)
        images:  (B, 16, 3, 224, 224)
    Returns dict of numpy arrays grouped per segment:
        cls_audio   (B, 16, 768)          fp32
        cls_video   (B, 16, 768)          fp32
        patch_audio (B, 16, 208, 768)     fp16
        patch_video (B, 16, 196, 768)     fp16
    """
    B, F = fbanks.shape[0], fbanks.shape[1]
    a = fbanks.reshape(B * F, fbanks.shape[2], fbanks.shape[3]).to(device)
    v = images.reshape(B * F, *images.shape[2:]).to(device)

    core = model.module if isinstance(model, torch.nn.DataParallel) else model
    with torch.amp.autocast("cuda"):
        tok_a, tok_v, cls_a, cls_v = core.forward_feat(a, v)

    def group(x, n_tok=None):
        x = x.float().cpu()
        if n_tok is None:  # cls: (B*F, 768)
            return x.view(B, F, -1).numpy()
        return x.view(B, F, n_tok, -1).numpy()

    return {
        "cls_audio": group(cls_a).astype(np.float32),
        "cls_video": group(cls_v).astype(np.float32),
        "patch_audio": group(tok_a, N_AUDIO_PATCHES).astype(np.float16),
        "patch_video": group(tok_v, N_VIDEO_PATCHES).astype(np.float16),
    }


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
def build_dataset(datafile: os.PathLike | str = DATAFILE,
                  label_csv: os.PathLike | str = LABEL_CSV,
                  mode: str = "retrieval",
                  efficient: bool = True):
    """Return the repo AudiosetDataset in retrieval mode (16 frames/segment).

    When ``efficient`` is True, an override computes the fbank a single time
    per segment (the stock retrieval __getitem__ recomputes it 16x, once per
    frame). Output is numerically identical to the stock dataloader.
    """
    import dataloader_sync

    if not efficient or mode != "retrieval":
        return dataloader_sync.AudiosetDataset(
            str(datafile), audio_conf=audio_conf(mode), label_csv=str(label_csv)
        )

    class EfficientRetrievalDataset(dataloader_sync.AudiosetDataset):
        def __getitem__(self, index):
            datum = self.decode_data(self.data[index])
            fbank_full = self._wav2fbank(datum["wav"])          # (1024, 128)
            fbanks, images, frame_indices = [], [], []
            for frame_idx in range(self.total_frame):
                start, end = self.map_frame_to_spectrogram(
                    frame_index=frame_idx, num_frames=self.total_frame,
                    spectrogram_length=fbank_full.shape[0],
                    target_length=self.target_length,
                )
                fbank = fbank_full[start:end, :]
                if not self.skip_norm:
                    fbank = (fbank - self.norm_mean) / self.norm_std
                frame_path = (f"{datum['video_path']}/frame_{frame_idx}/"
                              f"{datum['video_id']}.jpg")
                try:
                    image = self.get_image(frame_path)
                except Exception:
                    image = (images[-1].clone() if images
                             else torch.zeros(3, self.im_res, self.im_res))
                    self.failed_image_loadings += 1
                fbanks.append(fbank)
                images.append(image)
                frame_indices.append(frame_idx)
            label_indices = np.zeros(self.label_num) + (self.label_smooth / self.label_num)
            for label_str in datum["labels"].split(","):
                label_indices[int(self.index_dict[label_str])] = 1.0 - self.label_smooth
            return (torch.stack(fbanks), torch.stack(images),
                    torch.FloatTensor(label_indices), datum["video_id"],
                    torch.tensor(frame_indices))

    return EfficientRetrievalDataset(
        str(datafile), audio_conf=audio_conf(mode), label_csv=str(label_csv)
    )


def load_segment_order(datafile: os.PathLike | str = DATAFILE) -> list[str]:
    """Ordered list of segment video_ids as they appear in the datafile."""
    with open(datafile) as f:
        data = json.load(f)["data"]
    return [d["video_id"] for d in data]


# --------------------------------------------------------------------------
# Encoding NEW media (uploaded frame / sound) with the model
# --------------------------------------------------------------------------
# The audio and visual streams in forward_feat are processed independently
# (separate blocks then blocks_u with a modality tag), so we can obtain a
# video embedding by pairing the image with a dummy audio, and vice versa.

def preprocess_image(pil_img) -> torch.Tensor:
    """Replicate the dataloader's eval image transform -> (3,224,224)."""
    import PIL
    import torchvision.transforms as T
    tf = T.Compose([
        T.Resize(IM_RES, interpolation=PIL.Image.BICUBIC),
        T.CenterCrop(IM_RES),
        T.ToTensor(),
        T.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]),
    ])
    return tf(pil_img.convert("RGB"))


def preprocess_audio(wav_path: os.PathLike | str) -> torch.Tensor:
    """Replicate the dataloader fbank + centre 4s window -> (416,128)."""
    import torchaudio
    waveform, sr = torchaudio.load(str(wav_path))
    waveform = waveform - waveform.mean()
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
        window_type="hanning", num_mel_bins=NUM_MEL_BINS, dither=0.0, frame_shift=10)
    tl = 1024
    if fbank.shape[0] < tl:
        fbank = torch.nn.ZeroPad2d((0, 0, 0, tl - fbank.shape[0]))(fbank)
    else:
        fbank = fbank[:tl, :]
    # centre target-length window
    start = max(0, fbank.shape[0] // 2 - TARGET_LENGTH // 2)
    fbank = fbank[start:start + TARGET_LENGTH, :]
    if fbank.shape[0] < TARGET_LENGTH:
        fbank = torch.nn.ZeroPad2d((0, 0, 0, TARGET_LENGTH - fbank.shape[0]))(fbank)
    return (fbank - NORM_MEAN) / NORM_STD


def map_frame_to_spectrogram(frame_index: int, num_frames: int,
                             spectrogram_length: int, target_length: int) -> tuple[int, int]:
    """Exact copy of the dataloader's frame->spectrogram window mapping."""
    frame_position = int(round(frame_index * spectrogram_length / num_frames))
    start = max(0, frame_position - target_length // 2)
    end = start + target_length
    if end > spectrogram_length:
        end = spectrogram_length
        start = max(0, end - target_length)
    return start, end


def frame_fbank_window(wav_path: os.PathLike | str, frame_idx: int) -> np.ndarray:
    """Raw (unnormalised) fbank window (target_length, 128) feeding one frame.

    Mirrors the efficient retrieval dataset: compute the full 1024-frame fbank
    once, then slice the target-length window that this frame attends to. Used
    to render a spectrogram for the audio activation-map overlay.
    """
    fbank = torch.from_numpy(full_fbank(wav_path))          # (1024, 128)
    start, end = map_frame_to_spectrogram(frame_idx, TOTAL_FRAME, FBANK_FULL_LEN, TARGET_LENGTH)
    window = fbank[start:end, :]
    if window.shape[0] < TARGET_LENGTH:
        window = torch.nn.ZeroPad2d((0, 0, 0, TARGET_LENGTH - window.shape[0]))(window)
    return window.numpy()                                   # (416, 128)


def full_fbank(wav_path: os.PathLike | str) -> np.ndarray:
    """Raw (unnormalised) full-length fbank (1024, 128) for the whole segment.

    This is the ~10.24 s spectrogram the 16 frames are sampled from; each frame
    only sees a 416-frame (~4 s) window of it (see frame_fbank_window).
    """
    import torchaudio
    waveform, sr = torchaudio.load(str(wav_path))
    waveform = waveform - waveform.mean()
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
        window_type="hanning", num_mel_bins=NUM_MEL_BINS, dither=0.0, frame_shift=10)
    if fbank.shape[0] < FBANK_FULL_LEN:
        fbank = torch.nn.ZeroPad2d((0, 0, 0, FBANK_FULL_LEN - fbank.shape[0]))(fbank)
    else:
        fbank = fbank[:FBANK_FULL_LEN, :]
    return fbank.numpy()                                    # (1024, 128)


@torch.no_grad()
def encode_media(model, fbanks: torch.Tensor | None = None,
                 images: torch.Tensor | None = None,
                 device: str | torch.device = "cuda") -> dict:
    """Encode a batch of audio and/or images into L2-normalised cls vectors.

    fbanks: (B,416,128) or None ; images: (B,3,224,224) or None.
    Returns {'audio': (B,768)|None, 'video': (B,768)|None} numpy fp32.
    """
    b = fbanks.shape[0] if fbanks is not None else images.shape[0]
    if fbanks is None:
        fbanks = torch.zeros(b, TARGET_LENGTH, NUM_MEL_BINS)
    if images is None:
        images = torch.zeros(b, 3, IM_RES, IM_RES)
    core = model.module if isinstance(model, torch.nn.DataParallel) else model
    a, v = fbanks.to(device), images.to(device)
    with torch.amp.autocast("cuda"):
        _, _, cls_a, cls_v = core.forward_feat(a, v)

    def norm(x):
        x = x.float().cpu().numpy().reshape(b, -1)
        return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8, None)

    return {"audio": norm(cls_a), "video": norm(cls_v)}


# --------------------------------------------------------------------------
# On-disk embedding store
# --------------------------------------------------------------------------
@dataclass
class StoreSpec:
    n_segments: int
    total_frame: int = TOTAL_FRAME
    embed_dim: int = EMBED_DIM
    n_audio_patches: int = N_AUDIO_PATCHES
    n_video_patches: int = N_VIDEO_PATCHES


STORE_ARRAYS = {
    # name: (shape_fn, dtype)
    "cls_audio":   (lambda s: (s.n_segments, s.total_frame, s.embed_dim), np.float32),
    "cls_video":   (lambda s: (s.n_segments, s.total_frame, s.embed_dim), np.float32),
    "patch_audio": (lambda s: (s.n_segments, s.total_frame, s.n_audio_patches, s.embed_dim), np.float16),
    "patch_video": (lambda s: (s.n_segments, s.total_frame, s.n_video_patches, s.embed_dim), np.float16),
}


def create_store(emb_dir: os.PathLike | str, spec: StoreSpec,
                 seg_ids: list[str], save_patches: bool = True) -> dict[str, np.memmap]:
    """Preallocate memmapped .npy files for the embedding store."""
    emb_dir = Path(emb_dir)
    emb_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for name, (shape_fn, dtype) in STORE_ARRAYS.items():
        if name.startswith("patch") and not save_patches:
            continue
        path = emb_dir / f"{name}.npy"
        arrays[name] = np.lib.format.open_memmap(
            path, mode="w+", dtype=dtype, shape=shape_fn(spec)
        )
    (emb_dir / "seg_ids.json").write_text(json.dumps(seg_ids))
    meta = {
        "model_config": MODEL_CONFIG,
        "norm_mean": NORM_MEAN,
        "norm_std": NORM_STD,
        "target_length": TARGET_LENGTH,
        "num_mel_bins": NUM_MEL_BINS,
        "im_res": IM_RES,
        "n_segments": spec.n_segments,
        "save_patches": save_patches,
        "arrays": {name: {"shape": list(STORE_ARRAYS[name][0](spec)),
                          "dtype": np.dtype(STORE_ARRAYS[name][1]).name}
                   for name in arrays},
    }
    (emb_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return arrays


def open_store(emb_dir: os.PathLike | str = EMB_DIR,
               mmap_mode: str = "r") -> tuple[dict[str, np.memmap], dict, list[str]]:
    """Open an existing embedding store for reading (or 'r+' to append)."""
    emb_dir = Path(emb_dir)
    meta = json.loads((emb_dir / "meta.json").read_text())
    seg_ids = json.loads((emb_dir / "seg_ids.json").read_text())
    arrays = {}
    for name in meta["arrays"]:
        arrays[name] = np.load(emb_dir / f"{name}.npy", mmap_mode=mmap_mode)
    return arrays, meta, seg_ids
