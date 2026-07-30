"""
CAV-MAE Sync corpus search-engine + localization web app.

Run:
    source venv/bin/activate
    python webapp/server.py            # http://localhost:8000

Features:
  * Audio -> Video and Video -> Audio retrieval over the whole corpus
  * Query by an existing segment, or upload your own frame / sound
  * Strategies: fast (mean-pool), diagonal_mean/max, mean/max
  * Sound-source localization heatmap overlaid on any frame
"""

from __future__ import annotations

import io
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import cavmae_common as C  # noqa: E402
from retrieval_engine import get_engine  # noqa: E402

app = FastAPI(title="Corpus AV Search")

ENGINE = get_engine()
FRAMES_DIR = C.DATA_DIR / "frames"
AUDIO_DIR = C.DATA_DIR / "audio"
_MODEL = None  # lazy: only loaded when an upload query needs it


def model():
    global _MODEL
    if _MODEL is None:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _MODEL = C.build_model(device=dev)
    return _MODEL


# --------------------------------------------------------------------------
# Metadata / browse
# --------------------------------------------------------------------------
@app.get("/api/meta")
def meta():
    from collections import Counter
    series = Counter(p.get("series_id") for p in ENGINE.prov.values())
    types = Counter(p.get("media_type") for p in ENGINE.prov.values())
    return {
        "n_segments": len(ENGINE.seg_ids),
        "series": [{"id": k, "title": ENGINE.series_reference.get(k, {}).get("title", k),
                    "count": c} for k, c in sorted(series.items())],
        "media_types": dict(types),
        "strategies": ["fast", "diagonal_mean", "diagonal_max", "mean", "max"],
    }


@app.get("/api/segments")
def segments(q: str = "", series: str = "", media_type: str = "",
             silent: str = "", parent: str = "", offset: int = 0, limit: int = 60):
    ql = q.lower()
    out = []
    for i, sid in enumerate(ENGINE.seg_ids):
        p = ENGINE.prov.get(sid, {})
        if parent and p.get("parent_video_id") != parent:
            continue
        if series and p.get("series_id") != series:
            continue
        if media_type and p.get("media_type") != media_type:
            continue
        if silent == "yes" and not p.get("silent_source"):
            continue
        if silent == "no" and p.get("silent_source"):
            continue
        if ql and ql not in sid.lower() and ql not in str(p.get("title", "")).lower():
            continue
        out.append(_seg_brief(i, sid, p))
    if parent:                                              # whole film, in order
        out.sort(key=lambda b: b.get("start_sec") or 0)
        return {"total": len(out), "items": out}
    total = len(out)
    return {"total": total, "items": out[offset:offset + limit]}


@app.get("/api/films")
def films(series: str = "", media_type: str = "", silent: str = ""):
    """List the films (parent videos) of a series with matching segment counts."""
    agg: dict[str, dict] = {}
    for sid in ENGINE.seg_ids:
        p = ENGINE.prov.get(sid, {})
        if series and p.get("series_id") != series:
            continue
        if media_type and p.get("media_type") != media_type:
            continue
        if silent == "yes" and not p.get("silent_source"):
            continue
        if silent == "no" and p.get("silent_source"):
            continue
        pid = p.get("parent_video_id") or sid
        f = agg.get(pid)
        if f is None:
            agg[pid] = {
                "parent_id": pid,
                "title": p.get("title") or pid,
                "year": p.get("year"),
                "episode": p.get("episode"),
                "series_id": p.get("series_id"),
                "media_type": p.get("media_type"),
                "count": 1,
            }
        else:
            f["count"] += 1
    out = sorted(agg.values(),
                 key=lambda f: (f.get("year") or 0, f.get("episode") or 0, f["parent_id"]))
    return {"series": series, "total": len(out), "films": out}



def _seg_brief(i, sid, p):
    return {
        "seg_id": sid, "index": i,
        "title": p.get("title"), "series_id": p.get("series_id"),
        "series_title": p.get("series_title"), "year": p.get("year"),
        "media_type": p.get("media_type"), "start_sec": p.get("start_sec"),
        "end_sec": p.get("end_sec"), "silent_source": p.get("silent_source", False),
        "parent_video_id": p.get("parent_video_id"),
        "segment_index": p.get("segment_index"), "num_segments": p.get("num_segments"),
    }


@app.get("/api/segment/{seg_id}")
def segment(seg_id: str):
    if seg_id not in ENGINE.id2idx:
        raise HTTPException(404, "unknown segment")
    p = dict(ENGINE.provenance(seg_id))
    p["audible"] = _audible(seg_id)
    return p


@lru_cache(maxsize=8192)
def _audible(seg_id: str) -> bool:
    """True if the segment's wav actually carries sound.

    The metadata `has_audio` flag only means the source container had an audio
    stream; many silent-era sources ship a digitally silent (all-zero) track.
    """
    import soundfile as sf
    path = AUDIO_DIR / f"{seg_id}.wav"
    if not path.exists():
        return False
    data, _ = sf.read(str(path))
    return float(np.abs(data).max()) > 1e-4



# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------
@app.get("/api/retrieve")
def retrieve(seg_id: str, direction: str = "audio2video",
             strategy: str = "fast", top_k: int = 24,
             exclude_parent: bool = True):
    if seg_id not in ENGINE.id2idx:
        raise HTTPException(404, "unknown segment")
    idx = ENGINE.id2idx[seg_id]
    res = ENGINE.search(idx, direction=direction, strategy=strategy,
                        top_k=top_k, exclude_self_parent=exclude_parent)
    res["query"] = _seg_brief(idx, seg_id, ENGINE.provenance(seg_id))
    return res


@app.post("/api/retrieve_upload")
async def retrieve_upload(kind: str = Form(...), strategy: str = Form("fast"),
                          top_k: int = Form(24), file: UploadFile = File(...)):
    """kind='image' -> find matching SOUNDS ; kind='audio' -> find matching FRAMES."""
    raw = await file.read()
    m = model()
    if kind == "image":
        from PIL import Image
        img = C.preprocess_image(Image.open(io.BytesIO(raw))).unsqueeze(0)
        q = C.encode_media(m, images=img)["video"][0]
        db = ENGINE.clip_audio
        direction = "video2audio"
    elif kind == "audio":
        tmp = ROOT / "webapp" / "_upload_tmp.wav"
        tmp.write_bytes(raw)
        fb = C.preprocess_audio(tmp).unsqueeze(0)
        q = C.encode_media(m, fbanks=fb)["audio"][0]
        db = ENGINE.clip_video
        direction = "audio2video"
    else:
        raise HTTPException(400, "kind must be 'image' or 'audio'")

    scores = db @ q
    order = np.argsort(-scores)[:top_k]
    results = []
    for i in order:
        sid = ENGINE.seg_ids[int(i)]
        p = ENGINE.provenance(sid)
        b = _seg_brief(int(i), sid, p)
        b["score"] = float(scores[int(i)])
        b["parent_filename"] = p.get("parent_filename")
        results.append(b)
    return {"direction": direction, "strategy": "fast(upload)", "results": results}


# --------------------------------------------------------------------------
# Localization
# --------------------------------------------------------------------------
@app.get("/api/localize/{seg_id}/{frame_idx}")
def localize(seg_id: str, frame_idx: int, alpha: float = 0.5):
    if seg_id not in ENGINE.id2idx:
        raise HTTPException(404, "unknown segment")
    if not (0 <= frame_idx < C.TOTAL_FRAME):
        raise HTTPException(400, "frame_idx out of range")
    idx = ENGINE.id2idx[seg_id]
    heat = ENGINE.localize(idx, frame_idx)                    # (14,14)

    from PIL import Image
    import matplotlib.cm as cm
    frame_path = FRAMES_DIR / f"frame_{frame_idx}" / f"{seg_id}.jpg"
    base = Image.open(frame_path).convert("RGB").resize((224, 224))

    h = heat - heat.min()
    h = h / (h.max() + 1e-8)
    h_img = Image.fromarray((h * 255).astype(np.uint8)).resize((224, 224), Image.BICUBIC)
    h_arr = np.asarray(h_img) / 255.0
    color = (cm.jet(h_arr)[:, :, :3] * 255).astype(np.uint8)
    overlay = (np.asarray(base) * (1 - alpha) + color * alpha).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


# --------------------------------------------------------------------------
# Audio localization (spectrogram activation map)
# --------------------------------------------------------------------------
SPEC_W, SPEC_H = 624, 192          # display size of the 4 s window overlay
SPEC_FULL_W = 1560                 # display width of the full 10 s spectrogram


def _spec_image(spec: np.ndarray, w: int, h: int):
    """Normalise a (freq, time) array and render a resized grayscale image."""
    from PIL import Image
    spec = np.flipud(spec)                                   # low freq at bottom
    spec = spec - spec.min()
    spec = spec / (spec.max() + 1e-8)
    img = Image.fromarray((spec * 255).astype(np.uint8))     # 'L'
    return img.convert("RGB").resize((w, h), Image.BICUBIC)


def _spectrogram_base(seg_id: str, frame_idx: int):
    """Grayscale spectrogram (freq x time) for one frame's 4 s window."""
    wav_path = AUDIO_DIR / f"{seg_id}.wav"
    if not wav_path.exists():
        raise HTTPException(404, "no audio")
    win = C.frame_fbank_window(wav_path, frame_idx)          # (416 time, 128 freq)
    return _spec_image(win.T, SPEC_W, SPEC_H)


def _spectrogram_full(seg_id: str):
    """Grayscale spectrogram (freq x time) for the whole ~10 s segment."""
    wav_path = AUDIO_DIR / f"{seg_id}.wav"
    if not wav_path.exists():
        raise HTTPException(404, "no audio")
    full = C.full_fbank(wav_path)                            # (1024 time, 128 freq)
    return _spec_image(full.T, SPEC_FULL_W, SPEC_H)


@app.get("/api/spectrogram/{seg_id}/{frame_idx}")
def spectrogram(seg_id: str, frame_idx: int):
    if not (0 <= frame_idx < C.TOTAL_FRAME):
        raise HTTPException(400, "frame_idx out of range")
    base = _spectrogram_base(seg_id, frame_idx)
    buf = io.BytesIO()
    base.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.get("/api/spectrogram_full/{seg_id}")
def spectrogram_full(seg_id: str):
    base = _spectrogram_full(seg_id)
    buf = io.BytesIO()
    base.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.get("/api/localize_audio/{seg_id}/{frame_idx}")
def localize_audio(seg_id: str, frame_idx: int, alpha: float = 0.55):
    if seg_id not in ENGINE.id2idx:
        raise HTTPException(404, "unknown segment")
    if not (0 <= frame_idx < C.TOTAL_FRAME):
        raise HTTPException(400, "frame_idx out of range")
    idx = ENGINE.id2idx[seg_id]
    heat = ENGINE.localize_audio(idx, frame_idx)              # (8 freq, 26 time)

    from PIL import Image
    import matplotlib.cm as cm
    base = _spectrogram_base(seg_id, frame_idx)

    h = heat - heat.min()
    h = h / (h.max() + 1e-8)
    h = np.flipud(h)                                         # match spectrogram orientation
    h_img = Image.fromarray((h * 255).astype(np.uint8)).resize((SPEC_W, SPEC_H), Image.BICUBIC)
    h_arr = np.asarray(h_img) / 255.0
    color = (cm.jet(h_arr)[:, :, :3] * 255).astype(np.uint8)
    overlay = (np.asarray(base) * (1 - alpha) + color * alpha).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


# --------------------------------------------------------------------------
# Activation-gated audio playback
# --------------------------------------------------------------------------
@app.get("/api/filter_audio/{seg_id}/{frame_idx}")
def filter_audio(seg_id: str, frame_idx: int, lo: float = 0.0, hi: float = 1.0):
    """Resynthesise the frame's 4 s audio window keeping only time-frequency
    cells whose (normalised) image-grounding activation is within [lo, hi].

    The 8x26 patch_audio activation map is upsampled onto the STFT grid and used
    as a binary mask on the complex spectrogram; the original phase is kept, so
    the result is real audio (not vocoded), just spectrally/temporally gated.
    """
    if seg_id not in ENGINE.id2idx:
        raise HTTPException(404, "unknown segment")
    if not (0 <= frame_idx < C.TOTAL_FRAME):
        raise HTTPException(400, "frame_idx out of range")
    import torchaudio
    import soundfile as sf

    wav_path = AUDIO_DIR / f"{seg_id}.wav"
    if not wav_path.exists():
        raise HTTPException(404, "no audio")
    wav, sr = torchaudio.load(str(wav_path))
    wav = wav.mean(0)                                        # mono (samples,)

    # 4 s window aligned to this frame (same mapping the model used)
    hop, n_fft = 160, 400
    start_f, _ = C.map_frame_to_spectrogram(
        frame_idx, C.TOTAL_FRAME, C.FBANK_FULL_LEN, C.TARGET_LENGTH)
    s = start_f * hop
    n = C.TARGET_LENGTH * hop
    seg = wav[s:s + n]
    if seg.numel() < n:
        seg = torch.nn.functional.pad(seg, (0, n - seg.numel()))

    # normalised activation (matches the jet overlay: blue~0, red~1)
    heat = ENGINE.localize_audio(ENGINE.id2idx[seg_id], frame_idx)  # (8,26)
    h = heat - heat.min()
    h = h / (h.max() + 1e-8)
    h = torch.from_numpy(h).float()                         # (8 freq, 26 time)

    window = torch.hann_window(n_fft)
    stft = torch.stft(seg, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                      window=window, return_complex=True, center=True)  # (F,T)
    n_freq, n_time = stft.shape

    # map STFT freq bins -> mel band 0..7 (HTK mel, matching kaldi fbank)
    freqs = torch.linspace(0, sr / 2, n_freq)
    mel = 1127.0 * torch.log(1 + freqs / 700.0)
    mel_max = 1127.0 * torch.log(torch.tensor(1 + (sr / 2) / 700.0))
    band = ((mel / mel_max) * C.NUM_MEL_BINS).clamp(0, C.NUM_MEL_BINS - 1).long() // 16
    band = band.clamp(0, 7)                                 # (n_freq,)
    # map STFT frames -> activation time step 0..25
    t_idx = (torch.arange(n_time).float() / max(1, n_time - 1) * 25).round().clamp(0, 25).long()

    act = h[band][:, t_idx]                                 # (n_freq, n_time)
    mask = ((act >= lo) & (act <= hi)).float()
    out = torch.istft(stft * mask, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                      window=window, length=n)
    peak = out.abs().max()
    if peak > 0:
        out = out / peak * 0.98

    buf = io.BytesIO()
    sf.write(buf, out.numpy(), sr, format="WAV")
    buf.seek(0)
    return StreamingResponse(buf, media_type="audio/wav")


@app.get("/api/filter_spectrogram/{seg_id}/{frame_idx}")
def filter_spectrogram(seg_id: str, frame_idx: int, lo: float = 0.0, hi: float = 1.0):
    """Frame's 4 s spectrogram with time-frequency cells outside the activation
    band [lo, hi] darkened - a live preview of what `filter_audio` will play."""
    if seg_id not in ENGINE.id2idx:
        raise HTTPException(404, "unknown segment")
    if not (0 <= frame_idx < C.TOTAL_FRAME):
        raise HTTPException(400, "frame_idx out of range")
    from PIL import Image
    wav_path = AUDIO_DIR / f"{seg_id}.wav"
    if not wav_path.exists():
        raise HTTPException(404, "no audio")

    win = C.frame_fbank_window(wav_path, frame_idx).T        # (128 freq, 416 time)
    spec = np.flipud(win)
    spec = spec - spec.min()
    spec = spec / (spec.max() + 1e-8)

    heat = ENGINE.localize_audio(ENGINE.id2idx[seg_id], frame_idx)   # (8,26)
    h = heat - heat.min()
    h = h / (h.max() + 1e-8)
    h = np.flipud(h)                                        # match spectrogram orientation
    keep = ((h >= lo) & (h <= hi)).astype(np.float32)       # (8,26)
    keep_img = np.asarray(Image.fromarray((keep * 255).astype(np.uint8))
                          .resize((spec.shape[1], spec.shape[0]), Image.NEAREST)) / 255.0

    # kept cells: full brightness; removed cells: dimmed to 12%
    masked = spec * (0.12 + 0.88 * keep_img)
    img = Image.fromarray((masked * 255).astype(np.uint8)).convert("RGB")
    img = img.resize((SPEC_W, SPEC_H), Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


# --------------------------------------------------------------------------
# Media files
# --------------------------------------------------------------------------
@app.get("/media/frame/{seg_id}/{frame_idx}")
def frame(seg_id: str, frame_idx: int):
    path = FRAMES_DIR / f"frame_{frame_idx}" / f"{seg_id}.jpg"
    if not path.exists():
        raise HTTPException(404, "no frame")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/media/audio/{seg_id}")
def audio(seg_id: str):
    path = AUDIO_DIR / f"{seg_id}.wav"
    if not path.exists():
        raise HTTPException(404, "no audio")
    return FileResponse(path, media_type="audio/wav")


# static frontend (mounted last so /api and /media take precedence)
app.mount("/", StaticFiles(directory=str(Path(__file__).parent / "static"), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
