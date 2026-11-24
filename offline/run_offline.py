#!/usr/bin/env python3
# coding: utf-8
"""
Offline run of the webrtc_conference codec: a video file (+ audio) in, the
video the far side would see out. No camera, mic, GPU on the laptop, WebRTC or
UI needed — the sender, the wire format and the receiver are the repo's own
classes, called one frame at a time:

  sender    MotionEncoder.extract   (crop + motion extractor M)  -> MotionState
            protocol.pack_motion    (70 values, int16)           -> 150 B packet
            EncodecStreamCodec      (EnCodec 24 kHz, 240 ms chunks)
            protocol.pack_audio     (10-bit packed codes)        -> audio packet
            encode_avatar_jpeg      (portrait sent once)
  channel   bytes only; optional random motion loss (the motion datachannel is
            unreliable, a lost packet leaves the last frame on screen)
  receiver  protocol.unpack_*  ->  AvatarRenderer.render (W + G, 512x512)
            EncodecStreamCodec.decode + CrossfadeJoiner
            -> output.mp4 (video + decoded EnCodec audio)

Video frames are taken at --fps (the client's default is 20) on the source
video's timeline, and audio is cut into the same chunks the mic would deliver,
so packet sizes and bit rates are the ones a live call would send.

    conda activate video_semantics_receiver
    cd ~/FYP_SemCon/LivePortrait-encodec
    python offline/run_offline.py --video VIDEO.mp4 [--audio SPEECH.wav] [--avatar PORTRAIT.jpg]
"""

import argparse
import json
import os
import shutil
import ssl
import subprocess
import sys
import time
import wave
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def _fix_ssl():
    """The env's OpenSSL points at a CA file that no longer exists (left over
    from the conda env rename), so EnCodec's one-time weight download fails.
    Fall back to certifi's bundle."""
    if os.environ.get("SSL_CERT_FILE"):
        return
    if os.path.exists(ssl.get_default_verify_paths().openssl_cafile or ""):
        return
    try:
        import certifi
        os.environ["SSL_CERT_FILE"] = certifi.where()
    except ImportError:
        pass


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def read_audio(path: Path, sample_rate: int, ffmpeg: str):
    """Mono float32 PCM at `sample_rate`, or None if the file has no audio."""
    proc = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(path), "-vn", "-ac", "1",
         "-ar", str(sample_rate), "-f", "s16le", "-"],
        capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        return None
    return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def write_wav(path: Path, pcm: np.ndarray, sample_rate: int):
    data = (np.clip(pcm, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(data.tobytes())


def video_info(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, first = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"no frames in {path}")
    return fps, n, first


def frames_at(path: Path, fps: float, n_out: int):
    """Yield (k, bgr) for output frame k = the source frame nearest k / fps,
    i.e. what a camera polled at `fps` would have handed the encoder."""
    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    idx, frame = -1, None
    for k in range(n_out):
        want = int(round(k * src_fps / fps))
        while idx < want:
            ok, f = cap.read()
            if not ok:
                cap.release()
                return
            frame, idx = f, idx + 1
        yield k, frame
    cap.release()


class VideoWriter:
    """Raw BGR frames -> H.264 mp4 through an ffmpeg pipe."""

    def __init__(self, path: Path, size, fps: float, ffmpeg: str, crf: int = 18):
        w, h = size
        self.proc = subprocess.Popen(
            [ffmpeg, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf), str(path)],
            stdin=subprocess.PIPE)

    def write(self, bgr: np.ndarray):
        self.proc.stdin.write(np.ascontiguousarray(bgr).tobytes())

    def close(self):
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError("ffmpeg video encode failed")


def fixed_box_encoder(MotionEncoder):
    """MotionEncoder whose crop is fixed from the first frame with a face, as
    the FYP_SemCon transmitter does, instead of following the face every frame.
    The tracked crop moves with the head (and re-snaps at every re-detection),
    which removes real head motion and adds jitter; a fixed box keeps the head
    moving inside the crop, so that motion travels as pose/translation."""

    class FixedBoxEncoder(MotionEncoder):
        def __init__(self, engine):
            super().__init__(engine, crop_driving=True)
            self._m_o2c = None
            self._dsize = engine.crop_cfg.dsize

        def reset_tracking(self):
            super().reset_tracking()
            self._m_o2c = None

        def _crop(self, frame_bgr):
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if self._m_o2c is None:
                try:
                    with self.engine.lock:
                        info = self.engine.cropper.crop_source_image(frame_rgb, self.engine.crop_cfg)
                except Exception:
                    info = None
                if info is None:
                    return None
                self._m_o2c = np.asarray(info["M_o2c"], dtype=np.float64)[:2]
                return info["img_crop_256x256"]
            crop = cv2.warpAffine(frame_rgb, self._m_o2c, (self._dsize, self._dsize),
                                  flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
            return cv2.resize(crop, (256, 256), interpolation=cv2.INTER_AREA)

    return FixedBoxEncoder


def stats_ms(values):
    if not values:
        return None
    a = np.asarray(values, dtype=np.float64)
    return {"median": round(float(np.median(a)), 2),
            "p95": round(float(np.percentile(a, 95)), 2),
            "max": round(float(a.max()), 2)}


def run_audio(pcm: np.ndarray, args, P, EncodecStreamCodec, CrossfadeJoiner, SAMPLE_RATE):
    """Sender chunks -> EnCodec codes -> wire bytes -> receiver PCM."""
    tx = EncodecStreamCodec(bandwidth=args.bandwidth, device=args.audio_device,
                            chunk_ms=args.audio_chunk_ms)
    rx = EncodecStreamCodec(bandwidth=args.bandwidth, device=args.audio_device,
                            chunk_ms=args.audio_chunk_ms)
    joiner = CrossfadeJoiner()
    chunk = tx.chunk_samples
    n_chunks = (pcm.size + chunk - 1) // chunk
    padded = np.zeros(n_chunks * chunk, dtype=np.float32)
    padded[:pcm.size] = pcm

    out, sizes, enc_ms, dec_ms = [], [], [], []
    for i in range(n_chunks):
        piece = padded[i * chunk:(i + 1) * chunk]
        t_ms = int(round(i * chunk * 1000 / SAMPLE_RATE))
        codes = tx.encode(piece, SAMPLE_RATE)
        enc_ms.append(tx.stats.encode_ms)
        payload = P.pack_audio(i, t_ms, codes)
        sizes.append(len(payload))
        pkt = P.unpack_audio(payload)                 # receiver
        decoded = rx.decode(pkt.codes)
        dec_ms.append(rx.stats.decode_ms)
        out.append(joiner.push(decoded))
    if joiner._tail is not None:                      # a live stream never ends; flush
        out.append(joiner._tail)
    decoded = np.concatenate(out)[:pcm.size] if out else np.zeros(0, np.float32)
    info = {
        "bandwidth_kbps_setting": args.bandwidth,
        "codebooks": int(tx.n_q),
        "chunk_ms": round(chunk * 1000 / SAMPLE_RATE, 1),
        "packets": n_chunks,
        "packet_bytes": int(np.median(sizes)) if sizes else 0,
        "total_bytes": int(sum(sizes)),
        "encode_ms_per_chunk": stats_ms(enc_ms),
        "decode_ms_per_chunk": stats_ms(dec_ms),
        "device": args.audio_device,
    }
    return decoded, info


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, type=Path, help="driving video (the speaker)")
    ap.add_argument("--audio", type=Path, default=None,
                    help="speech audio (default: the video's own audio track)")
    ap.add_argument("--avatar", type=Path, default=None,
                    help="portrait sent once at call setup (default: first video frame "
                         "= self-reconstruction)")
    ap.add_argument("--output", type=Path, default=None,
                    help="output mp4 (default: offline/results/<video>_<time>/output.mp4)")
    ap.add_argument("--max-seconds", type=float, default=0, help="only the first N s")
    g = ap.add_argument_group("video (client.py defaults)")
    g.add_argument("--fps", type=float, default=20, help="motion vectors per second")
    g.add_argument("--det-interval", type=int, default=30)
    g.add_argument("--driving-crop", choices=("track", "fixed"), default="track",
                   help="track: their crop, follows the face every frame (re-detects every "
                        "--det-interval); fixed: box fixed from the first face (FYP_SemCon transmitter)")
    g.add_argument("--no-crop", action="store_true",
                   help="skip face detection (face already fills the frame)")
    g.add_argument("--no-crop-avatar", action="store_true",
                   help="use the whole portrait as the source (with --no-crop on an already-framed "
                        "square video: output is aligned with the input, for PSNR/SSIM/LPIPS)")
    g.add_argument("--driving-multiplier", type=float, default=1.0)
    g.add_argument("--avatar-quality", type=int, default=90)
    g.add_argument("--cpu", action="store_true", help="LivePortrait on the CPU")
    g = ap.add_argument_group("audio (client.py defaults)")
    g.add_argument("--no-audio", action="store_true")
    g.add_argument("--bandwidth", type=float, default=6.0, help="EnCodec kbit/s")
    g.add_argument("--audio-chunk-ms", type=int, default=240)
    g.add_argument("--audio-device", default="cpu", help="cpu | cuda")
    g.add_argument("--keep-original-audio", action="store_true",
                   help="mux the original audio instead of the EnCodec-decoded one")
    g = ap.add_argument_group("channel")
    g.add_argument("--motion-loss", type=float, default=0.0,
                   help="fraction of motion packets dropped at random (0..1)")
    g.add_argument("--seed", type=int, default=0)
    ap.add_argument("--side-by-side", action="store_true",
                    help="also write sender crop | received avatar")
    args = ap.parse_args()

    _fix_ssl()
    ffmpeg = _ffmpeg()

    src_fps, n_src, first_bgr = video_info(args.video)
    duration = n_src / src_fps
    if args.max_seconds:
        duration = min(duration, args.max_seconds)
    n_out = int(duration * args.fps)

    if args.output is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        work = REPO / "offline" / "results" / f"{args.video.stem}_{stamp}"
        args.output = work / "output.mp4"
    work = args.output.parent
    work.mkdir(parents=True, exist_ok=True)

    # Imported here so --help works without the models.
    from webrtc_conference import protocol as P
    from webrtc_conference.motion_codec import (RENDER_SIZE, AvatarRenderer, LivePortraitEngine,
                                                MotionEncoder, decode_avatar_jpeg,
                                                encode_avatar_jpeg, load_avatar_rgb)

    print(f"video {args.video.name}: {n_src} frames @ {src_fps:.2f} fps -> "
          f"{n_out} frames @ {args.fps:g} fps ({duration:.1f} s)")
    ratio = src_fps / args.fps
    if abs(ratio - round(ratio)) > 1e-3:
        print(f"  note: {src_fps:g} -> {args.fps:g} fps is not a whole-frame step ({ratio:.2f}); "
              f"frames are taken unevenly and motion will judder. Use --fps "
              f"{src_fps / max(1, int(ratio)):g} or {src_fps / (int(ratio) + 1):g} for even steps.")
    t_start = time.perf_counter()

    # ── audio ───────────────────────────────────────────────────────────────
    audio_info, decoded, original = None, None, None
    if not args.no_audio:
        from webrtc_conference.audio_codec import (SAMPLE_RATE, CrossfadeJoiner,
                                                   EncodecStreamCodec)
        original = read_audio(args.audio or args.video, SAMPLE_RATE, ffmpeg)
        if original is None:
            print(f"  no audio in {args.audio or args.video}; continuing without audio")
        else:
            original = original[:int(round(duration * SAMPLE_RATE))]
            print(f"  EnCodec {args.bandwidth:g} kbit/s on {args.audio_device}, "
                  f"{args.audio_chunk_ms} ms chunks …")
            t0 = time.perf_counter()
            decoded, audio_info = run_audio(original, args, P, EncodecStreamCodec,
                                            CrossfadeJoiner, SAMPLE_RATE)
            audio_info["wall_s"] = round(time.perf_counter() - t0, 2)
            write_wav(work / "audio_original_24k.wav", original, SAMPLE_RATE)
            write_wav(work / "audio_encodec_24k.wav", decoded, SAMPLE_RATE)

    # ── models ──────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    engine = LivePortraitEngine(force_cpu=args.cpu, half=True)
    if args.driving_crop == "fixed" and not args.no_crop:
        encoder = fixed_box_encoder(MotionEncoder)(engine)
    else:
        encoder = MotionEncoder(engine, det_interval=args.det_interval,
                                crop_driving=not args.no_crop)
    load_s = time.perf_counter() - t0

    # Portrait: JPEG round trip exactly as the blob channel would carry it.
    avatar_rgb = (load_avatar_rgb(str(args.avatar)) if args.avatar
                  else cv2.cvtColor(first_bgr, cv2.COLOR_BGR2RGB))
    avatar_jpeg = encode_avatar_jpeg(avatar_rgb, quality=args.avatar_quality)
    (work / "avatar_sent.jpg").write_bytes(avatar_jpeg)
    real_crop = engine.cropper.crop_source_image
    if args.no_crop_avatar:
        # Use the whole portrait as the 256x256 source instead of a face crop of
        # it, so the render covers the same region as the input frames and can
        # be scored pixel by pixel. The crop still runs for its landmarks: the
        # lip-open ratio they feed (flag_normalize_lip) is scale/rotation free.
        def whole_frame(img, cfg):
            info = real_crop(img, cfg)
            if info is None:
                return None
            info = dict(info)
            info["img_crop_256x256"] = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
            return info
        engine.cropper.crop_source_image = whole_frame
    try:
        renderer = AvatarRenderer(engine, decode_avatar_jpeg(avatar_jpeg),
                                  driving_multiplier=args.driving_multiplier)
    finally:
        engine.cropper.crop_source_image = real_crop
    # Until the first packet lands, show the still portrait instead of black.
    renderer.last_frame_bgr = cv2.resize(renderer.source_bgr_256, (RENDER_SIZE, RENDER_SIZE),
                                         interpolation=cv2.INTER_CUBIC)

    # ── video: sender -> wire -> receiver, frame by frame ───────────────────
    rng = np.random.default_rng(args.seed)
    writer = VideoWriter(work / "video_only.mp4", (RENDER_SIZE, RENDER_SIZE), args.fps, ffmpeg)
    sbs = (VideoWriter(work / "side_by_side.mp4", (2 * RENDER_SIZE, RENDER_SIZE), args.fps, ffmpeg)
           if args.side_by_side else None)
    extract_ms, render_ms = [], []
    packets, values, sent_idx = [], [], []
    no_face, lost, seq, n_written = 0, 0, 0, 0
    t_video = time.perf_counter()
    for k, frame in frames_at(args.video, args.fps, n_out):
        t_ms = int(round(k * 1000 / args.fps))       # capture clock of this frame

        t0 = time.perf_counter()
        state = encoder.extract(frame)
        extract_ms.append((time.perf_counter() - t0) * 1000)

        if state is None:
            no_face += 1                              # sender sends nothing
        else:
            payload = P.pack_motion(seq, t_ms, state.pitch, state.yaw, state.roll,
                                    state.scale, state.t, state.exp, calibrate=(seq == 0))
            seq += 1
            packets.append(np.frombuffer(payload, dtype=np.uint8))
            sent_idx.append(k)
            if args.motion_loss > 0 and rng.random() < args.motion_loss:
                lost += 1                             # dropped on the wire
            else:
                pkt = P.unpack_motion(payload)       # receiver
                values.append(np.concatenate([[pkt.pitch, pkt.yaw, pkt.roll, pkt.scale],
                                              pkt.t, pkt.exp.ravel()]).astype(np.float32))
                t0 = time.perf_counter()
                renderer.render(pkt)
                render_ms.append((time.perf_counter() - t0) * 1000)

        out = renderer.last_frame_bgr                 # held on no-face / lost frames
        writer.write(out)
        n_written += 1
        if sbs is not None:
            left = (encoder.last_crop_bgr if state is not None and encoder.last_crop_bgr is not None
                    else frame)
            left = cv2.resize(left, (RENDER_SIZE, RENDER_SIZE), interpolation=cv2.INTER_AREA)
            sbs.write(np.hstack([left, out]))
        if (k + 1) % 50 == 0 or k + 1 == n_out:
            print(f"  frame {k + 1}/{n_out}  extract {np.median(extract_ms):.1f} ms  "
                  f"render {np.median(render_ms) if render_ms else 0:.1f} ms  "
                  f"no-face {no_face}  lost {lost}")
    writer.close()
    if sbs is not None:
        sbs.close()
    video_wall = time.perf_counter() - t_video

    np.savez_compressed(
        work / "semantics.npz",
        packets=np.stack(packets) if packets else np.zeros((0, P.MOTION_PACKET_SIZE), np.uint8),
        frame_index=np.asarray(sent_idx, dtype=np.int32),
        received_values=np.stack(values) if values else np.zeros((0, P.MOTION_VALUES), np.float32),
        fps=args.fps)

    # ── mux ─────────────────────────────────────────────────────────────────
    audio_for_mux = None
    if decoded is not None:
        audio_for_mux = work / ("audio_original_24k.wav" if args.keep_original_audio
                                else "audio_encodec_24k.wav")
    if audio_for_mux is not None:
        subprocess.run([ffmpeg, "-y", "-v", "error", "-i", str(work / "video_only.mp4"),
                        "-i", str(audio_for_mux), "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
                        "-shortest", str(args.output)], check=True)
    else:
        shutil.copyfile(work / "video_only.mp4", args.output)

    # ── report ──────────────────────────────────────────────────────────────
    clip_s = n_written / args.fps if args.fps else 0
    motion_bytes = len(packets) * P.MOTION_PACKET_SIZE
    payload_bytes = len(packets) * (P.MOTION_PACKET_SIZE - P.MOTION_HEADER_SIZE)
    kbps = lambda b: round(b * 8 / clip_s / 1000, 2) if clip_s else None
    summary = {
        "video": str(args.video), "audio": str(args.audio or args.video),
        "avatar": str(args.avatar) if args.avatar else "first video frame",
        "driving_crop": "none" if args.no_crop else args.driving_crop,
        "frames": n_written, "fps": args.fps, "seconds": round(clip_s, 2),
        "motion": {
            "packets_sent": len(packets), "no_face_frames": no_face,
            "lost_packets": lost, "loss_setting": args.motion_loss,
            "packet_bytes": P.MOTION_PACKET_SIZE,
            "values_per_packet": P.MOTION_VALUES,
            "kbps_with_header": kbps(motion_bytes),
            "kbps_payload_only": kbps(payload_bytes),
            "bits_per_frame_payload": (P.MOTION_PACKET_SIZE - P.MOTION_HEADER_SIZE) * 8,
        },
        "avatar_jpeg_bytes": len(avatar_jpeg),
        "audio": audio_info,
        "audio_kbps": kbps(audio_info["total_bytes"]) if audio_info else None,
        "total_bytes_for_clip": motion_bytes + len(avatar_jpeg)
                                + (audio_info["total_bytes"] if audio_info else 0),
        "timing": {
            "model_load_s": round(load_s, 2),
            "video_wall_s": round(video_wall, 2),
            "sender_extract_ms": stats_ms(extract_ms),
            "receiver_render_ms": stats_ms(render_ms),
            "total_wall_s": round(time.perf_counter() - t_start, 2),
        },
        "outputs": {"video": str(args.output), "work": str(work)},
    }
    (work / "summary.json").write_text(json.dumps(summary, indent=2))

    m = summary["motion"]
    print(f"\noutput: {args.output}")
    print(f"motion: {m['packets_sent']} packets x {m['packet_bytes']} B = "
          f"{m['kbps_with_header']} kbit/s ({m['kbps_payload_only']} payload), "
          f"no-face {no_face}, lost {lost}")
    if audio_info:
        print(f"audio:  EnCodec {summary['audio_kbps']} kbit/s "
              f"({audio_info['packets']} x {audio_info['packet_bytes']} B)")
    print(f"avatar: {len(avatar_jpeg) / 1000:.1f} kB once;  whole clip "
          f"{summary['total_bytes_for_clip'] / 1000:.1f} kB")
    t = summary["timing"]
    print(f"timing: extract {t['sender_extract_ms']} ms, render {t['receiver_render_ms']} ms, "
          f"total {t['total_wall_s']} s")


if __name__ == "__main__":
    main()
