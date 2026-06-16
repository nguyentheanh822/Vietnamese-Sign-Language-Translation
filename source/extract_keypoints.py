"""
extract_keypoints.py
--------------------
Trich xuat MediaPipe Holistic keypoints tu 2100 videos.
Su dung multiprocessing de chay song song (nhanh hon ~4-8x).

Cau hinh landmarks (tuong duong OpenPose):
  - Pose:       25 landmarks (body, khong lay chan)
  - Face:       70 landmarks (mien mat quan trong)
  - Left hand:  21 landmarks
  - Right hand: 21 landmarks
  Tong: 137 landmarks x 3 coords (x,y,z) = 411 features/frame

Output:
  data/keypoints/SENT001_S01_R01_F.npy  shape: (T, 411)

Usage:
  python source/extract_keypoints.py
  python source/extract_keypoints.py --num_workers 8
  python source/extract_keypoints.py --resume  # bo qua file da co
"""

import argparse
import os
import json
import time
import traceback
import multiprocessing as mp
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mediapipe_lib
import mediapipe.python.solutions as mp_solutions

# ---- Landmark selection ---------------------------------------------------
# POSE: 25 landmarks (tuong duong OpenPose BODY_25, bo phan chan duoi)
POSE_LANDMARKS = [
    0,   # nose
    1, 2, 3, 4, 5, 6, 7, 8,   # eyes, ears
    11, 12,                    # shoulders
    13, 14,                    # elbows
    15, 16,                    # wrists
    17, 18, 19, 20,            # hand tips (pinky, index, thumb outer)
    23, 24,                    # hips
    25, 26,                    # knees
    27, 28,                    # ankles
]  # 25 landmarks

# FACE: 70 landmarks (lip, eyes, nose, face contour)
# Lay tu MediaPipe face mesh - cac diem quan trong nhat
FACE_LANDMARKS = [
    # Lips (20)
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    291, 308, 324, 318, 402, 317, 14, 87, 178, 88,
    # Left eye (6)
    33, 7, 163, 144, 145, 153,
    # Right eye (6)
    362, 382, 381, 380, 374, 373,
    # Eyebrows left (5)
    46, 53, 52, 65, 55,
    # Eyebrows right (5)
    285, 295, 282, 283, 276,
    # Nose (9)
    168, 6, 197, 195, 5, 4, 1, 19, 94,
    # Face contour (19)
    10, 338, 297, 332, 284, 251, 389, 356, 454,
    323, 361, 288, 397, 365, 379, 378, 400, 377, 152,
]  # 70 landmarks

# HANDS: 21 + 21 landmarks (full)
# MediaPipe da co 21 landmark cho moi ban tay

POSE_DIM  = len(POSE_LANDMARKS) * 3   # 25 * 3 = 75
FACE_DIM  = len(FACE_LANDMARKS) * 3   # 70 * 3 = 210
HAND_DIM  = 21 * 3                    # 21 * 3 = 63
TOTAL_DIM = POSE_DIM + FACE_DIM + HAND_DIM * 2  # 75+210+63+63 = 411

print(f"Landmark config: Pose={len(POSE_LANDMARKS)}, Face={len(FACE_LANDMARKS)}, "
      f"Hands=21+21, Total={POSE_DIM+FACE_DIM+HAND_DIM*2}dims/frame")

# ---- Paths ----------------------------------------------------------------
BASE_DIR    = Path("/workspace/yhnn/VSL_GH")
VIDEO_DIR   = BASE_DIR / "data" / "videos"
KP_DIR      = BASE_DIR / "data" / "keypoints"
DATASET_JSON = BASE_DIR / "data" / "dataset.json"

KP_DIR.mkdir(exist_ok=True)

# ---- Helper: xu ly 1 video ------------------------------------------------
def extract_one(args):
    """Trich xuat keypoints tu 1 video, luu .npy, tra ve (stem, ok, msg)"""
    video_path, kp_path, resume = args

    stem = Path(video_path).stem

    if resume and Path(kp_path).exists():
        return (stem, True, "skip")

    try:
        mp_holistic = mp_solutions.holistic

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return (stem, False, f"Cannot open video: {video_path}")

        frames_kp = []

        with mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as holistic:

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Resize xuong 360x360 de xu ly nhanh hon (ty le 1:1 giu nguyen)
                frame_rgb = cv2.cvtColor(
                    cv2.resize(frame, (360, 360)),
                    cv2.COLOR_BGR2RGB
                )
                results = holistic.process(frame_rgb)

                frame_vec = np.zeros(TOTAL_DIM, dtype=np.float32)
                idx = 0

                # --- POSE ---
                if results.pose_landmarks:
                    lm = results.pose_landmarks.landmark
                    for li in POSE_LANDMARKS:
                        frame_vec[idx]     = lm[li].x
                        frame_vec[idx + 1] = lm[li].y
                        frame_vec[idx + 2] = lm[li].z
                        idx += 3
                else:
                    idx += POSE_DIM

                # --- FACE ---
                if results.face_landmarks:
                    lm = results.face_landmarks.landmark
                    for li in FACE_LANDMARKS:
                        frame_vec[idx]     = lm[li].x
                        frame_vec[idx + 1] = lm[li].y
                        frame_vec[idx + 2] = lm[li].z
                        idx += 3
                else:
                    idx += FACE_DIM

                # --- LEFT HAND ---
                if results.left_hand_landmarks:
                    lm = results.left_hand_landmarks.landmark
                    for li in range(21):
                        frame_vec[idx]     = lm[li].x
                        frame_vec[idx + 1] = lm[li].y
                        frame_vec[idx + 2] = lm[li].z
                        idx += 3
                else:
                    idx += HAND_DIM

                # --- RIGHT HAND ---
                if results.right_hand_landmarks:
                    lm = results.right_hand_landmarks.landmark
                    for li in range(21):
                        frame_vec[idx]     = lm[li].x
                        frame_vec[idx + 1] = lm[li].y
                        frame_vec[idx + 2] = lm[li].z
                        idx += 3
                else:
                    idx += HAND_DIM

                frames_kp.append(frame_vec)

        cap.release()

        if len(frames_kp) == 0:
            return (stem, False, "No frames extracted")

        keypoints = np.array(frames_kp, dtype=np.float32)  # (T, 411)
        np.save(str(kp_path), keypoints)
        return (stem, True, f"{keypoints.shape}")

    except Exception as e:
        return (stem, False, f"ERROR: {traceback.format_exc()[-200:]}")

# ---- Main -----------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=8,
                        help="So luong worker process song song")
    parser.add_argument("--resume", action="store_true",
                        help="Bo qua file .npy da ton tai")
    args = parser.parse_args()

    # Load danh sach video
    video_files = sorted(VIDEO_DIR.glob("*.mp4"))
    print(f"Tong so videos: {len(video_files)}")
    print(f"Workers: {args.num_workers}")
    print(f"Resume mode: {args.resume}")
    print(f"Output dim: {TOTAL_DIM} features/frame")
    print("=" * 60)

    tasks = [
        (str(vf), str(KP_DIR / (vf.stem + ".npy")), args.resume)
        for vf in video_files
    ]

    # Chay song song voi multiprocessing Pool
    t0 = time.time()
    done = 0
    skipped = 0
    failed = []

    with mp.Pool(processes=args.num_workers) as pool:
        for stem, ok, msg in pool.imap_unordered(extract_one, tasks):
            done += 1
            if msg == "skip":
                skipped += 1
            elif ok:
                pass
            else:
                failed.append((stem, msg))

            # In tien trinh moi 50 videos
            if done % 50 == 0 or done == len(tasks):
                elapsed = time.time() - t0
                speed = done / elapsed
                eta = (len(tasks) - done) / speed if speed > 0 else 0
                print(f"  [{done:4d}/{len(tasks)}] "
                      f"Done={done-len(failed)-skipped}, "
                      f"Skip={skipped}, Fail={len(failed)} | "
                      f"Speed={speed:.1f} vid/s | ETA={eta:.0f}s")

    elapsed = time.time() - t0

    print("\n" + "=" * 60)
    print("KET QUA TRICH XUAT")
    print("=" * 60)
    print(f"  Tong xu ly       : {done}")
    print(f"  Thanh cong       : {done - len(failed) - skipped}")
    print(f"  Da co (skip)     : {skipped}")
    print(f"  That bai         : {len(failed)}")
    print(f"  Thoi gian        : {elapsed:.1f}s ({elapsed/60:.1f} phut)")

    if failed:
        print("\n  Cac video that bai:")
        for stem, msg in failed[:10]:
            print(f"    {stem}: {msg}")

    # Xac nhan files da tao
    npy_files = list(KP_DIR.glob("*.npy"))
    print(f"\n  Files .npy da tao: {len(npy_files)}")
    if npy_files:
        sample = np.load(str(npy_files[0]))
        print(f"  Sample shape     : {sample.shape}  (frames x {TOTAL_DIM})")

    print("\n[DONE]")

if __name__ == "__main__":
    main()
