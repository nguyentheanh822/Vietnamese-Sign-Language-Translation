"""
data/dataset.py
---------------
PyTorch Dataset cho VSL-GH Sign Language Translation.
Doc keypoints .npy + gloss/translation tu dataset.json.
Ho tro data augmentation cho signer-independent training.

v2: skeleton-relative normalization + velocity features (411->822 dim)
"""

import json
import math
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


# ── Constants ────────────────────────────────────────────────────────────────
# Index cac nhom landmark trong vector 411-dim
# Pose:  0:75   (25 landmarks x 3)   -- 25 landmarks
# Face:  75:285 (70 landmarks x 3)   -- 70 landmarks
# LHand: 285:348 (21 landmarks x 3)  -- 21 landmarks
# RHand: 348:411 (21 landmarks x 3)  -- 21 landmarks
POSE_SLICE  = slice(0,   75)
FACE_SLICE  = slice(75,  285)
LHAND_SLICE = slice(285, 348)
RHAND_SLICE = slice(348, 411)

# MediaPipe Pose landmark indices for shoulder (trong 25-landmark subset)
# MediaPipe full body: left_shoulder=11, right_shoulder=12
# Trong file .npy cua du an nay, pose co 25 landmarks theo thu tu MediaPipe Pose
# left shoulder = landmark 11, right shoulder = landmark 12
LS_X = 11 * 3      # left shoulder x index trong pose slice
LS_Y = 11 * 3 + 1  # left shoulder y
RS_X = 12 * 3      # right shoulder x index
RS_Y = 12 * 3 + 1  # right shoulder y


class VSLDataset(Dataset):
    """
    Dataset cho VSL-GH SLT.

    Moi sample tra ve:
      keypoints : Tensor (T, 411)  float32
      gloss_ids : Tensor (G,)      long     — gloss token IDs
      trans_ids : Tensor (L,)      long     — translation token IDs (co BOS/EOS)
      gloss_len : int
      trans_len : int
      video_id  : str
    """

    def __init__(
        self,
        base_dir: str,
        split: str,                # "train" | "val" | "test"
        gloss_vocab,               # utils.vocab.Vocab
        trans_vocab,               # utils.vocab.Vocab
        max_seq_len: int = 300,
        max_gloss_len: int = 20,
        max_trans_len: int = 30,
        augment: bool = False,
        aug_flip_prob: float = 0.5,
        aug_speed: tuple = (0.8, 1.2),
        aug_noise_std: float = 0.01,
        aug_rotation_deg: float = 15.0,
        use_face: bool = True,
        use_velocity: bool = True,
        view_mode: str = "F", # F, S, FS
    ):
        super().__init__()
        base_dir = Path(base_dir)
        self.kp_dir    = base_dir / "data" / "keypoints"
        self.split     = split
        self.gloss_vocab = gloss_vocab
        self.trans_vocab = trans_vocab
        self.max_seq_len   = max_seq_len
        self.max_gloss_len = max_gloss_len
        self.max_trans_len = max_trans_len
        self.augment   = augment

        # Augmentation params
        self.aug_flip_prob     = aug_flip_prob
        self.aug_speed         = aug_speed
        self.aug_noise_std     = aug_noise_std
        self.aug_rotation_deg  = aug_rotation_deg
        
        self.use_face          = use_face
        self.use_velocity      = use_velocity
        self.view_mode         = view_mode

        # Load metadata
        with open(base_dir / "data" / "dataset.json", encoding="utf-8") as f:
            all_data = json.load(f)

        self.samples = [d for d in all_data if d["split"] == split]
        # Loc bo cac sample ma file npy khong ton tai
        self.samples = [
            d for d in self.samples
            if (self.kp_dir / f"{d['id']}.npy").exists()
        ]
        print(f"[VSLDataset] split={split}: {len(self.samples)} samples loaded")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry = self.samples[idx]
        video_id = entry["id"]

        # ── Load keypoints ──────────────────────────────────────────────────
        kp_f = np.load(str(self.kp_dir / f"{video_id}.npy"))  # (T, 411)
        
        if self.view_mode in ["S", "FS"]:
            video_id_s = video_id.replace("_F", "_S")
            s_path = self.kp_dir / f"{video_id_s}.npy"
            kp_s = np.load(str(s_path)) if s_path.exists() else kp_f.copy()
            
            min_len = min(len(kp_f), len(kp_s))
            kp_f = kp_f[:min_len]
            kp_s = kp_s[:min_len]

        def process_single_view(k):
            if not self.use_face:
                k = np.concatenate([k[:, :75], k[:, 285:]], axis=1) # (T, 201)
            return self._normalize_skeleton(k)

        if self.view_mode == "F":
            kp = process_single_view(kp_f)
        elif self.view_mode == "S":
            kp = process_single_view(kp_s)
        else:
            kf = process_single_view(kp_f)
            ks = process_single_view(kp_s)
            kp = np.concatenate([kf, ks], axis=-1)

        # Truncate to max_seq_len
        T = min(len(kp), self.max_seq_len)
        kp = kp[:T]

        # ── Augmentation (train only) ────────────────────────────────────────
        if self.augment and self.split == "train":
            kp = self._augment(kp)

        # Cap lai T sau augmentation
        T = min(len(kp), self.max_seq_len)
        kp = kp[:T]

        # ── Velocity features (motion dynamics) ──────────────────────────────
        if self.use_velocity:
            kp_vel = np.zeros_like(kp)
            if T > 1:
                kp_vel[1:] = kp[1:] - kp[:-1]
            kp_combined = np.concatenate([kp, kp_vel], axis=-1)
        else:
            kp_combined = kp
            
        kp_tensor = torch.from_numpy(kp_combined).float()

        # ── Gloss IDs ────────────────────────────────────────────────────────
        glosses = entry["gloss_sequence"][:self.max_gloss_len]
        gloss_ids = self.gloss_vocab.encode(glosses)
        gloss_tensor = torch.tensor(gloss_ids, dtype=torch.long)

        # ── Translation IDs ─────────────────────────────────────────────────
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent.parent))
        from source.utils.vocab import tokenize_vi
        trans_tokens = tokenize_vi(entry["translation"])
        trans_tokens = trans_tokens[:self.max_trans_len - 2]  # room for bos/eos
        trans_ids = (
            [self.trans_vocab.bos_idx]
            + self.trans_vocab.encode(trans_tokens)
            + [self.trans_vocab.eos_idx]
        )
        trans_tensor = torch.tensor(trans_ids, dtype=torch.long)

        # ── Frame-level Gloss IDs ──────────────────────────────────────────────
        frame_gloss_ids = torch.full((T,), self.gloss_vocab.blank_idx, dtype=torch.long)
        boundaries = entry.get("gloss_frame_boundaries", [])
        for b in boundaries:
            start_f = min(b["start"], T)
            end_f = min(b["end"], T)
            if start_f < end_f:
                g_id = self.gloss_vocab.token2idx.get(b["gloss"], self.gloss_vocab.unk_idx)
                if g_id != self.gloss_vocab.unk_idx:
                    frame_gloss_ids[start_f:end_f] = g_id

        return {
            "keypoints":  kp_tensor,          # (T, 822/411)
            "gloss_ids":  gloss_tensor,        # (G,)
            "frame_gloss_ids": frame_gloss_ids, # (T,)
            "trans_ids":  trans_tensor,        # (L,)
            "gloss_len":  len(gloss_ids),
            "trans_len":  len(trans_ids),
            "seq_len":    T,
            "video_id":   video_id,
        }

    # ── Skeleton normalization ───────────────────────────────────────────────
    def _normalize_skeleton(self, kp: np.ndarray) -> np.ndarray:
        """
        Skeleton-relative normalization cho signer independence:
        1. Translate: lay midpoint 2 vai lam goc toa do (0,0)
        2. Scale: chia cho khoang cach 2 vai (shoulder width)
        3. Zero keypoints (undetected) van giu la 0 sau normalize
        """
        kp = kp.copy().astype(np.float32)  # (T, 411)
        T  = len(kp)

        # Lay toa do xy cua left/right shoulder tu pose slice
        ls_x = kp[:, LS_X]          # (T,) left shoulder x
        ls_y = kp[:, LS_Y]          # (T,) left shoulder y
        rs_x = kp[:, RS_X]          # (T,) right shoulder x
        rs_y = kp[:, RS_Y]          # (T,) right shoulder y

        # Body center: midpoint of 2 shoulders (T, 1)
        cx = ((ls_x + rs_x) / 2).reshape(T, 1)  # (T, 1)
        cy = ((ls_y + rs_y) / 2).reshape(T, 1)  # (T, 1)

        # Shoulder width as scale (T, 1)
        scale = (np.sqrt((rs_x - ls_x) ** 2 + (rs_y - ls_y) ** 2) + 1e-6).reshape(T, 1)

        # Mask: vi tri == 0 la chua detect duoc
        detected_x = (kp[:, 0::3] != 0)  # (T, 137)
        detected_y = (kp[:, 1::3] != 0)  # (T, 137)

        # Translate x, y; scale z
        kp[:, 0::3] = np.where(detected_x, (kp[:, 0::3] - cx) / scale, 0.0)
        kp[:, 1::3] = np.where(detected_y, (kp[:, 1::3] - cy) / scale, 0.0)
        kp[:, 2::3] = kp[:, 2::3] / scale  # z: scale only

        return kp

    # ── Augmentation helpers ────────────────────────────────────────────────
    def _augment(self, kp: np.ndarray) -> np.ndarray:
        """Ap dung cac phep bien doi ngau nhien cho signer-independence."""
        # 1. Horizontal flip (trao doi tay trai/phai)
        if random.random() < self.aug_flip_prob:
            kp = self._horizontal_flip(kp)

        # 2. Temporal jitter (thay doi toc do ky hieu)
        speed = random.uniform(*self.aug_speed)
        kp = self._temporal_scale(kp, speed)

        # 3. Gaussian noise
        if self.aug_noise_std > 0:
            noise = np.random.normal(0, self.aug_noise_std, kp.shape).astype(np.float32)
            kp = kp + noise

        # 4. Random rotation (tren toa do x,y)
        if self.aug_rotation_deg > 0:
            deg = random.uniform(-self.aug_rotation_deg, self.aug_rotation_deg)
            kp = self._rotate_xy(kp, deg)

        # 5. Random body scale (simulate khac biet tam voc)
        if hasattr(self, 'aug_scale') and self.aug_scale:
            scale_factor = random.uniform(self.aug_scale[0], self.aug_scale[1])
            kp[:, 0::3] *= scale_factor  # x
            kp[:, 1::3] *= scale_factor  # y

        return kp

    def _horizontal_flip(self, kp: np.ndarray) -> np.ndarray:
        kp = kp.copy()
        # Flip x coordinate (x = 1 - x) for pose + face
        # Flip va trao doi left/right hand
        kp[:, POSE_SLICE][..., ::3]  = 1.0 - kp[:, POSE_SLICE][..., ::3]   # x of pose
        kp[:, FACE_SLICE][..., ::3]  = 1.0 - kp[:, FACE_SLICE][..., ::3]   # x of face
        # Swap left/right hand
        lhand = kp[:, LHAND_SLICE].copy()
        rhand = kp[:, RHAND_SLICE].copy()
        lhand[:, ::3] = 1.0 - lhand[:, ::3]
        rhand[:, ::3] = 1.0 - rhand[:, ::3]
        kp[:, LHAND_SLICE] = rhand
        kp[:, RHAND_SLICE] = lhand
        return kp

    def _temporal_scale(self, kp: np.ndarray, speed: float) -> np.ndarray:
        T = len(kp)
        new_T = max(2, int(T / speed))
        indices = np.linspace(0, T - 1, new_T).astype(int)
        return kp[indices]

    def _rotate_xy(self, kp: np.ndarray, deg: float) -> np.ndarray:
        kp = kp.copy()
        rad = math.radians(deg)
        cos_r, sin_r = math.cos(rad), math.sin(rad)
        # Rotate tat ca x,y quanh tam (0.5, 0.5)
        cx, cy = 0.5, 0.5
        # Quay x (index ::3) va y (index 1::3)
        x = kp[:, ::3]  - cx
        y = kp[:, 1::3] - cy
        kp[:, ::3]  = x * cos_r - y * sin_r + cx
        kp[:, 1::3] = x * sin_r + y * cos_r + cy
        return kp


def collate_fn(batch):
    """
    Padding batch ve cung chieu dai.
    keypoints: (B, T_max, 411)
    gloss_ids: (B, G_max)
    trans_ids: (B, L_max)
    """
    B = len(batch)
    max_T = max(b["seq_len"]  for b in batch)
    max_G = max(b["gloss_len"] for b in batch)
    max_L = max(b["trans_len"] for b in batch)
    D = batch[0]["keypoints"].shape[-1]  # 411

    keypoints  = torch.zeros(B, max_T, D)
    gloss_ids  = torch.zeros(B, max_G, dtype=torch.long)
    frame_gloss_ids = torch.zeros(B, max_T, dtype=torch.long)
    trans_ids  = torch.zeros(B, max_L, dtype=torch.long)
    kp_mask    = torch.ones(B, max_T,  dtype=torch.bool)   # True = padding
    gloss_mask = torch.ones(B, max_G,  dtype=torch.bool)
    trans_mask = torch.ones(B, max_L,  dtype=torch.bool)

    seq_lens   = torch.zeros(B, dtype=torch.long)
    gloss_lens = torch.zeros(B, dtype=torch.long)
    trans_lens = torch.zeros(B, dtype=torch.long)
    video_ids  = []

    for i, b in enumerate(batch):
        T = b["seq_len"]
        G = b["gloss_len"]
        L = b["trans_len"]

        keypoints[i, :T]  = b["keypoints"]
        gloss_ids[i, :G]  = b["gloss_ids"]
        frame_gloss_ids[i, :T] = b["frame_gloss_ids"]
        trans_ids[i, :L]  = b["trans_ids"]

        kp_mask[i, :T]    = False
        gloss_mask[i, :G] = False
        trans_mask[i, :L] = False

        seq_lens[i]   = T
        gloss_lens[i] = G
        trans_lens[i] = L
        video_ids.append(b["video_id"])

    return {
        "keypoints":   keypoints,    # (B, T, D)
        "kp_mask":     kp_mask,      # (B, T) True=pad
        "gloss_ids":   gloss_ids,    # (B, G)
        "frame_gloss_ids": frame_gloss_ids, # (B, T)
        "gloss_mask":  gloss_mask,   # (B, G)
        "trans_ids":   trans_ids,    # (B, L)
        "trans_mask":  trans_mask,   # (B, L)
        "seq_lens":    seq_lens,     # (B,)
        "gloss_lens":  gloss_lens,   # (B,)
        "trans_lens":  trans_lens,   # (B,)
        "video_ids":   video_ids,
    }
