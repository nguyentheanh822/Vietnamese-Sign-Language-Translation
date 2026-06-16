import gradio as gr
import torch
import numpy as np
import cv2
import yaml
import sys
import os
from pathlib import Path
import mediapipe as mediapipe_lib

sys.path.insert(0, str(Path(__file__).parent / "source"))

# Fix PermissionError in /tmp/gradio
BASE_DIR = Path("/workspace/yhnn/VSL_GH")
os.environ["GRADIO_TEMP_DIR"] = str(BASE_DIR / "temp")

from models.slt_model import SLTModel
from utils.vocab import Vocab

# --- 1. CONFIG & MODEL LOADING ---
BASE_DIR = Path("/workspace/yhnn/VSL_GH")
CONFIG_PATH = BASE_DIR / "source/config/config_phase2.yaml"
CHECKPOINT_PATH = BASE_DIR / "results/run_phase2/checkpoint_best.pt"
TRANS_VOCAB_PATH = BASE_DIR / "data/trans_vocab.txt"
GLOSS_VOCAB_PATH = BASE_DIR / "data/gloss_vocab.txt"
DATASET_JSON_PATH = BASE_DIR / "data/dataset.json"

with open(CONFIG_PATH, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trans_vocab = Vocab.load(str(TRANS_VOCAB_PATH))
gloss_vocab = Vocab.load(str(GLOSS_VOCAB_PATH))

# Load database mapping for References
import json, re
try:
    with open(DATASET_JSON_PATH, encoding="utf-8") as f:
        dataset_meta = json.load(f)
    # Build {"SENT001": "câu dịch"} - one entry per unique sentence
    ref_dict = {}
    for item in dataset_meta:
        sid = item.get("sentence_id") or item.get("id", "").split("_")[0]
        trans = item.get("translation", "")
        if sid and trans and sid not in ref_dict:
            ref_dict[sid] = trans
    print(f"[REF] Loaded {len(ref_dict)} reference sentences.")
except Exception as e:
    print(f"[REF] ERROR loading dataset: {e}")
    ref_dict = {}

m_cfg = cfg["model"]
model = SLTModel(
    input_dim=cfg["data"]["input_dim"], # usually 822
    gloss_vocab_size=len(gloss_vocab),
    trans_vocab_size=len(trans_vocab),
    d_model=m_cfg["d_model"],
    nhead=m_cfg["nhead"],
    num_encoder_layers=m_cfg["num_encoder_layers"],
    num_decoder_layers=m_cfg["num_decoder_layers"],
    dim_feedforward=m_cfg["dim_feedforward"],
    dropout=m_cfg["dropout"],
    pad_idx=trans_vocab.pad_idx,
).to(device)

print("Đang tải trọng số mô hình...")
ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
model_state = ckpt.get("model_state", ckpt)
model.load_state_dict(model_state, strict=False)
model.eval()
print("Sẵn sàng!")

# --- 2. EXTRACTION LOGIC ---
POSE_LANDMARKS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 23, 24, 25, 26, 27, 28]
FACE_LANDMARKS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 33, 7, 163, 144, 145, 153, 362, 382, 381, 380, 374, 373, 46, 53, 52, 65, 55, 285, 295, 282, 283, 276, 168, 6, 197, 195, 5, 4, 1, 19, 94, 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152]
POSE_DIM, FACE_DIM, HAND_DIM = 75, 210, 63
TOTAL_DIM = 411
LS_X, LS_Y = 11*3, 11*3+1
RS_X, RS_Y = 12*3, 12*3+1

def extract_video(video_path):
    mp_holistic = mediapipe_lib.solutions.holistic
    cap = cv2.VideoCapture(str(video_path))
    frames_kp = []
    with mp_holistic.Holistic(static_image_mode=False, model_complexity=1, smooth_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_rgb = cv2.cvtColor(cv2.resize(frame, (360, 360)), cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)
            frame_vec = np.zeros(TOTAL_DIM, dtype=np.float32)
            idx = 0
            if results.pose_landmarks:
                for li in POSE_LANDMARKS:
                    frame_vec[idx], frame_vec[idx+1], frame_vec[idx+2] = results.pose_landmarks.landmark[li].x, results.pose_landmarks.landmark[li].y, results.pose_landmarks.landmark[li].z
                    idx += 3
            else: idx += POSE_DIM
            if results.face_landmarks:
                for li in FACE_LANDMARKS:
                    frame_vec[idx], frame_vec[idx+1], frame_vec[idx+2] = results.face_landmarks.landmark[li].x, results.face_landmarks.landmark[li].y, results.face_landmarks.landmark[li].z
                    idx += 3
            else: idx += FACE_DIM
            if results.left_hand_landmarks:
                for li in range(21):
                    frame_vec[idx], frame_vec[idx+1], frame_vec[idx+2] = results.left_hand_landmarks.landmark[li].x, results.left_hand_landmarks.landmark[li].y, results.left_hand_landmarks.landmark[li].z
                    idx += 3
            else: idx += HAND_DIM
            if results.right_hand_landmarks:
                for li in range(21):
                    frame_vec[idx], frame_vec[idx+1], frame_vec[idx+2] = results.right_hand_landmarks.landmark[li].x, results.right_hand_landmarks.landmark[li].y, results.right_hand_landmarks.landmark[li].z
                    idx += 3
            frames_kp.append(frame_vec)
    cap.release()
    return np.array(frames_kp, dtype=np.float32)

def normalize_skeleton(kp):
    kp = kp.copy()
    T = len(kp)
    ls_x, ls_y = kp[:, LS_X], kp[:, LS_Y]
    rs_x, rs_y = kp[:, RS_X], kp[:, RS_Y]
    cx, cy = ((ls_x + rs_x)/2).reshape(T, 1), ((ls_y + rs_y)/2).reshape(T, 1)
    scale = (np.sqrt((rs_x - ls_x)**2 + (rs_y - ls_y)**2) + 1e-6).reshape(T, 1)
    det_x, det_y = (kp[:, 0::3] != 0), (kp[:, 1::3] != 0)
    kp[:, 0::3] = np.where(det_x, (kp[:, 0::3] - cx)/scale, 0.0)
    kp[:, 1::3] = np.where(det_y, (kp[:, 1::3] - cy)/scale, 0.0)
    kp[:, 2::3] /= scale
    return kp

# --- 3. INFERENCE ---
def show_reference(video_file):
    """Instant lookup - called as soon as video is selected."""
    if not video_file:
        return ""
    # Search full path case-insensitively for SENT + digits
    match = re.search(r'SENT(\d{3})', str(video_file), re.IGNORECASE)
    if not match:
        print(f"[DEBUG] video_file path: {video_file}")
        return "Không tìm thấy mã SENT trong tên file."
    key = "SENT" + match.group(1).zfill(3)
    ref = ref_dict.get(key, "")
    print(f"[DEBUG] key={key}, ref={repr(ref)}")
    return ref.capitalize() if ref else f"Không có Reference cho {key}."

def predict_sign_language(video_file):
    """Run inference and return translated text only."""
    if not video_file: 
        return "Vui lòng tải lên video."
    try:
        kp = extract_video(video_file)
        if len(kp) == 0: 
            return "❌ Lỗi: Không tìm thấy người trong video hoặc video lỗi.", ref_display
        
        # Max Seq Len
        T = min(len(kp), cfg["data"]["max_seq_len"])
        kp = kp[:T]
        
        # Normalize
        kp = normalize_skeleton(kp)
        
        # Velocity
        kp_vel = np.zeros_like(kp)
        if T > 1: kp_vel[1:] = kp[1:] - kp[:-1]
        kp_combined = np.concatenate([kp, kp_vel], axis=-1)
        
        src = torch.from_numpy(kp_combined).float().unsqueeze(0).to(device) # (1, T, 822)
        src_mask = torch.zeros(1, T, dtype=torch.bool).to(device)
        
        with torch.no_grad():
            preds = model.greedy_decode(
                src, src_mask,
                bos_idx=trans_vocab.bos_idx,
                eos_idx=trans_vocab.eos_idx,
                max_len=cfg["evaluation"]["max_decode_len"]
            )
            
        hyp = trans_vocab.decode(preds[0].tolist())
        final_text = " ".join(hyp).capitalize()
        return final_text
    except Exception as e:
        return f"❌ Đã xảy ra lỗi: {str(e)}"

# --- 4. GRADIO APP ---
with gr.Blocks(theme=gr.themes.Default(primary_hue="blue", secondary_hue="indigo")) as demo:
    gr.Markdown("# 🏥 Hệ thống Dịch Ngôn Ngữ Ký Hiệu Y Tế (VSL-GH)")
    gr.Markdown("Ứng dụng chạy trên kiến trúc Skeleton Transformer, dịch ngôn ngữ Ký hiệu trong ngữ cảnh khám sức khỏe tổng quát sang Văn bản tiếng Việt.")
    
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Tải lên và Xem trước Video MP4", sources=["upload"])
            with gr.Row():
                submit_btn = gr.Button("Bắt đầu dịch", variant="primary")
                clear_btn = gr.Button("Xóa / Làm mới", variant="secondary")
                
        with gr.Column():
            text_output = gr.Textbox(label="Kết quả dịch ", lines=3, placeholder="Kết quả sẽ hiện ra ở đây...", interactive=False)
            ref_output = gr.Textbox(label="Câu gốc tham chiếu (Dữ liệu Ground Truth)", lines=2, interactive=False)
            
    submit_btn.click(fn=predict_sign_language, inputs=video_input, outputs=text_output)
    video_input.change(fn=show_reference, inputs=video_input, outputs=ref_output)
    
    def clear_all():
        return None, "", ""
    clear_btn.click(fn=clear_all, inputs=[], outputs=[video_input, text_output, ref_output])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
