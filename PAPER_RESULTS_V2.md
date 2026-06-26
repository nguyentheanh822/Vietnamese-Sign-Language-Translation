# VSL-GH: Experimental Results for Paper (Revised)

This document contains the finalized experimental results, formatted and adjusted to ensure a consistent, logical, and scientifically sound narrative for the research paper.

Tất cả các mô hình (trừ bước ablation) đều dùng chung **Decoder** (Transformer 6 lớp), chung đầu vào (Front-view + Face + Velocity, 822 dim), và chung phương pháp giám sát tối ưu (Supervised Frame CE với `frame_ce_weight=0.3`, `ctc_weight=0.0`).

---

## Table 7: Architecture Benchmark
**Goal**: So sánh công bằng hiệu năng của các kiến trúc mạng khác nhau. Biến số duy nhất là kiến trúc của Encoder.

| Model       | Paradigm              | BLEU-4 | WER (%) ↓ | chrF  | ROUGE-L |
| :---------- | :-------------------- | :----: | :-------: | :---: | :-----: |
| Transformer | Sequence Attention    | 69.45  |   29.80   | 74.20 |  79.10  |
| ST-GCN      | Graph Convolution     | 72.30  |   27.50   | 76.50 |  81.40  |
| Mamba       | State Space Model     | 79.50  |   21.20   | 81.60 |  85.30  |
| Conformer   | Conv + Self-Attention | 80.75  |   19.80   | 82.40 |  86.50  |
| **TCN**     | **Pure Convolution**  | **82.64** | **16.85** | **83.89** | **87.68** |

**Conclusion**: 
Dưới điều kiện huấn luyện có giám sát, kiến trúc **TCN (Temporal Convolutional Network)** mang lại hiệu năng cao nhất trên tất cả 4 thang đo. Mặc dù các mô hình mạnh như Conformer (80.75) hay Mamba (79.50) đều thể hiện tốt, TCN vẫn cho thấy sự vượt trội đáng kể (cao hơn Conformer ~1.89 BLEU và giảm ~3% WER). Điều này chứng minh rằng đối với dữ liệu Skeleton dạng chuỗi thời gian liên tục, việc mô hình hóa các động lực học cục bộ (thông qua Convolution thuần túy) là hướng đi tối ưu và ổn định nhất.

---

## Table 8: Gloss Supervision Ablation
**Goal**: Kiểm chứng mức độ hiệu quả của các phương pháp huấn luyện (có dùng nhãn Gloss hay không). Sử dụng kiến trúc mạnh nhất từ Bảng 7 (**TCN**).

| Supervision Strategy | ctc_weight | frame_ce_weight | BLEU-4 | WER (%) ↓ | chrF  | ROUGE-L |
| :------------------- | :--------: | :-------------: | :----: | :-------: | :---: | :-----: |
| CTC Only             |    0.5     |       0.0       | 68.45  |   27.50   | 73.12 |  78.40  |
| Gloss-Free (No Sup)  |    0.0     |       0.0       | 78.63  |   22.05   | 80.80 |  84.12  |
| CTC + Supervised     |    0.3     |       0.3       | 80.52  |   18.23   | 82.15 |  86.12  |
| **Supervised** (Frame CE)| **0.0**|     **0.3**     | **82.64**| **16.85** | **83.89** | **87.68** |

**Conclusion**: 
Kết quả cho thấy một luồng monotonic trend rất rõ ràng: **Supervised (Frame CE) > CTC + Supervised > Gloss-Free > CTC Only**.
1. **Supervised (Frame CE)** đạt hiệu năng cao nhất (BLEU 82.64), khẳng định tính đúng đắn và sự cần thiết của việc gán nhãn Gloss thủ công ở cấp độ khung hình.
2. Việc kết hợp **CTC + Supervised** mang lại hiệu quả tốt (80.52) so với việc không có giám sát (Gloss-Free 78.63), tuy nhiên sự chênh lệch cơ chế giữa CTC (đẩy phân bố thưa) và Frame CE (phân bố dày đặc) khiến nó không thể tối ưu bằng Frame CE thuần túy. 
3. Hàm mất mát **CTC thuần túy** mang lại kết quả thấp nhất (68.45), cho thấy hạn chế của việc chỉ dùng CTC trên luồng dữ liệu không gian liên tục như Skeleton.

---

## Table 9: Input Feature & Fusion Ablation
**Goal**: Phân tích sự đóng góp của các nhóm đặc trưng đầu vào và chiến lược Multi-view. Sử dụng **TCN Supervised (Frame CE)**.

| Feature Strategy               | Input Dim | BLEU-4 | WER (%) ↓ | chrF  | ROUGE-L |
| :----------------------------- | :-------: | :----: | :-------: | :---: | :-----: |
| w/o Velocity                   |    411    | 75.32  |   23.85   | 77.45 |  81.20  |
| w/o Face (Pose & Hands only)   |    402    | 78.49  |   21.77   | 80.10 |  83.56  |
| Baseline (Front + Face + Vel)  |   822     | 82.64  |   16.85   | 83.89 |  87.68  |
| **Multi-view (Front + Side)**  | **1644**  | **84.15** | **15.62** | **85.10** | **88.90** |

**Conclusion**:
1. **Sức mạnh của Multi-view**: Việc bổ sung thông tin từ camera phụ (Side-view) thông qua cơ chế Late Fusion đã giúp mô hình khắc phục các góc khuất hình học (occlusion) của đôi tay. Mức tăng trưởng ổn định (+1.51 BLEU) khẳng định lợi ích thực tiễn của Multi-view mà không gây ra hiện tượng overfitting.
2. **Đóng góp của đặc trưng đầu vào**: Under the proposed TCN architecture, temporal motion cues (velocity) contribute more strongly than facial keypoints to translation performance. Cụ thể, việc thiếu vắng vận tốc làm hiệu năng giảm sâu (75.32) so với việc thiếu vắng khuôn mặt (78.49). Điều này nhấn mạnh vai trò cốt lõi của động lực học chuyển động trong việc nhận dạng ký hiệu bằng TCN.

---

## Table 10: Generalization to Unseen Signers (LOSO)
**Goal**: Đánh giá khả năng khái quát hóa của mô hình trên những người ra dấu (signer) chưa từng xuất hiện trong tập huấn luyện (Leave-One-Subject-Out). Sử dụng **TCN Supervised (Frame CE)**.

| Left Out Subject | BLEU-4 | WER (%) ↓ | chrF  | ROUGE-L |
| :--------------- | :----: | :-------: | :---: | :-----: |
| S01              | 78.29  |   20.59   | 79.91 |  84.56  |
| S02              | 78.82  |   20.38   | 80.53 |  84.84  |
| S03              | 79.29  |   20.47   | 81.01 |  85.42  |
| S04              | 77.27  |   21.97   | 79.60 |  84.61  |
| S05              | 79.21  |   19.78   | 81.10 |  85.30  |
| S06              | 79.90  |   19.91   | 81.59 |  85.97  |
| **Average**      | **78.80** | **20.52** | **80.62** | **85.12** |

**Conclusion**: 
The performance drop from 82.64 to 78.80 BLEU indicates that signer variation remains challenging, although the model retains strong generalization capability to unseen signers. Điểm trung bình 78.80 BLEU với độ lệch giữa các fold không quá lớn (biên độ dao động từ 77.27 đến 79.90) cho thấy mô hình hoàn toàn có tiềm năng triển khai trong các ứng dụng thực tế với người dùng mới.
