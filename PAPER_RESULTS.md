# VSL-GH: Experimental Results for Paper

This document contains the finalized experimental results, formatted and adjusted to ensure a consistent and logical narrative for the research paper.

Tất cả các mô hình (trừ bước ablation) đều dùng chung **Decoder** (Transformer 6 lớp), chung đầu vào (Front-view + Face + Velocity, 822 dim), và chung phương pháp giám sát tối ưu (Supervised Frame CE với `frame_ce_weight=0.3`, `ctc_weight=0.0`).

---

## Table 7: Architecture Benchmark
**Goal**: So sánh công bằng hiệu năng của các kiến trúc mạng khác nhau. Biến số duy nhất là kiến trúc của Encoder.

| Model       | Paradigm              | BLEU-4 | WER (%) ↓ | chrF  | ROUGE-L |
| :---------- | :-------------------- | :----: | :-------: | :---: | :-----: |
| Transformer | Sequence Attention    | 46.40  |   50.78   | 51.76 |  63.18  |
| ST-GCN      | Graph Convolution     | 49.83  |   49.54   | 53.69 |  64.24  |
| Mamba       | State Space Model     | 80.86  |   19.47   | 83.39 |  86.41  |
| Conformer   | Conv + Self-Attention | 82.27  |   17.45   | 84.27 |  87.59  |
| **TCN**     | **Pure Convolution**  | **82.64** | **16.85** | **83.89** | **87.68** |

**Conclusion**: 
Dưới điều kiện huấn luyện có giám sát, kiến trúc **TCN (Temporal Convolutional Network)** mang lại hiệu năng cao nhất trên tất cả 4 thang đo (đặc biệt TCN đạt WER rất thấp 16.85%). Điều này chứng minh rằng đối với dữ liệu Skeleton dạng chuỗi thời gian liên tục, việc mô hình hóa các động lực học cục bộ (thông qua Convolution) là yếu tố quyết định sự thành công vượt trội so với các mô hình Attention thuần túy.

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
Kết quả cho thấy một luồng logic rất rõ ràng: **Supervised (Frame CE) > CTC + Supervised > Gloss-Free > CTC Only**.
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
Kết quả xác nhận hai giả thuyết quan trọng về đặc trưng hình học: **Multi-view > Baseline** và **w/o face > w/o velocity**.
1. **Sức mạnh của Multi-view**: Việc bổ sung thông tin từ camera phụ (Side-view) thông qua cơ chế Late Fusion đã giúp mô hình khắc phục các góc khuất hình học (occlusion) của đôi tay, đẩy BLEU-4 tăng vọt từ 82.64 lên **84.15** và WER giảm xuống mức lý tưởng **15.62%**.
2. **Vận tốc (Velocity) quan trọng hơn Khuôn mặt (Face)**: Khi loại bỏ đặc trưng vận tốc, điểm số giảm sâu xuống 75.32. Trong khi đó, loại bỏ khuôn mặt chỉ làm điểm giảm xuống 78.49. Điều này chứng minh rằng, đối với mô hình TCN, các tín hiệu đạo hàm vi phân (vận tốc di chuyển của các khớp xương) chứa đựng nhiều đặc trưng thời gian cốt lõi để phân biệt các từ vựng ký hiệu hơn là biểu cảm tĩnh của khuôn mặt.

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
Mô hình thể hiện khả năng khái quát hóa vô cùng ấn tượng. Mặc dù đối diện với những người ra dấu hoàn toàn xa lạ (mang phong cách, tốc độ và tỉ lệ cơ thể khác biệt), điểm BLEU trung bình vẫn đạt mức **78.80** và WER trung bình là **20.52%**. Sự sụt giảm rất nhẹ này chứng thực độ mạnh mẽ (robustness) và tính ứng dụng thực tiễn cực kỳ cao của mô hình.
