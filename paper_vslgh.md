# VSL-GH: A Sentence-Level Vietnamese Sign Language Dataset for Health Communication

> **Paper writing file — VSL-GH Dataset Paper**
> Last updated: 2026-04-17

---

## ✅ ABSTRACT

Việc thiếu hụt dữ liệu ngôn ngữ ký hiệu tiếng Việt ở mức câu đã khiến các bài toán dịch và sinh ngôn ngữ ký hiệu liên tục tại Việt Nam chưa phát triển mạnh mẽ; hiện nay chưa tồn tại bất kỳ bộ dữ liệu VSL nào cung cấp câu hoàn chỉnh kèm annotation chi tiết, đặc biệt trong domain y tế. Để lấp đầy khoảng trống này, chúng tôi xây dựng VSL-GH – bộ dữ liệu ngôn ngữ ký hiệu tiếng Việt đầu tiên ở mức câu trong ngữ cảnh khám sức khỏe tổng quát (biến thể TP. Hồ Chí Minh), với dữ liệu video đa góc nhìn và annotation đầy đủ. Bộ dữ liệu gồm 8.400 video multiview tương ứng 300 câu, được chú thích gloss theo thời gian, câu tiếng Việt và keypoints cơ thể, hỗ trợ trực tiếp các bài toán dịch và sinh ngôn ngữ ký hiệu. Để minh chứng tính hữu dụng của bộ dữ liệu, chúng tôi tiến hành thực nghiệm cơ sở (baseline) với nhiều kiến trúc mạng nơ-ron khác nhau, đánh giá theo chiến lược signer-independent — trong đó người ký thuộc tập kiểm tra hoàn toàn không xuất hiện trong quá trình huấn luyện. Kết quả thực nghiệm cho thấy mô hình tốt nhất đạt BLEU-4 **66,82%** trên tập kiểm tra, khẳng định bộ dữ liệu VSL-GH mang chất lượng đủ cao để huấn luyện hệ thống dịch ngôn ngữ ký hiệu có khả năng tổng quát hóa.

---

## 1. INTRODUCTION

Ngôn ngữ ký hiệu là phương thức giao tiếp tự nhiên và chủ yếu của cộng đồng người khiếm thính trên toàn thế giới. Tại Việt Nam, theo ước tính của Tổ chức Y tế Thế giới (WHO), có khoảng 1,2 triệu người khiếm thính, trong đó phần lớn sử dụng Ngôn ngữ Ký hiệu Việt Nam (Vietnamese Sign Language – VSL) như ngôn ngữ mẹ đẻ [CITE]. Rào cản giao tiếp giữa cộng đồng khiếm thính và người nghe không biết ngôn ngữ ký hiệu tồn tại trong hầu hết mọi lĩnh vực của cuộc sống, nhưng trở nên đặc biệt nghiêm trọng trong môi trường y tế — nơi sự hiểu nhầm thông tin có thể dẫn đến hậu quả sức khỏe không thể đảo ngược. Tình trạng này đặt ra yêu cầu cấp thiết về các hệ thống dịch ngôn ngữ ký hiệu tự động (Sign Language Translation – SLT) có khả năng đóng vai trò phiên dịch số trong các tình huống y tế thực tiễn.

Trong những năm gần đây, các nghiên cứu về nhận dạng và dịch ngôn ngữ ký hiệu liên tục (Continuous Sign Language Recognition/Translation – CSLR/SLT) đã đạt được nhiều tiến bộ đáng kể nhờ sự phát triển của học sâu, đặc biệt là kiến trúc Transformer và cơ chế Connectionist Temporal Classification (CTC) [CITE]. Tuy nhiên, sự tiến bộ này chủ yếu tập trung vào các ngôn ngữ ký hiệu có nguồn lực dữ liệu phong phú như Ngôn ngữ Ký hiệu Đức (GSL) với bộ dữ liệu PHOENIX-14T [CITE], hay Ngôn ngữ Ký hiệu Mỹ (ASL) [CITE]. Phần lớn các hệ thống SLT tiên tiến hiện nay xử lý đầu vào là video RGB thô, kéo theo hai hạn chế căn bản: (1) yêu cầu tài nguyên tính toán cao do phải xử lý từng khung hình ảnh đầy đủ; và (2) tiềm ẩn rủi ro về quyền riêng tư do lưu trữ và xử lý khuôn mặt bệnh nhân — một vấn đề đặc biệt nhạy cảm trong môi trường y tế.

Đối với Ngôn ngữ Ký hiệu Việt Nam, khoảng cách về tài nguyên dữ liệu là rất lớn. Các nghiên cứu hiện có về VSL chủ yếu dừng lại ở mức **từ đơn lẻ** (isolated sign recognition), chưa cung cấp bộ dữ liệu câu hoàn chỉnh với annotation đầy đủ phục vụ bài toán dịch liên tục [CITE]. Sự vắng mặt hoàn toàn của bộ dữ liệu VSL cấp câu — đặc biệt trong lĩnh vực y tế — là rào cản trực tiếp khiến cộng đồng nghiên cứu chưa thể xây dựng và đánh giá các hệ thống SLT cho tiếng Việt một cách hệ thống.

Để lấp đầy khoảng trống này, bài báo này giới thiệu **VSL-GH** (Vietnamese Sign Language – General Health), bộ dữ liệu ngôn ngữ ký hiệu tiếng Việt đầu tiên ở mức câu trong ngữ cảnh khám sức khỏe tổng quát. VSL-GH được thu thập theo biến thể TP. Hồ Chí Minh với hệ thống ghi hình đa góc nhìn (multiview), gồm 8.400 video tương ứng 300 câu giao tiếp y tế thực tiễn, thực hiện bởi 6 người ký. Mỗi video được chú thích đầy đủ gồm: nhãn gloss căn chỉnh theo thời gian (temporal gloss annotation), bản dịch tiếng Việt và chuỗi keypoints cơ thể trích xuất qua MediaPipe — tạo nền tảng trực tiếp cho cả hai bài toán dịch (SLT) và sinh ngôn ngữ ký hiệu (Sign Language Production – SLP). Thay vì sử dụng video RGB, chúng tôi chọn biểu diễn dựa trên **skeleton keypoints** — giải pháp nhẹ, bảo vệ quyền riêng tư, đồng thời chứa đủ thông tin ngữ nghĩa về hình dạng tay, chuyển động cơ thể và biểu cảm khuôn mặt.

Tóm lại, những đóng góp chính của bài báo bao gồm:

1. **Bộ dữ liệu VSL-GH**: Bộ dữ liệu ngôn ngữ ký hiệu tiếng Việt cấp câu đầu tiên trong lĩnh vực y tế, với annotation đa lớp (gloss, văn bản, keypoints) và thiết kế multiview, được kiểm định chuyên môn bởi chuyên gia ngôn ngữ ký hiệu.

2. **Giao thức đánh giá Signer-Independent**: Chiến lược phân chia dữ liệu nghiêm ngặt tách biệt hoàn toàn người ký ở tập kiểm tra, đảm bảo đánh giá khả năng tổng quát hóa thực sự của mô hình trong điều kiện triển khai thực tế.

3. **Thực nghiệm cơ sở toàn diện**: Đánh giá nhiều kiến trúc mô hình và cấu hình đặc trưng trên VSL-GH, kèm nghiên cứu cắt bỏ (ablation study) và phân tích lỗi định tính sâu, cung cấp đường cơ sở (baseline) tham chiếu cho các nghiên cứu tương lai.

Phần còn lại của bài báo được tổ chức như sau: Mục 2 trình bày các công trình liên quan về bộ dữ liệu ngôn ngữ ký hiệu và các phương pháp SLT. Mục 3 mô tả chi tiết quy trình xây dựng và đặc điểm của VSL-GH. Mục 4 trình bày thiết lập thực nghiệm và kết quả. Mục 5 thảo luận các phát hiện, và Mục 6 kết luận bài báo.

---

## 2. RELATED WORK

### 2.1. Sign Language Datasets

Sự phát triển của các hệ thống nhận dạng và dịch ngôn ngữ ký hiệu phụ thuộc trực tiếp vào sự tồn tại của các bộ dữ liệu lớn, được chú thích đầy đủ. Trong những năm qua, một số bộ dữ liệu quy mô lớn đã được công bố cho các ngôn ngữ ký hiệu phổ biến.

Đối với Ngôn ngữ Ký hiệu Đức (DGS), **PHOENIX-14T** [CITE] là bộ dữ liệu chuẩn được sử dụng rộng rãi nhất trong nghiên cứu CSLR và SLT, gồm hơn 8.000 video câu từ chương trình dự báo thời tiết với nhãn gloss và bản dịch tiếng Đức. Bộ dữ liệu này đã trở thành benchmark de facto cho các mô hình SLT dựa trên video. Tuy nhiên, do được thu thập trong ngữ cảnh chuyên biệt (dự báo thời tiết), khả năng tổng quát hóa sang các miền khác còn hạn chế.

Đối với Ngôn ngữ Ký hiệu Mỹ (ASL), **How2Sign** [CITE] là một bộ dữ liệu đa phương thức và đa góc nhìn nổi bật, gồm hơn 80 giờ video ASL liên tục kèm bản ghi âm, phụ đề tiếng Anh và dữ liệu độ sâu. Một tập con 3 giờ được thu thập tại Panoptic Studio cho phép ước lượng tư thế 3D chi tiết. How2Sign cung cấp nền tảng mạnh mẽ cho nghiên cứu đa phương thức nhưng giới hạn ở ngữ cảnh giáo dục tổng quát và ngôn ngữ tiếng Anh. Một số bộ dữ liệu ASL khác như **OpenASL** [CITE] cũng được thu thập từ video trên internet với quy mô lớn, song chất lượng annotation không đồng đều.

Đối với Ngôn ngữ Ký hiệu Hàn Quốc, **bộ dữ liệu KETI** [CITE] gồm 14.672 video chất lượng cao tương ứng 105 câu trong tình huống khẩn cấp. Đây là một trong số ít bộ dữ liệu ngôn ngữ ký hiệu chuyên biệt theo miền, với phương pháp sử dụng keypoints cơ thể để dịch video ký hiệu sang câu ngôn ngữ tự nhiên. Kết quả thực nghiệm trên KETI cho thấy độ chính xác dịch đạt 93,28% trên tập kiểm định với 105 câu, chứng minh tiềm năng của các bộ dữ liệu chuyên biệt theo miền trong bài toán SLT.

Đối với Ngôn ngữ Ký hiệu Trung Quốc (CSL), các bộ dữ liệu như **CSL-Daily** [CITE] và **CSL** [CITE] cung cấp hàng chục nghìn video câu phục vụ nghiên cứu CSLR và SLT. Các bộ dữ liệu này đã tạo động lực cho nhiều nghiên cứu tiên tiến trong cộng đồng nhận dạng ngôn ngữ ký hiệu châu Á.

Nhìn chung, hầu hết các bộ dữ liệu ngôn ngữ ký hiệu quy mô lớn tập trung vào các ngôn ngữ ký hiệu có nguồn lực phong phú (DGS, ASL, CSL), trong khi các ngôn ngữ ký hiệu ít tài nguyên (low-resource) như VSL vẫn chưa được chú trọng đúng mức.

---

### 2.2. Sign Language Recognition and Translation Methods

Nghiên cứu về nhận dạng ngôn ngữ ký hiệu liên tục (CSLR) và dịch ngôn ngữ ký hiệu (SLT) đã có những bước tiến đáng kể nhờ các kiến trúc học sâu hiện đại. Các phương pháp dựa trên video RGB kết hợp mạng tích chập (CNN) và Transformer đã đạt được kết quả ấn tượng trên các bộ dữ liệu chuẩn như PHOENIX-14T [CITE]. Cơ chế Connectionist Temporal Classification (CTC) [CITE] được ứng dụng rộng rãi để căn chỉnh chuỗi gloss với chuỗi video mà không cần phân đoạn tường minh, trở thành thành phần then chốt trong nhiều hệ thống CSLR hiện đại.

Các phương pháp SLT tiên tiến hiện nay như CorrNet [CITE], TwoStream-SLR [CITE] hay C²SLR [CITE] tận dụng kết hợp đặc trưng ngoại hình (appearance) và chuyển động (motion) trích xuất từ video RGB, đạt WER thấp trên PHOENIX-14T. Tuy nhiên, những phương pháp này đòi hỏi tài nguyên tính toán lớn và không phù hợp với ràng buộc về quyền riêng tư — yếu tố đặc biệt quan trọng trong môi trường y tế khi video chứa khuôn mặt người dùng.

---

### 2.3. Skeleton-based Sign Language Understanding

Các phương pháp dựa trên keypoints cơ thể (skeleton-based) đã nổi lên như một hướng tiếp cận hiệu quả và nhẹ hơn so với phương pháp RGB, đặc biệt khi tài nguyên dữ liệu hoặc tính toán bị hạn chế. Keypoints cơ thể được trích xuất qua các công cụ như MediaPipe [CITE] hay OpenPose [CITE] cung cấp biểu diễn nhỏ gọn, không chứa thông tin nhận dạng cá nhân, phù hợp cho các ứng dụng nhạy cảm về quyền riêng tư. Nghiên cứu của Hwang et al. [CITE] trên bộ dữ liệu KETI đã chứng minh rằng keypoints chuẩn hóa từ tay, khuôn mặt và cơ thể là đủ để đạt độ chính xác dịch cao ngay cả khi dữ liệu huấn luyện có quy mô hạn chế.

Các mạng tích chập đồ thị (Graph Convolutional Networks – GCN) [CITE] khai thác cấu trúc liên kết tự nhiên của bộ xương người để mô hình hóa quan hệ không gian-thời gian giữa các khớp, đã được áp dụng thành công trong nhận dạng hành động và dần được nghiên cứu cho bài toán CSLR. Hướng tiếp cận này phù hợp đặc biệt với ngôn ngữ ký hiệu do tính diễn đạt cao của các chuyển động tay và cơ thể.

---

### 2.4. Vietnamese Sign Language Resources

Nghiên cứu về Ngôn ngữ Ký hiệu Việt Nam (VSL) hiện vẫn còn rất hạn chế so với các ngôn ngữ ký hiệu khác trên thế giới. Các công trình hiện có chủ yếu tập trung vào **nhận dạng ký hiệu đơn lẻ** (isolated sign recognition), tức là nhận dạng từng từ/ký hiệu riêng lẻ thay vì chuỗi câu liên tục [CITE]. Một số tập dữ liệu VSL nhỏ đã được công bố trong các nghiên cứu về nhận dạng ngôn ngữ ký hiệu cô lập [CITE], song các tập này không có annotation gloss căn chỉnh theo thời gian và không hỗ trợ bài toán dịch liên tục.

Đến nay, **chưa tồn tại bộ dữ liệu VSL nào ở mức câu** với annotation đầy đủ phục vụ bài toán CSLR và SLT, đặc biệt trong lĩnh vực y tế. Sự vắng mặt này là rào cản trực tiếp khiến cộng đồng nghiên cứu chưa thể xây dựng và đánh giá hệ thống SLT cho tiếng Việt một cách hệ thống. Bộ dữ liệu VSL-GH được đề xuất trong bài báo này nhằm lấp đầy khoảng trống đó, cung cấp nền tảng dữ liệu đầu tiên cho các nghiên cứu SLT và SLP trên VSL trong ngữ cảnh y tế.

---,/

## 3. DATASET CONSTRUCTION (VSL-GH)

### 3.1. Motivation and Domain

### 3.2. Data Collection Protocol

### 3.3. Signers

### 3.4. Sentence Set (300 sentences)

### 3.5. Recording Setup (multiview, angles)

### 3.6. Annotation Pipeline
- Temporal gloss annotation
- Vietnamese sentence labels
- Body keypoint extraction (MediaPipe)

### 3.7. Dataset Statistics

| Property | Value |
|---|---|
| Total videos | 8,400 |
| Sentences | 300 |
| Signers | 6 (S01–S06) |
| Views per sentence per signer | 4 (multiview) |
| Domain | General health check-up (HCM variant) |
| Annotations | Gloss (temporal) + Vietnamese sentence + keypoints |

---

## 4. BASELINE EXPERIMENTS

### 4.1. Experimental Setup

**Train/Val/Test Split (Signer-Independent):**

| Split | Signers | Samples |
|---|---|---|
| Train | S01 → S04 | 3,600 |
| Validation | S05 | 300 |
| Test | S06 | 300 |

**Evaluation Metrics:** BLEU-4 (↑), WER (↓)

### 4.2. Architectures Evaluated

*(Không kể tên cụ thể — mô tả theo loại kiến trúc)*
- Kiến trúc Transformer-based sử dụng keypoint tuyến tính
- Kiến trúc Graph Convolutional Network kết hợp Transformer

### 4.3. Results

| Setup | BLEU-4 Test (S06) | WER Test (S06) |
|---|---|---|
| Baseline (150 sentences) | 54.57% | 40.05% |
| Extended (300 sentences) | **66.82%** | **31.58%** |

### 4.4. Ablation Study (150-sentence subset)

| Configuration | BLEU-4 Val | BLEU-4 Test | WER Test |
|---|---|---|---|
| Full model (proposed) | **67.49%** | **54.57%** | **40.05%** |
| w/o face keypoints | 55.64% | 43.28% | 50.83% |
| w/o gloss supervision | 69.77% | 56.58% | 40.52% |
| Smaller capacity | 36.93% | 30.42% | 54.89% |
| Higher dropout (0.2) | 45.00% | 36.42% | 51.36% |
| Equal CTC-CE weight | 47.76% | 41.18% | 45.49% |
| Beam Search (k=4) | 46.26% | 44.40% | 43.37% |

---

## 5. DISCUSSION

*(Phần này cần viết thêm)*

---

## 6. CONCLUSION

*(Phần này cần viết thêm)*

---

## REFERENCES

*(Phần này cần bổ sung)*

---

## 📝 WRITING NOTES & TODO

- [ ] Viết phần Introduction: nêu gap, contribution
- [ ] Viết phần Related Work: SLT datasets, VSL resources, continuous SLR
- [ ] Expand Section 3 với chi tiết thu thập dữ liệu và annotation process
- [ ] Viết phần Discussion: phân tích error, giới hạn, hướng tương lai
- [ ] Viết Conclusion
- [ ] Bổ sung đầy đủ References (IEEE format)
- [ ] Chèn hình minh họa dataset (multiview, annotation sample, keypoint visualization)
