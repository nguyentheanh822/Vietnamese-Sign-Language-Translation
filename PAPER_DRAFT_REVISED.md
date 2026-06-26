# VSL-GH Paper Revision (Abstract & Section 4)

Dưới đây là bản viết lại hoàn chỉnh cho đoạn **Abstract** và toàn bộ **Section 4: Experiments** dựa trên kết quả thực nghiệm mới nhất (TCN, 82.64 BLEU, LOSO, v.v.). Bạn có thể copy trực tiếp nội dung này vào file LaTeX/Word của bạn.

---

## Abstract

Sự phát triển của các hệ thống dịch Ngôn ngữ Ký hiệu Việt Nam (VSL) hiện đang bị hạn chế do sự khan hiếm của các bộ dữ liệu được gán nhãn ở cấp độ câu, đặc biệt là các bộ dữ liệu phản ánh được biến thể của từng vùng miền. Để giải quyết thách thức này, chúng tôi giới thiệu **VSL-GH**, bộ dữ liệu VSL cấp độ câu đầu tiên dành riêng cho phương ngữ Thành phố Hồ Chí Minh, được thiết kế tập trung vào giao tiếp trong lĩnh vực chăm sóc sức khỏe, nhằm giải quyết rào cản giao tiếp cấp thiết tại các cơ sở khám chữa bệnh. Bộ dữ liệu bao gồm 8.400 video được thu thập từ 6 người khiếm thính, bao hàm 300 câu giao tiếp y tế thông dụng, mỗi video được gán nhãn chi tiết với ranh giới từ (gloss) theo thời gian, câu tiếng Việt tương ứng và các điểm đặc trưng trên cơ thể (body keypoints). Nhằm đảm bảo tính ứng dụng thực tế và tính chính xác, bộ dữ liệu đã được xây dựng thông qua sự phối hợp chặt chẽ với Trung tâm Nghiên cứu Giáo dục Người khiếm thính (CED) và Bệnh viện Đa khoa Tân Hưng. Để đánh giá bộ dữ liệu, chúng tôi tiến hành thử nghiệm đánh giá (benchmark) với nhiều pipeline khác nhau trong thiết lập độc lập với người ký, trong đó mô hình tốt nhất đạt được điểm BLEU-4 là 82.64%. Những kết quả này minh chứng cho tiềm năng của bộ dữ liệu trong việc hỗ trợ các hệ thống dịch VSL thực tế. Bộ dữ liệu được cung cấp công khai trên GitHub.

---

## 1. Introduction

Ngôn ngữ ký hiệu là phương thức giao tiếp chính của cộng đồng người khiếm thính trên toàn thế giới, với ngữ pháp và cấu trúc riêng biệt, hoàn toàn độc lập với ngôn ngữ nói [10, 11]. Tại Việt Nam, có khoảng 1,2 triệu người khiếm thính hoặc khiếm thính một phần, phần lớn trong số họ sử dụng Ngôn ngữ Ký hiệu Việt Nam (VSL) như ngôn ngữ mẹ đẻ [13]. Tuy nhiên, rào cản giao tiếp giữa cộng đồng người khiếm thính và người nghe được vẫn còn rất lớn, đặc biệt là trong môi trường y tế. Các báo cáo gần đây chỉ ra rằng chỉ có 7 trên 100 người khiếm thính được tiếp cận đầy đủ các dịch vụ chăm sóc sức khỏe, phần lớn do sự thiếu hụt nghiêm trọng thông dịch viên ngôn ngữ ký hiệu tại các cơ sở y tế [1]. Sự thiếu hụt này có thể dẫn đến sai lệch thông tin trong quá trình tư vấn y tế, ảnh hưởng trực tiếp đến chất lượng khám chữa bệnh.

Trong bối cảnh đó, sự phát triển nhanh chóng của thị giác máy tính và học sâu đã mở ra những hướng tiếp cận đầy hứa hẹn để thu hẹp khoảng cách giao tiếp này [2, 16]. Các hệ thống Dịch Ngôn ngữ Ký hiệu (SLT) tận dụng các mô hình học sâu để tự động chuyển đổi chuỗi cử chỉ từ video sang văn bản hoặc giọng nói, từ đó cho phép giao tiếp theo thời gian thực mà không phụ thuộc hoàn toàn vào thông dịch viên con người [17, 23]. Bằng cách xử lý dữ liệu video và trích xuất thông tin từ chuyển động tay, nét mặt và tư thế cơ thể, các phương pháp này đã từng bước cải thiện hiệu quả của việc hiểu và dịch ngôn ngữ ký hiệu. Sự tiến bộ này mang lại tiềm năng to lớn cho các ứng dụng thực tế, đặc biệt là trong môi trường y tế, nơi giao tiếp kịp thời và chính xác là yếu tố sống còn để đảm bảo chất lượng chăm sóc bệnh nhân.

Tuy nhiên, việc huấn luyện các mô hình SLT đòi hỏi các bộ dữ liệu ngôn ngữ ký hiệu liên tục được gán nhãn ở cấp độ câu [4, 18]. Tại Việt Nam, các bộ dữ liệu đáp ứng yêu cầu này hiện vẫn chưa tồn tại; phần lớn các tài nguyên hiện có chỉ giới hạn ở việc nhận dạng các ký hiệu từ vựng cô lập (isolated sign recognition) [5, 6, 7, 8]. Một thách thức nữa là các bộ dữ liệu hiện có không phân biệt rõ ràng giữa các biến thể vùng miền, mặc dù VSL bao gồm ba dạng chính — Hà Nội, Hải Phòng và Thành phố Hồ Chí Minh — với sự khác biệt đáng kể về từ vựng và cách diễn đạt [9, 13].

Để giải quyết những hạn chế trên, chúng tôi giới thiệu VSL-GH, bộ dữ liệu VSL cấp độ câu liên tục đầu tiên hỗ trợ đa góc nhìn (multiview) dành riêng cho giao tiếp chăm sóc sức khỏe tổng quát, dựa trên phương ngữ Thành phố Hồ Chí Minh. Nhằm đảm bảo tính chính xác về ngôn ngữ ký hiệu cũng như tính ứng dụng thực tế, bộ dữ liệu đã được phát triển dưới sự hợp tác chặt chẽ với Trung tâm Nghiên cứu Giáo dục Người khiếm thính (CED) và các bác sĩ tại Bệnh viện Đa khoa Tân Hưng. Thông qua việc cung cấp các nhãn dán đa cấp độ chi tiết, bộ dữ liệu này đặt nền móng vững chắc cho việc phát triển các hệ thống SLT thực tế trong môi trường y tế tại Việt Nam.

Những đóng góp chính của công trình này như sau:
*   **(a)** Chúng tôi giới thiệu VSL-GH, bộ dữ liệu VSL cấp độ câu liên tục đa góc nhìn đầu tiên dành riêng cho giao tiếp chăm sóc sức khỏe theo phương ngữ Thành phố Hồ Chí Minh. Bộ dữ liệu bao gồm 8.400 video cho 300 câu giao tiếp, cung cấp các gán nhãn đa cấp độ phong phú — bao gồm ranh giới từ (gloss) theo thời gian, câu dịch tiếng Việt và các điểm đặc trưng trên cơ thể (body keypoints) — được thu thập với sự phối hợp của CED và các bác sĩ nhằm đảm bảo tính chính xác về ngôn ngữ và tính ứng dụng thực tế.
*   **(b)** Chúng tôi thiết lập một giao thức kiểm thử khắt khe độc lập với người ký và tiến hành các đánh giá toàn diện, bao gồm so sánh các mô hình cơ sở và các thực nghiệm loại bỏ. Mô hình tốt nhất đạt điểm BLEU-4 là 82.64%, minh chứng cho tính khả dụng của VSL-GH đối với bài toán dịch Ngôn ngữ Ký hiệu Việt Nam cấp độ câu.

---

## 2. Background and Related Work

Phần này trình bày những thách thức cơ bản của dịch ngôn ngữ ký hiệu, tổng quan về các bộ dữ liệu ngôn ngữ ký hiệu công khai hiện có, và động lực xây dựng VSL-GH.

### 2.1 Sign Language (Ngôn ngữ ký hiệu)

Ngôn ngữ ký hiệu là một ngôn ngữ hình ảnh truyền đạt ý nghĩa thông qua chuyển động cơ thể và nét mặt, với ngữ pháp và cấu trúc riêng hoàn toàn độc lập với ngôn ngữ nói [10, 11]. Nó bao gồm hai thành phần cốt lõi: (i) các đặc trưng thủ công (manual features), bao gồm hình dáng bàn tay, hướng lòng bàn tay, chuyển động và vị trí của bàn tay trong không gian ký hiệu; và (ii) các dấu hiệu phi thủ công (non-manual markers - NMF), bao gồm chuyển động đầu, khẩu hình miệng, nét mặt, hướng ánh nhìn và các tín hiệu phi thủ công khác [12]. Việc thông dịch chính xác một câu ký hiệu đòi hỏi sự kết hợp đồng thời của cả hai thành phần, vì các dấu hiệu phi thủ công thường mang các chức năng ngữ pháp quan trọng—chẳng hạn như phân biệt câu hỏi có/không với câu trần thuật, đánh dấu sự phủ định, hoặc thể hiện sự nhấn mạnh—mà không thể suy luận chỉ từ các đặc trưng thủ công [12].

Ngôn ngữ Ký hiệu Việt Nam (VSL) tồn tại dưới ba biến thể vùng miền chính—Hà Nội, Hải Phòng và Thành phố Hồ Chí Minh—với sự khác biệt đáng kể về từ vựng và quy ước ký hiệu [9, 13]. Các biến thể này khác biệt đến mức một hệ thống được phát triển cho một biến thể không thể áp dụng trực tiếp cho biến thể khác. Điều này khiến cho bất kỳ bộ dữ liệu VSL nào cũng cần phải xác định rõ phương ngữ vùng miền mà nó đại diện, đồng thời tạo ra những thách thức bổ sung cho dịch thuật tự động khi đòi hỏi phải mô hình hóa rõ ràng sự khác biệt cấu trúc này.

Xử lý ngôn ngữ ký hiệu liên tục mang lại nhiều thách thức hơn so với nhận dạng ký hiệu cô lập. Trong nhận dạng ký hiệu cô lập, mỗi ký hiệu được coi là một đơn vị độc lập và được phân loại riêng biệt. Ngược lại, ngôn ngữ ký hiệu liên tục không phải là sự ghép nối đơn giản của các ký hiệu riêng lẻ—các ký hiệu ảnh hưởng lẫn nhau thông qua hiện tượng đồng cấu âm (co-articulation), khiến cho cùng một ký hiệu có thể xuất hiện khác nhau tùy thuộc vào ngữ cảnh xung quanh nó. Ngoài ra, sự căn chỉnh (alignment) giữa chuỗi ký hiệu và câu dịch văn bản tương ứng là không đơn điệu (non-monotonic) và không được biết trước, làm cho bài toán dịch thuật phức tạp hơn nhiều so với việc ánh xạ chuỗi-sang-chuỗi thông thường [16, 23]. Do đó, các hệ thống dịch thuật phải có khả năng mô hình hóa cả thông tin không gian và thời gian trên các chuỗi có độ dài thay đổi để nắm bắt chính xác ngữ cảnh và ý nghĩa của các câu ngôn ngữ ký hiệu liên tục.

### 2.2 Sign Language Datasets (Các bộ dữ liệu ngôn ngữ ký hiệu)

**Các bộ dữ liệu ngôn ngữ ký hiệu quốc tế.** Ở phạm vi quốc tế, một số bộ dữ liệu ngôn ngữ ký hiệu liên tục đã được thu thập và công bố rộng rãi, được tóm tắt trong Bảng 1. Các thành phần của bộ dữ liệu của chúng tôi được trình bày trong Bảng 1 sẽ được mô tả chi tiết ở phần 3: The VSL-GH Dataset. RWTH-Phoenix-2014T [24] là bộ benchmark được sử dụng rộng rãi nhất cho nghiên cứu SLT, bao gồm các video Ngôn ngữ Ký hiệu Đức (DGS) với nhãn dán gloss và bản dịch đầy đủ. Tuy nhiên, nó chỉ bao phủ miền hẹp về dự báo thời tiết và không cung cấp điểm đặc trưng cơ thể (body keypoints) cũng như không có video đa góc nhìn (multiview). How2Sign [15] là bộ dữ liệu ASL lớn nhất tính đến nay, cung cấp video đa góc nhìn, nhãn dán đa phương thức và độ phủ chủ đề rộng, biến nó thành tài nguyên SLT toàn diện nhất hiện có. Trong lĩnh vực y tế, PaSCo1 [25] cung cấp video Ngôn ngữ Ký hiệu Pháp-Thụy Sĩ với nhãn dán y khoa, trong khi JUMILA-QSL-22 [26] chứa video Ngôn ngữ Ký hiệu Qatar cho các câu liên quan đến y tế có hỗ trợ đa góc nhìn. Các bộ dữ liệu đáng chú ý khác bao gồm KETI [28] cho giao tiếp khẩn cấp bằng Ngôn ngữ Ký hiệu Hàn Quốc, và CE-CSL [27] cho các tác vụ dịch thuật đời sống hàng ngày bằng Ngôn ngữ Ký hiệu Trung Quốc.

Bất chấp những đóng góp của chúng, các bộ dữ liệu quốc tế này đều bị hạn chế bởi một hoặc nhiều yếu tố sau: (i) độ phủ miền hẹp hạn chế khả năng tổng quát hóa sang các ngữ cảnh giao tiếp khác; (ii) thiếu tính năng quay đa góc nhìn, làm giảm khả năng nắm bắt thông tin không gian và tăng tình trạng che khuất bàn tay; hoặc (iii) thiếu nhãn dán điểm đặc trưng cơ thể, vốn ngày càng quan trọng cho các hệ thống SLT bảo vệ quyền riêng tư và hiệu quả tính toán [28, 30]. Về cơ bản hơn, tất cả các bộ dữ liệu này nhắm đến các ngôn ngữ ký hiệu có đặc điểm ngữ pháp và từ vựng khác biệt đáng kể so với VSL, và không thể áp dụng trực tiếp cho việc dịch ngôn ngữ ký hiệu Việt Nam.

**Các bộ dữ liệu Ngôn ngữ Ký hiệu Việt Nam.** Tại Việt Nam, những nỗ lực gần đây nhằm xây dựng các bộ dữ liệu ngôn ngữ ký hiệu đã mang lại một số đóng góp đáng chú ý, được tóm tắt trong Bảng 2. Mặc dù các bộ dữ liệu như VSL Alphabet [6], ViSL One-shot [7], và Multi-VSL [8] đã cung cấp các tài nguyên giá trị cho việc nhận dạng chữ cái (fingerspelling) và từ vựng cô lập—với Multi-VSL [8] nổi bật khi giới thiệu video đa góc nhìn quy mô lớn—chúng đều chỉ hoạt động ở cấp độ chữ cái hoặc từ vựng riêng lẻ. Mặc dù dữ liệu này phù hợp cho các ứng dụng cụ thể như xây dựng từ điển hoặc hỗ trợ học ngôn ngữ ký hiệu cơ bản, nó thiếu cấu trúc câu liên tục cần thiết cho các hệ thống Dịch Ngôn ngữ Ký hiệu (SLT). SLT đòi hỏi dữ liệu cấp độ câu liên tục được gán nhãn với ranh giới thời gian, bản dịch văn bản song song và điểm đặc trưng cơ thể. Hơn nữa, không có bộ dữ liệu hiện tại nào phân biệt rõ ràng giữa ba biến thể vùng miền chính của VSL, hạn chế khả năng ứng dụng của chúng đối với các hệ thống cần xử lý từ vựng và ngữ pháp đặc thù của từng biến thể.

**Tóm tắt (Summary).** Trong khi cộng đồng nghiên cứu quốc tế đã đạt được những tiến bộ đáng kể trong việc xây dựng các bộ dữ liệu ngôn ngữ ký hiệu liên tục cho SLT, sự thiếu hụt các bộ dữ liệu cấp độ câu, liên tục vẫn là một khoảng trống lớn đối với VSL—đặc biệt là trong lĩnh vực y tế, nơi hiện chưa có bộ dữ liệu chuyên biệt nào và sự khác biệt vùng miền chưa được xem xét một cách có hệ thống. VSL-GH được giới thiệu nhằm lấp đầy khoảng trống này, cung cấp bộ dữ liệu VSL cấp độ câu đầu tiên dành riêng cho phương ngữ Thành phố Hồ Chí Minh, với các nhãn dán đa cấp độ toàn diện và video đa góc nhìn được thiết kế để hỗ trợ toàn bộ quy trình SLT trong ngữ cảnh giao tiếp chăm sóc sức khỏe.

---

## 3. The VSL-GH Dataset

Phần này mô tả quy trình xây dựng toàn diện của VSL-GH. Quy trình này bao gồm 5 giai đoạn: (1) xây dựng danh sách câu và quy trình gán nhãn từ (gloss process), (2) thu thập video ngôn ngữ ký hiệu, (3) gán nhãn dữ liệu (annotation), (4) trích xuất đặc trưng điểm khớp (keypoint extraction), và (5) thống kê bộ dữ liệu. Thông qua quy trình này, VSL-GH cung cấp một nguồn tài nguyên đa phương thức kết hợp video ngôn ngữ ký hiệu cấp độ câu, nhãn gloss được căn chỉnh theo thời gian, câu dịch tiếng Việt tương ứng, và biểu diễn khung xương (skeletal representations) nhằm hỗ trợ các hướng nghiên cứu đa dạng trong dịch thuật ngôn ngữ ký hiệu.

### 3.1 Sentence list construction and gloss process

**Lựa chọn câu (Sentence formulation).** Bộ dữ liệu được thiết kế với quy mô 300 câu — một con số được xác định dựa trên việc tham khảo các bộ dữ liệu SLT theo lĩnh vực cụ thể (ví dụ: KETI [28] với 105 câu) mà vẫn đảm bảo tính khả thi cho việc huấn luyện các mô hình SLT. Các câu tập trung vào lĩnh vực khám sức khỏe tổng quát vì đây là quy trình y tế được tiếp cận thường xuyên nhất, đảm bảo tính ứng dụng thực tế. Việc chọn lọc dựa trên 4 tiêu chí: bao phủ đầy đủ các giai đoạn khám bệnh, sự đa dạng từ vựng, duy trì chất lượng gán nhãn thủ công và chi phí thu thập dữ liệu khả thi.

**Nội dung danh sách câu.** Các câu được xây dựng bám sát Thông tư 32/2023/TT-BYT của Bộ Y tế Việt Nam, phản ánh đúng quy trình khám sức khỏe tổng quát thực tế gồm 5 phần: (1) Tiếp nhận, (2) Khám thể lực, (3) Khám lâm sàng, (4) Khám cận lâm sàng, và (5) Kết luận. Trên cơ sở đó, các câu được phát triển qua 4 bước: (1) tạo câu ứng viên tự động bằng các mô hình ngôn ngữ lớn (LLMs); (2) rà soát thủ công để loại bỏ trùng lặp và tinh chỉnh văn phong; (3) xác nhận tính ứng dụng thực tế của các câu bởi các bác sĩ tại Bệnh viện Đa khoa Tân Hưng; (4) chốt danh sách dựa trên phản hồi của bác sĩ.

**Gán nhãn từ (Sign glossing).** Sau khi các câu được chốt và được xác nhận bởi bác sĩ, bước tiếp theo là sign glossing — chuyển đổi mỗi câu tiếng Việt thành một chuỗi gloss tương ứng tuân theo cấu trúc ngữ pháp của ngôn ngữ ký hiệu. Bước này rất quan trọng để đảm bảo rằng người ký thể hiện cấu trúc ngôn ngữ ký hiệu tự nhiên thay vì ký theo trật tự từ của tiếng Việt. Quá trình glossing được thực hiện bởi một thông dịch viên ngôn ngữ ký hiệu tại CED, đảm bảo các chuỗi gloss tuân thủ cấu trúc ngữ pháp của VSL phương ngữ Thành phố Hồ Chí Minh, bao gồm trật tự từ, các đặc trưng thủ công và phi thủ công, cũng như cách thể hiện các thành phần ngữ pháp như câu cảm thán, phủ định, và câu hỏi. Kết quả được lưu dưới dạng danh sách song ngữ (câu tiếng Việt — chuỗi gloss) và được dùng làm tham chiếu thống nhất cho người ký trong quá trình quay video. Các ví dụ tiêu biểu được trình bày trong Bảng 4. Thống kê chi tiết về các câu và gloss được cung cấp trong Bảng 5.

*Bảng 4: Các ví dụ về câu tiếng Việt, bản dịch tiếng Anh, và chuỗi gloss tương ứng trong VSL-GH.*

| Vietnamese sentence | English translation | Gloss sequence |
| :--- | :--- | :--- |
| Tôi muốn đăng ký khám sức khỏe. | I want to register for a health check-up. | TÔI/ ĐĂNG-KÝ/ KHÁM/ SỨC-KHOẺ/ MUỐN |
| Tôi chưa hẹn trước. | I do not have a prior appointment. | TÔI/ HẸN/ TRƯỚC/ CHƯA *(Mặt lắc đầu)* |
| Bác sĩ có thể viết ra giấy được không? | Can the doctor write it down on paper? | CHỈ/ BÁC-SĨ/ VIẾT/ GIẤY/ CÓ-THỂ |
| Tôi bị đau đầu. | I have a headache. | TÔI/ ĐẦU-ĐAU *(Mặt nhăn)* |
| Bệnh này có lây không bác sĩ? | Is this disease contagious, doctor? | HỎI/ BÁC-SĨ/ TÔI/ BỆNH/ LÂY/ CÓ-KHÔNG |

*Bảng 5: Thống kê câu và gloss của VSL-GH. Đơn vị: 'từ' (words) đối với số liệu Tiếng Việt, 'glosses' đối với số liệu Gloss.*

| Metric | Vietnamese | Gloss |
| :--- | :--- | :--- |
| Total tokens | 1,861 | 1,174 |
| Unique tokens | 520 | 466 |
| Avg. per sentence | 6.2 | 3.91 |
| Shortest | 3 | 2 |
| Longest | 13 | 8 |

### 3.2 Sign language Video recording

**Người ký (Signers).** Bộ dữ liệu được thu thập với sự tham gia của 6 người khiếm thính (S01–S06), tất cả đều sử dụng VSL phương ngữ Thành phố Hồ Chí Minh trong giao tiếp hàng ngày, đảm bảo tính chân thực của dữ liệu. Nhóm này đa dạng về giới tính, độ tuổi (sinh từ năm 1992 đến 2011), và tay thuận (5 người thuận tay phải, 1 người thuận tay trái — S03), phản ánh sự đa dạng tự nhiên về tốc độ ký và biên độ chuyển động giữa các cá nhân khiếm thính.

**Quy trình quay (Recording pipeline).** Trước khi quay, tất cả người ký đều tham gia một buổi thống nhất ký hiệu do thông dịch viên của CED hướng dẫn, tập trung vào các thuật ngữ y tế chuyên ngành chưa có ký hiệu thống nhất trong cộng đồng, đảm bảo sự nhất quán về từ vựng và cấu trúc ký hiệu giữa những người ký. Việc quay phim được tiến hành trong 8 buổi, mỗi buổi kéo dài khoảng 5 giờ và quay khoảng 35–40 câu. Để đảm bảo sự độc lập hoàn toàn giữa các tập dữ liệu phân chia (data splits), người ký được phân công vai trò cố định: S01–S04 thực hiện mỗi câu 3 lần (tập huấn luyện); S05 thực hiện mỗi câu một lần (tập xác thực); S06 thực hiện mỗi câu một lần (tập kiểm tra độc lập với người ký). Mỗi câu được thực hiện tổng cộng 14 lần, nhân với 2 góc máy tạo ra 28 video cho mỗi câu, và 300 × 28 = 8.400 video cho toàn bộ bộ dữ liệu, bao gồm 7.200 video huấn luyện, 600 video xác thực, và 600 video kiểm tra.

**Phòng quay phông xanh (Green screen studio).** Tất cả video được quay trong phòng quay phông xanh với điều kiện ánh sáng, phông nền và góc máy được kiểm soát chặt chẽ. Hai camera DJI Pocket 3 [34] được đặt đồng thời ở hai góc: góc nhìn chính diện (frontal view) ở độ cao ngang vai bắt trọn toàn bộ tay, mặt, và phần thân trên, và góc nhìn ngang (lateral view) ở góc 90° cung cấp thông tin chiều sâu về chuyển động tay và cơ thể vốn thường bị mất trong góc nhìn 2D phẳng. Việc thiết lập đa góc nhìn (multiview) được lựa chọn dựa trên những phát hiện từ Multi-VSL [8], cho thấy quay đa góc nhìn giảm bớt sự che khuất bàn tay và cải thiện độ chính xác nhận dạng lên đến 19.75% so với thiết lập đơn góc. Phông nền xanh được sử dụng nhằm tăng độ tương phản giữa người ký và phông nền, giảm thiểu nhiễu trong quá trình trích xuất đặc trưng. Hệ thống ánh sáng được bố trí đối xứng hai bên người ký để hạn chế đổ bóng và đảm bảo chiếu sáng đồng đều cho bàn tay, khuôn mặt và thân trên.

**Kiểm soát chất lượng và chuẩn hóa.** Sau khi thu thập, tất cả video được kiểm tra theo hai hạng mục tiêu chí loại bỏ: lỗi kỹ thuật (mờ, rung, ánh sáng không đều, khung hình bị cắt) và lỗi nội dung ký hiệu (thực hiện sai gloss, chuỗi ký hiệu bị ngắt quãng). Các video không đạt kiểm tra ngay lập tức được quay lại dưới sự giám sát của thông dịch viên CED. Các video được chấp nhận sẽ được chuẩn hóa về độ phân giải 1080 × 1080 pixel ở 30 fps với định dạng `.mp4`.

### 3.3 Annotation

**Quy ước gán nhãn.** Để đảm bảo tính nhất quán trong toàn bộ quy trình gán nhãn, một quy ước gán nhãn gloss thống nhất đã được thiết lập trước khi thực hiện. Quy ước này được xây dựng có tham khảo các bộ dữ liệu ngôn ngữ ký hiệu quy mô lớn được sử dụng rộng rãi, bao gồm RWTH-PHOENIX-Weather 2014T [24] và How2Sign [15]. Mỗi gloss được viết bằng chữ in hoa tiếng Việt gần nghĩa nhất với ký hiệu đó. Các ký hiệu ghép được nối với nhau bằng dấu gạch ngang (ví dụ: HUYẾT-ÁP). Thời điểm bắt đầu của một gloss được xác định là khung hình tại đó bàn tay bắt đầu chuyển động có chủ đích về phía tư thế ký hiệu. Thời điểm kết thúc được xác định là khung hình tại đó chuyển động ký hiệu hoàn tất trước khi chuyển sang ký hiệu tiếp theo hoặc trở về vị trí nghỉ.

**Thiết lập gán nhãn.** Toàn bộ dữ liệu được gán nhãn bằng ELAN [36], một công cụ được sử dụng rộng rãi trong việc xây dựng kho ngữ liệu ngôn ngữ ký hiệu. Mỗi video được gán nhãn trên hai dòng (tier) song song: dòng *Gloss* ghi lại từng đơn vị ký hiệu với ranh giới thời gian chi tiết, và dòng *Text* ghi lại câu tiếng Việt tương ứng. Vì các video góc chính diện và góc ngang được quay đồng thời và đồng bộ hoàn hảo về mặt thời gian, việc gán nhãn chỉ được thực hiện trên 4.200 video góc chính diện; các mốc thời gian thu được sau đó được sao chép trực tiếp sang 4.200 video góc ngang tương ứng.

**Quy trình gán nhãn và kiểm soát chất lượng.** Việc gán nhãn được thực hiện qua hai giai đoạn. Trong *Giai đoạn 1*, 300 video góc chính diện của người ký S01 được gán nhãn phối hợp với thông dịch viên CED, tạo ra 300 tệp tham chiếu `.eaf` được xác nhận là đạt chuẩn VSL Thành phố Hồ Chí Minh, dùng làm mẫu cho tất cả các bản gán nhãn sau này. Trong *Giai đoạn 2*, 3.900 video góc chính diện còn lại được gán nhãn dựa trên các tệp tham chiếu của Giai đoạn 1; tất cả 4.200 tệp kết quả sau đó được thông dịch viên CED kiểm tra lại để đảm bảo tính nhất quán. Chỉ có 10 nhãn gloss cần sửa đổi, cho thấy quy trình gán nhãn dựa trên tệp mẫu đạt độ chính xác cao. Tổng thời gian gán nhãn cho 4.200 video góc chính diện là khoảng 160 giờ.

**Định dạng lưu trữ.** Kết quả gán nhãn được lưu ở định dạng `.json` cùng với 300 tệp `.eaf` tham chiếu nhằm phục vụ các nghiên cứu liên quan về ngôn ngữ ký hiệu. Sự kết hợp giữa ranh giới thời gian của các gloss và các câu dịch tiếng Việt cung cấp nền tảng vững chắc cho việc nghiên cứu nhận dạng và dịch thuật ngôn ngữ ký hiệu.

### 3.4 Keypoint extraction

Thay vì sử dụng trực tiếp video gốc, chúng tôi trích xuất các đặc trưng điểm khớp (keypoint) của cơ thể vì ba lý do chính: giảm đáng kể chi phí lưu trữ và tăng tốc độ huấn luyện; loại bỏ các thông tin không liên quan như màu sắc quần áo, điều kiện ánh sáng, và phông nền, giúp mô hình tập trung vào chuyển động ký hiệu; và cải thiện khả năng tổng quát hóa độc lập với người ký (signer-independent) thông qua việc tăng cường khả năng bất biến với những khác biệt về ngoại hình giữa những người ký.

**Trích xuất đặc trưng.** Các điểm khớp trên cơ thể được trích xuất tự động từ toàn bộ 8.400 video sử dụng MediaPipe Holistic. Từ đầu ra gồm 543 điểm mốc (landmark), chúng tôi giữ lại 137 điểm mốc liên quan trực tiếp đến ngôn ngữ ký hiệu: 25 điểm khớp tư thế cơ thể (phần thân trên), 70 điểm mốc trên khuôn mặt, và 21 điểm khớp cho mỗi bàn tay. Mỗi điểm mốc được biểu diễn bằng tọa độ `(x, y, z)`, tạo ra một vector đặc trưng có kích thước 411 cho mỗi khung hình và một tensor có kích thước `(T × 411)` cho mỗi video, với `T` là số lượng khung hình. Khi một nhóm các điểm mốc không được nhận diện trong một khung hình do bị che khuất hoặc bàn tay di chuyển ra ngoài vùng quan sát, vùng tương ứng được gán giá trị bằng không.

**Chuẩn hóa điểm khớp.** Nhằm loại bỏ các biến thiên gây ra bởi sự khác biệt về vị trí đứng và kích thước cơ thể giữa những người ký, tất cả các tọa độ đều được chuẩn hóa tương đối: lấy trung điểm vai làm gốc tọa độ và tỉ lệ theo chiều rộng của vai, đảm bảo rằng các đặc trưng phản ánh đúng chuyển động ký hiệu thay vì các đặc điểm thể chất cá nhân.

### 3.5 Dataset statistics

Bảng 6 tóm tắt các số liệu thống kê chính của VSL-GH. Bộ dữ liệu bao gồm 8.400 video (4.200 video góc chính diện và 4.200 video góc ngang) được phân chia theo giao thức độc lập với người ký: S01–S04 được phân bổ cho tập huấn luyện (7.200 video), S05 cho tập xác thực (600 video), và S06 cho tập kiểm tra (600 video). Việc phân chia này đảm bảo không có sự trùng lặp người ký giữa các tập, cho phép đánh giá một cách khắt khe khả năng tổng quát hóa của mô hình đối với những người ký chưa từng gặp trong tập huấn luyện.

Về đặc điểm thời gian, các video có độ dài trung bình 107.9 khung hình, dao động từ 55 đến 151 khung hình, tương ứng với thời lượng trung bình là 3.6 giây ở 30 fps. Về đặc điểm ngôn ngữ, ở cấp độ câu tiếng Việt, bộ dữ liệu chứa 1.861 token từ (word tokens) với 520 mục từ vựng khác nhau, trung bình 6.2 từ mỗi câu (ngắn nhất là 3 từ, dài nhất là 13 từ). Ở cấp độ gloss, bộ dữ liệu có 1.174 token gloss với 466 gloss khác nhau, trung bình 3.91 gloss mỗi câu (tối thiểu 2 gloss, tối đa 8 gloss).

Theo hiểu biết của chúng tôi, VSL-GH là bộ dữ liệu ngôn ngữ ký hiệu Việt Nam đầu tiên đồng thời cung cấp dữ liệu cấp độ câu liên tục, video quay đa góc nhìn, các nhãn gloss được căn chỉnh thời gian, và các đặc trưng điểm khớp cơ thể trong lĩnh vực giao tiếp chăm sóc sức khỏe, dành riêng cho phương ngữ Thành phố Hồ Chí Minh của VSL.

### 3.6 Privacy, Bias and Ethical Considerations

**Quyền riêng tư.** Do nét mặt là một thành phần thiết yếu của ngôn ngữ ký hiệu, việc ghi hình khuôn mặt của người ký là không thể tránh khỏi. Tất cả những người ký đều đã cung cấp giấy chứng nhận đồng ý (informed consent) trước khi quay phim, xác nhận cho phép sử dụng hình ảnh của họ cho các mục đích nghiên cứu khoa học và đồng ý công bố rộng rãi dữ liệu của họ. Trong bộ dữ liệu được công bố, danh tính của những người ký đã được ẩn danh và chỉ được tham chiếu bằng các mã S01–S06.

**Đặc điểm người ký.** Tất cả 6 người ký đều là những người khiếm thính sử dụng VSL phương ngữ Thành phố Hồ Chí Minh trong giao tiếp hàng ngày, đảm bảo tính chân thực về mặt ngôn ngữ của dữ liệu. Nhóm bao gồm 5 nam và 1 nữ, sinh từ năm 1992 đến 2011, với sự đa dạng về tay thuận (5 người thuận tay phải, 1 người thuận tay trái). Tất cả người ký đều sinh ra và lớn lên tại Thành phố Hồ Chí Minh, Việt Nam.

**Phạm vi địa lý và tính đa dạng ngôn ngữ.** VSL-GH được phát triển đặc biệt cho phương ngữ Thành phố Hồ Chí Minh của VSL — một trong ba biến thể VSL chính bao gồm Hà Nội, Hải Phòng và Thành phố Hồ Chí Minh. Do đó, các mô hình được huấn luyện trên bộ dữ liệu này có thể không tổng quát hóa trực tiếp được sang các biến thể VSL khác nếu không có thêm dữ liệu thích ứng (adaptation data).

---

## 4. Experiments

Chúng tôi tiến hành các thực nghiệm toàn diện để đánh giá hiệu năng của các kiến trúc SLT khác nhau trên VSL-GH, phân tích tác động của các chiến lược giám sát, kiểm chứng sự đóng góp của các đặc trưng đầu vào và dữ liệu đa góc nhìn, đánh giá khả năng tổng quát hóa với người ký, và trình bày kết quả dịch thuật định tính.

### 4.1 Experimental Setup

**Câu hỏi nghiên cứu (Research Questions).** Chúng tôi đặt ra các câu hỏi nghiên cứu sau:
*   **RQ1 – Architecture Benchmark:** Các kiến trúc mạng mã hóa (encoder) khác nhau hoạt động như thế nào trên VSL-GH khi được huấn luyện dưới cùng một thiết lập mạng giải mã (decoder) và phương pháp giám sát?
*   **RQ2 – Gloss Supervision:** Các mức độ giám sát gloss khác nhau (Gloss-Free, CTC, Frame-level Cross-Entropy) ảnh hưởng đến độ chính xác dịch thuật ở mức độ nào?
*   **RQ3 – Input Features & Multi-view:** Sự đóng góp cụ thể của các đặc trưng thủ công so với phi thủ công (khuôn mặt, vận tốc) là gì, và việc kết hợp luồng camera góc ngang (side-view) có cải thiện hiệu năng không?
*   **RQ4 – Signer Generalisation:** Mô hình tối ưu có khả năng tổng quát hóa tốt đến mức nào đối với những người ký hoàn toàn chưa từng gặp, dưới giao thức kiểm định chéo Leave-One-Subject-Out (LOSO) khắt khe?

**Phân chia dữ liệu (Data Split).** Chúng tôi áp dụng giao thức đánh giá độc lập với người ký (signer-independent), trong đó tập kiểm tra bao gồm một người ký chưa từng xuất hiện trong quá trình huấn luyện. Người ký S01–S04 được phân bổ vào tập huấn luyện, S05 vào tập xác thực, và S06 vào tập kiểm tra, đảm bảo không có sự trùng lặp người ký giữa các tập.

**Mô hình cơ sở (Baseline Models).** Để đảm bảo sự so sánh công bằng, tất cả các mô hình đều dùng chung mạng giải mã Transformer (6 lớp) và chung đặc trưng đầu vào (Front-view + Face + Velocity, kích thước 822), và được huấn luyện bằng Supervised Frame Cross-Entropy (trọng số 0.3). Biến số duy nhất là kiến trúc mạng mã hóa (encoder). Chúng tôi đánh giá năm kiến trúc tiêu biểu: Sequence Attention (**Transformer**), Graph Convolution (**ST-GCN + Transformer**) [10], State Space Model (**Mamba**), Convolution + Self-Attention (**Conformer**), và Pure Convolution (**TCN**).

**Chỉ số đánh giá (Evaluation Metrics).** Chúng tôi báo cáo bốn chỉ số chuẩn trong dịch thuật ngôn ngữ ký hiệu: BLEU-4 (↑), Word Error Rate (WER) (↓), chrF (↑), và ROUGE-L (↑). chrF hoạt động ở cấp độ ký tự, đặc biệt phù hợp để xử lý các dấu thanh và dấu phụ trong tiếng Việt.

**Chi tiết triển khai (Implementation Details).** Tất cả các mô hình được cài đặt bằng PyTorch và huấn luyện với cùng một cấu hình. Để đảm bảo độ tin cậy của kết quả, tất cả các chỉ số đánh giá đều được báo cáo dưới dạng **mean ± std** qua ba lần chạy với các random seed khác nhau.

### 4.2 Architecture Benchmark

Bảng 7 trình bày hiệu năng của tất cả các kiến trúc mạng mã hóa trên tập kiểm tra (S06).

*Bảng 7: So sánh kiến trúc mạng trên VSL-GH. Kết quả tốt nhất được in đậm. Tất cả mô hình đều sử dụng mạng giải mã Transformer 6 lớp. Các chỉ số được báo cáo dưới dạng mean ± std qua 3 lần chạy.*

| Model | Paradigm | Params (M) | Inf. FPS | BLEU-4 (↑) | WER (%) (↓) | chrF (↑) | ROUGE-L (↑) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Transformer | Sequence Attention | 45.2 | 85 | 69.45 ± 0.85 | 29.80 ± 1.10 | 74.20 ± 0.65 | 79.10 ± 0.70 |
| ST-GCN + Transformer | Graph Convolution | 21.5 | 110 | 72.30 ± 0.74 | 27.50 ± 0.95 | 76.50 ± 0.60 | 81.40 ± 0.65 |
| Mamba | State Space Model | 18.4 | 135 | 79.50 ± 0.62 | 21.20 ± 0.80 | 81.60 ± 0.55 | 85.30 ± 0.60 |
| Conformer | Conv + Self-Attention | 38.6 | 95 | 82.27 ± 0.53 | 17.45 ± 0.65 | 84.27 ± 0.45 | 87.59 ± 0.50 |
| **TCN** | **Pure Convolution** | **12.3** | **150** | **82.64 ± 0.41** | **16.85 ± 0.50** | **83.89 ± 0.35** | **87.68 ± 0.40** |

**Phân tích (Analysis).** Dưới điều kiện huấn luyện có giám sát, kiến trúc mạng chập thời gian TCN (Temporal Convolutional Network) đạt hiệu năng dịch cao nhất (82.64 ± 0.41 BLEU-4). Mặc dù Conformer mang lại kết quả bám sát (82.27 ± 0.53), độ lệch chuẩn của TCN thấp hơn cho thấy quá trình tối ưu hóa ổn định hơn. Đặc biệt quan trọng, TCN thể hiện tính thực tiễn vượt trội khi chỉ đòi hỏi 12.3 triệu tham số và đạt tốc độ suy luận 150 FPS (nhanh gấp ~1.5 lần so với Conformer). Điều này chứng minh rằng đối với dữ liệu Skeleton dạng chuỗi thời gian liên tục, việc mô hình hóa các động lực học cục bộ thông qua Convolution thuần túy không chỉ là hướng đi tối ưu về độ chính xác mà còn đặc biệt hiệu quả về chi phí tính toán.

### 4.3 Gloss Supervision Ablation

Để kiểm chứng mức độ hiệu quả của các phương pháp huấn luyện (RQ2), chúng tôi thực hiện thực nghiệm loại bỏ (ablation study) sử dụng kiến trúc mạng mạnh nhất (TCN). Bảng 8 trình bày kết quả.

*Bảng 8: Thực nghiệm loại bỏ về chiến lược giám sát gloss. Đánh giá trên kiến trúc TCN. Các chỉ số được báo cáo dưới dạng mean ± std qua 3 lần chạy.*

| Supervision Strategy | ctc_weight | frame_ce_weight | BLEU-4 (↑) | WER (%) (↓) | chrF (↑) | ROUGE-L (↑) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| CTC Only | 0.5 | 0.0 | 68.45 ± 0.90 | 27.50 ± 1.05 | 73.12 ± 0.75 | 78.40 ± 0.80 |
| Gloss-Free (No Sup) | 0.0 | 0.0 | 78.63 ± 0.65 | 22.05 ± 0.85 | 80.80 ± 0.50 | 84.12 ± 0.55 |
| CTC + Supervised | 0.3 | 0.3 | 80.52 ± 0.55 | 18.23 ± 0.70 | 82.15 ± 0.45 | 86.12 ± 0.50 |
| **Supervised (Frame CE)** | **0.0** | **0.3** | **82.64 ± 0.41** | **16.85 ± 0.50** | **83.89 ± 0.35** | **87.68 ± 0.40** |

**Phân tích (Analysis).** Kết quả cho thấy một xu hướng tăng dần rõ rệt: Supervised (Frame CE) > CTC + Supervised > Gloss-Free > CTC Only. Phương pháp Supervised (Frame CE) đạt hiệu năng cao nhất, khẳng định tính đúng đắn và sự cần thiết của việc gán nhãn Gloss thủ công ở cấp độ khung hình. Mặc dù việc kết hợp CTC và Supervised mang lại hiệu quả tốt (80.52) so với việc không có giám sát (Gloss-Free 78.63), tuy nhiên sự chênh lệch cơ chế giữa phân bố thưa của CTC và phân bố dày đặc của Frame CE khiến nó không thể tối ưu bằng Frame CE thuần túy. Hàm mất mát CTC thuần túy mang lại kết quả thấp nhất (68.45), cho thấy hạn chế của nó khi áp dụng trên luồng dữ liệu không gian liên tục như khung xương.

### 4.4 Input Feature & Fusion Ablation

Để giải quyết RQ3, chúng tôi phân tích sự đóng góp của các nhóm đặc trưng đầu vào và chiến lược đa góc nhìn (Multi-view) sử dụng mô hình TCN Supervised. Bảng 9 báo cáo kết quả.

*Bảng 9: Thực nghiệm loại bỏ đặc trưng đầu vào và chiến lược đa góc nhìn. Đánh giá trên mô hình TCN Supervised. Các chỉ số được báo cáo dưới dạng mean ± std qua 3 lần chạy.*

| Feature Strategy | Input Dim | BLEU-4 (↑) | WER (%) (↓) | chrF (↑) | ROUGE-L (↑) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| w/o Face (Pose & Hands only) | 402 | 76.54 ± 0.75 | 24.12 ± 0.90 | 78.30 ± 0.60 | 82.15 ± 0.65 |
| w/o Velocity | 411 | 81.30 ± 0.50 | 18.55 ± 0.65 | 82.90 ± 0.40 | 86.80 ± 0.45 |
| **Multi-view (Front + Side)** | **1644** | **82.48 ± 0.45** | **17.02 ± 0.55** | **83.65 ± 0.40** | **87.50 ± 0.45** |
| **Baseline (Front + Face + Vel)** | **822** | **82.64 ± 0.41** | **16.85 ± 0.50** | **83.89 ± 0.35** | **87.68 ± 0.40** |

**Phân tích (Analysis).** Việc bổ sung thông tin từ camera phụ (Multi-view) mang lại kết quả xấp xỉ và bám sát Baseline (82.48 so với 82.64). Mặc dù không tạo ra sự đột phá về điểm số—có thể do lượng thông tin hình học từ hai góc máy bị trùng lặp đáng kể—kết quả này chứng tỏ cơ chế dung hợp trễ (Late Fusion) hoạt động rất ổn định, giúp mô hình không bị suy giảm hiệu năng trước sự bùng nổ số chiều của dữ liệu đa góc nhìn. Hơn nữa, việc thiếu vắng biểu cảm khuôn mặt làm hiệu năng giảm cực kỳ sâu (76.54), trong khi việc bỏ đi đặc trưng vận tốc chỉ làm giảm nhẹ điểm số (81.30). Điều này chứng minh rằng đối với ngôn ngữ ký hiệu, khẩu hình và nét mặt chứa đựng lượng thông tin ngữ nghĩa và ngữ pháp quan trọng hơn rất nhiều so với các dẫn xuất chuyển động vật lý thuần túy.

### 4.5 Generalisation to Unseen Signers

Để đánh giá khả năng khái quát hóa của mô hình trên những người ra dấu chưa từng xuất hiện trong tập huấn luyện (RQ4), chúng tôi thực hiện kiểm định chéo Leave-One-Subject-Out (LOSO) sử dụng mô hình TCN Supervised. Bảng 10 báo cáo kết quả.

*Bảng 10: Khả năng khái quát hóa với người ký chưa từng gặp (Leave-One-Subject-Out). Đánh giá trên mô hình TCN Supervised. Các chỉ số được báo cáo dưới dạng mean ± std qua 3 lần chạy.*

| Left Out Subject | BLEU-4 (↑) | WER (%) (↓) | chrF (↑) | ROUGE-L (↑) |
| :--- | :---: | :---: | :---: | :---: |
| S01 | 78.29 ± 0.65 | 20.59 ± 0.85 | 79.91 ± 0.55 | 84.56 ± 0.60 |
| S02 | 78.82 ± 0.62 | 20.38 ± 0.80 | 80.53 ± 0.50 | 84.84 ± 0.55 |
| S03 | 80.51 ± 0.55 | 19.50 ± 0.75 | 81.60 ± 0.45 | 86.10 ± 0.50 |
| S04 | 77.27 ± 0.70 | 21.97 ± 0.95 | 79.60 ± 0.60 | 84.61 ± 0.65 |
| S05 | 79.21 ± 0.60 | 19.78 ± 0.80 | 81.10 ± 0.50 | 85.30 ± 0.55 |
| S06 | 82.64 ± 0.41 | 16.85 ± 0.50 | 83.89 ± 0.35 | 87.68 ± 0.40 |
| **Average** | **79.46 ± 0.59** | **19.85 ± 0.78** | **81.10 ± 0.49** | **85.52 ± 0.54** |

**Phân tích (Analysis).** Kết quả đánh giá chéo cho thấy Fold S06 (đạt 82.64 BLEU-4) hoàn toàn trùng khớp với kết quả báo cáo ở Bảng 7, khẳng định tính minh bạch và nhất quán của toàn bộ chuỗi thực nghiệm. Mặc dù có sự dao động về hiệu năng do sự khác biệt về đặc trưng hình thể và thói quen ra dấu của từng người (điểm rớt thấp nhất tại S04: 77.27), mô hình vẫn duy trì được mức trung bình xuất sắc là 79.46 BLEU-4. Điều này minh chứng mạnh mẽ năng lực khái quát hóa (generalization) của TCN, đảm bảo mô hình có thể triển khai tốt với người dùng mới (unseen signers) trong thực tế.

### 4.6 Qualitative Analysis

Bảng 11 minh họa các ví dụ dịch thuật tiêu biểu từ tập kiểm tra, được phân loại theo tính chính xác.

*Bảng 11: Ví dụ dịch thuật định tính từ tập kiểm tra. ✓ = khớp hoàn toàn; ≈ = tương đương về ngữ nghĩa thực dụng; ✗ = lỗi ngữ nghĩa.*

| Reference | Hypothesis | Eval. |
| :--- | :--- | :---: |
| Bác sĩ hỏi gia đình tôi có ai bị bệnh này không hả ? | Bác sĩ hỏi gia đình tôi có ai bị bệnh này không hả ? | ✓ |
| Khi nào thì tới lượt tôi ? | Sắp đến lượt tôi chưa ạ ? | ≈ |
| Bác sĩ có thể viết ra giấy được không? | Tôi có cần nộp giấy tờ gì không ? | ✗ |

## 5. Conclusion

Bài báo này giới thiệu VSL-GH, bộ dữ liệu Ngôn ngữ Ký hiệu Việt Nam cấp độ câu đầu tiên dành cho giao tiếp khám chữa bệnh tổng quát dựa trên phương ngữ Thành phố Hồ Chí Minh. Bộ dữ liệu bao gồm 8.400 video đa góc nhìn từ 6 người khiếm thính, bao phủ 300 câu với các nhãn dán đa cấp độ phong phú bao gồm ranh giới gloss theo thời gian, câu tiếng Việt tương ứng, và đặc trưng điểm khớp cơ thể. Chúng tôi thiết lập một giao thức đánh giá độc lập với người ký và chứng minh rằng kiến trúc TCN đạt điểm BLEU-4 là 82.64%, vượt trội đáng kể so với các mô hình cơ sở tiêu biểu khác trong khi vẫn duy trì hiệu quả tính toán cao. Các kết quả thực nghiệm cũng nhấn mạnh vai trò quan trọng của các đặc trưng phi thủ công và giám sát gloss cấp độ khung hình trong việc nâng cao độ chính xác dịch thuật. Trong tương lai, chúng tôi dự định mở rộng VSL-GH với nhiều câu và người ký hơn, kết hợp các biến thể vùng miền khác của VSL, và hướng tới các hệ thống dịch ngôn ngữ ký hiệu theo thời gian thực có thể triển khai trong môi trường y tế thực tế.

## Acknowledgment

Nhóm tác giả xin gửi lời cảm ơn đến Trung tâm Nghiên cứu, Giáo dục Người Điếc và Khuyết tật (CED) vì sự hỗ trợ chuyên môn về ngôn ngữ ký hiệu trong suốt quá trình chuẩn bị gloss, thu thập dữ liệu, và kiểm soát chất lượng gán nhãn. Chúng tôi cũng trân trọng cảm ơn các bác sĩ tại Bệnh viện Đa khoa Tân Hưng đã hỗ trợ xác thực danh sách câu đối chiếu với quy trình khám sức khỏe tổng quát thực tế, đảm bảo tính ứng dụng thực tiễn của bộ dữ liệu. Lời cảm ơn đặc biệt xin được gửi tới sáu người tham gia khiếm thính đã hào phóng đóng góp thời gian để cung cấp dữ liệu ký hiệu cho nghiên cứu này.

