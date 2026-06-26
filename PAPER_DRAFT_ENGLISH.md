# VSL-GH Paper Revision (Abstract & Section 4)

---

## Abstract

Progress in Vietnamese Sign Language (VSL) translation has been limited by the scarcity of sentence-level annotated datasets, particularly those capturing regional variations. To address this gap, we introduce **VSL-GH**, the first sentence-level VSL dataset dedicated to the Ho Chi Minh City variant, designed for general healthcare communication to address the urgent communication barriers deaf individuals face in medical settings. The dataset comprises 8,400 videos from 6 deaf signers, covering 300 healthcare-related sentences, each annotated with temporal gloss boundaries, Vietnamese sentences, and body keypoints. To ensure practical applicability and accuracy, the dataset was constructed in close collaboration with the Center for Research and Education of the Deaf and Hard of Hearing (CED) and Tan Hung General Hospital. To evaluate the dataset, we benchmark several baseline pipelines in a signer-independent setting, with the best-performing architecture achieving a BLEU-4 score of 82.64%. These results demonstrate its potential for supporting practical VSL translation systems. The dataset is publicly available on GitHub.

---

## 1. Introduction

Sign language is the primary mode of communication for deaf communities worldwide, with its own grammar and structure distinct from spoken languages [10, 11]. In Vietnam, approximately 1.2 million people are deaf or hard of hearing, the majority of whom use Vietnamese Sign Language (VSL) as their first language [13]. However, communication barriers between the deaf community and hearing individuals remain substantial, particularly in healthcare settings. Recent reports indicate that only 7 in 100 deaf individuals have full access to healthcare services, largely due to the critical shortage of sign language interpreters at medical facilities [1]. This shortage can lead to miscommunication during medical consultations, potentially compromising the quality of patient care.

In this context, the rapid development of computer vision and deep learning has opened up promising approaches to bridging this communication gap [2, 16]. Sign Language Translation (SLT) systems leverage deep learning models to automatically convert sequences of gestures from video into text or speech, thereby enabling real-time communication without relying entirely on human interpreters [17, 23]. By processing video data and extracting information from hand movements, facial expressions, and body posture, these methods have progressively improved the effectiveness of understanding and translating sign language. This progress offers significant potential for real-world applications, particularly in healthcare settings, where timely and accurate communication is critical to ensuring the quality of patient care.

However, training SLT models requires continuous sign language datasets in which data is annotated at the sentence level [4, 18]. In Vietnam, datasets meeting this requirement do not yet exist; the majority of available resources are limited to isolated sign recognition [5, 6, 7, 8]. A further challenge is that existing datasets do not distinguish between regional variants, despite the fact that VSL encompasses three major forms—Hanoi, Hai Phong, and Ho Chi Minh City—which differ substantially in vocabulary and expression [9, 13].

To address the limitations above, we introduce VSL-GH, the first continuous sentence-level multiview VSL dataset for general healthcare communication, based on the Ho Chi Minh City variant. To ensure both sign language accuracy and practical applicability, the dataset was developed in close collaboration with the Center for Research and Education of the Deaf and Hard of Hearing (CED) and physicians at Tan Hung General Hospital. By providing comprehensive multi-level annotations, this dataset lays a solid foundation for supporting the development of practical SLT systems in Vietnamese healthcare settings.

The main contributions of this work are as follows:
*   **(a)** We introduce VSL-GH, the first continuous sentence-level multiview VSL dataset dedicated to general healthcare communication in the Ho Chi Minh City variant. The dataset comprises 8,400 videos across 300 sentences, featuring rich multi-level annotations—including temporal gloss boundaries, Vietnamese text translations, and body keypoints—collected in collaboration with CED and physicians to ensure both linguistic and practical applicability.
*   **(b)** We establish a rigorous signer-independent benchmark protocol and conduct comprehensive evaluations, including baseline comparisons and ablation studies. The best model achieves a BLEU-4 score of 82.64%, demonstrating the usability of VSL-GH for sentence-level Vietnamese Sign Language translation.

---

## 2. Background and Related Work

This section presents the fundamental challenges of sign language translation, an overview of currently available public sign language datasets, and the motivation for VSL-GH.

### 2.1 Sign Language

Sign language is a visual language that conveys meaning through body movements and facial expressions, with its own grammar and structure entirely independent of spoken language [10, 11]. It comprises two core components: (i) manual features, including handshape, palm orientation, movement, and location of the hands in signing space; and (ii) non-manual markers (NMF), including head movements, mouth patterns, facial expressions, eye gaze, and other non-manual cues [12]. Accurately interpreting a signed utterance requires the simultaneous integration of both components, as non-manual markers often carry critical grammatical functions—such as distinguishing yes/no questions from declarative sentences, marking negation, or expressing emphasis—that cannot be inferred from manual features alone [12].

Vietnamese Sign Language (VSL) exists in three major regional variants—Hanoi, Hai Phong, and Ho Chi Minh City—which differ substantially in vocabulary and signing conventions [9, 13]. These variants are mutually distinct to the extent that a system developed for one variant cannot be directly applied to another, making it essential for any VSL dataset to clearly specify the regional form it represents, creating additional challenges for automatic translation that requires explicit modeling of this structural divergence.

Processing continuous sign language presents additional challenges beyond isolated sign recognition. In isolated sign recognition, each sign is treated as an independent unit and classified independently. In contrast, continuous sign language is not a simple concatenation of individual signs—signs influence one another through co-articulation, causing the same sign to appear differently depending on its surrounding context. Additionally, the alignment between sign sequences and their corresponding text translations is non-monotonic and not known in advance, making the translation task significantly more complex than a straightforward sequence-to-sequence mapping [16, 23]. Consequently, translation systems must be capable of modeling both spatial and temporal information across variable-length sequences to accurately capture the context and meaning of continuous sign language utterances.

### 2.2 Sign Language Datasets

**International sign language datasets.** At the international level, a number of continuous sign language datasets have been collected and made publicly available, as summarised in Table 1. The components of our dataset presented in Table 1 will be detailed in Section 3: The VSL-GH Dataset. RWTH-Phoenix-2014T [24] is the most widely adopted benchmark for SLT research, featuring German Sign Language (DGS) videos with full gloss and translation annotations. However, it covers only the narrow domain of weather forecasting and provides neither body keypoints nor multiview recordings. How2Sign [15] is the largest ASL dataset to date, providing multiview recording, multi-modal annotations, and broad topical coverage, making it the most comprehensive SLT resource currently available. In the healthcare domain, PaSCo1 [25] provides French-Swiss Sign Language videos with medical annotations, while JUMILA-QSL-22 [26] contains Qatari Sign Language videos across healthcare-related sentences with multiview support. Other notable datasets include KETI [28] for emergency communication in Korean Sign Language, and CE-CSL [27] for daily-life translation tasks in Chinese Sign Language.

Despite their contributions, these international datasets are collectively constrained by one or more of the following limitations: (i) narrow domain coverage that restricts generalisation to other communication contexts; (ii) absence of multiview recording, which limits the capture of spatial information and increases hand occlusion; or (iii) absence of body keypoint annotations, which are increasingly important for privacy-preserving and computationally efficient SLT systems [28, 30]. More fundamentally, all of these datasets target sign languages with grammatical and lexical characteristics that differ substantially from VSL, and none can be directly applied to Vietnamese sign language translation.

**Vietnamese Sign Language datasets.** In Vietnam, recent efforts to build sign language datasets have yielded several noteworthy contributions, as summarised in Table 2. While datasets such as VSL Alphabet [6], ViSL One-shot [7], and Multi-VSL [8] have provided valuable resources for fingerspelling and isolated word recognition—with Multi-VSL [8] notably introducing large-scale multiview recordings—they all operate strictly at the isolated letter or word level. Although such data is suitable for specific applications like dictionary construction or basic sign language learning support, it lacks the continuous, sentence-level structure required for Sign Language Translation (SLT) systems. SLT demands continuous sentence-level data annotated with temporal boundaries, parallel text translations, and body keypoints. Furthermore, none of the existing datasets clearly distinguishes between the three major VSL regional variants, limiting their applicability to systems that need to handle variant-specific vocabulary and grammar.

**Summary.** While the international research community has made significant progress in building continuous sign language datasets for SLT, the lack of continuous, sentence-level datasets remains a critical gap for VSL—particularly in the healthcare domain, where no specialised dataset currently exists and where regional variations have not been systematically considered. VSL-GH is introduced to fill this gap, providing the first sentence-level VSL dataset dedicated to the Ho Chi Minh City variant, with comprehensive multi-level annotations and multiview recordings designed to support the full SLT pipeline in a healthcare communication context.

---

## 3. The VSL-GH Dataset

This section describes the comprehensive construction pipeline of VSL-GH. The construction pipeline consists of five stages: (1) sentence list construction and gloss process, (2) sign language video recording, (3) annotation, (4) keypoint extraction, and (5) dataset statistics. Through this pipeline, VSL-GH provides a multimodal resource combining sentence-level sign language videos, temporally aligned gloss annotations, corresponding Vietnamese sentences, and skeletal representations to support diverse research directions in sign language translation.

### 3.1 Sentence list construction and gloss process

**Sentence formulation.** The dataset was designed with a scale of 300 sentences—a size determined by reference to domain-specific SLT datasets (e.g., KETI [28] with 105 sentences) that remain suitable for training SLT models. The sentences focus on the general health examination domain as it is the most frequently accessed medical procedure, ensuring practical relevance. The selection was guided by four criteria: full coverage of all examination stages, sufficient vocabulary diversity, maintainable manual annotation quality, and feasible data collection cost.

**Sentence list content.** The sentences were constructed following Circular 32/2023/TT-BYT of the Vietnamese Ministry of Health, reflecting the actual general health examination procedure divided into five parts: (1) Registration, (2) Physical examination, (3) Clinical examination, (4) Paraclinical examination, and (5) Conclusion. On this basis, the sentences were developed through four steps: (1) candidate sentences were automatically generated using large language models (LLMs); (2) manual review to remove duplicates and refine phrasing; (3) validation of the practical applicability of sentences by physicians at Tan Hung General Hospital; (4) finalisation based on physician feedback.

**Sign glossing.** After the sentences were finalised and confirmed by physicians, the next step was sign glossing—converting each Vietnamese sentence into a corresponding gloss sequence following the grammatical structure of sign language. This step is crucial to ensure that signers produce natural sign language structure rather than signing in the word order of Vietnamese. The glossing process was carried out by a sign language interpreter at CED, ensuring that the gloss sequences conform to the grammatical structure of the Ho Chi Minh City variant of VSL, including sign order, manual and non-manual features, and the expression of grammatical components such as exclamations, negations, and questions. The results were stored as a bilingual list (Vietnamese sentence — gloss sequence) and used as a unified sign reference for signers during video recording. Representative examples are presented in Table 4. Detailed statistics of the sentences and glosses are provided in Table 5.

*Table 4: Examples of Vietnamese sentence, English translation, and gloss sequence pairs in VSL-GH.*

| Vietnamese sentence | English translation | Gloss sequence |
| :--- | :--- | :--- |
| Tôi muốn đăng ký khám sức khỏe. | I want to register for a health check-up. | TÔI/ ĐĂNG-KÝ/ KHÁM/ SỨC-KHOẺ/ MUỐN |
| Tôi chưa hẹn trước. | I do not have a prior appointment. | TÔI/ HẸN/ TRƯỚC/ CHƯA *(Mặt lắc đầu)* |
| Bác sĩ có thể viết ra giấy được không? | Can the doctor write it down on paper? | CHỈ/ BÁC-SĨ/ VIẾT/ GIẤY/ CÓ-THỂ |
| Tôi bị đau đầu. | I have a headache. | TÔI/ ĐẦU-ĐAU *(Mặt nhăn)* |
| Bệnh này có lây không bác sĩ? | Is this disease contagious, doctor? | HỎI/ BÁC-SĨ/ TÔI/ BỆNH/ LÂY/ CÓ-KHÔNG |

*Table 5: Sentence and gloss statistics of VSL-GH. Units: 'words' for Vietnamese metrics, 'glosses' for Gloss metrics.*

| Metric | Vietnamese | Gloss |
| :--- | :--- | :--- |
| Total tokens | 1,861 | 1,174 |
| Unique tokens | 520 | 466 |
| Avg. per sentence | 6.2 | 3.91 |
| Shortest | 3 | 2 |
| Longest | 13 | 8 |

### 3.2 Sign language Video recording

**Signers.** The dataset was collected with the participation of 6 deaf signers (S01–S06), all of whom use the Ho Chi Minh City variant of VSL in daily communication, ensuring the authenticity of the data. The group is diverse in gender, age (born between 1992 and 2011), and handedness (5 right-handed, 1 left-handed — S03), reflecting natural variability in signing speed and movement amplitude among deaf individuals.

**Recording pipeline.** Prior to recording, all signers attended a sign unification session led by a CED sign language interpreter, focusing on specialised medical terminology that lacks a unified sign in the community, ensuring consistency in vocabulary and sign structure across signers. Recording was conducted over 8 sessions, each lasting approximately 5 hours and covering approximately 35–40 sentences. To ensure complete independence between data splits, signers were assigned to fixed roles: S01–S04 performed each sentence 3 times (training set); S05 performed each sentence once (validation set); S06 performed each sentence once (signer-independent test set). Each sentence was performed a total of 14 times, multiplied by 2 camera angles yielding 28 videos per sentence, and 300 × 28 = 8,400 videos for the entire dataset, comprising 7,200 training videos, 600 validation videos, and 600 test videos.

**Green screen studio.** All videos were recorded in a green screen studio under tightly controlled conditions of lighting, background, and camera angles. Two DJI Pocket 3 [34] cameras were placed simultaneously at two angles: a frontal view at shoulder height capturing the full hand, face, and upper body region, and a lateral view at 90° providing depth information about hand and body movements that is typically lost in a flat 2D view. The multiview setup was chosen based on findings from Multi-VSL [8], which showed that multiview recording reduces hand occlusion and improves recognition accuracy by up to 19.75% compared to single-view setups. A green screen background was used to increase the contrast between the signer and the background, minimising noise during feature extraction. Lighting was arranged symmetrically on both sides of the signer to limit shadows and ensure uniform illumination of the hands, face, and upper body.

**Quality control and standardisation.** Following data collection, all videos were inspected according to two categories of rejection criteria: technical failures (blur, shake, uneven lighting, cropped frames) and sign content errors (incorrect gloss execution, interrupted signing sequences). Videos that failed inspection were immediately re-recorded under CED interpreter supervision. Accepted videos were standardised to 1080 × 1080 pixels at 30 fps in `.mp4` format.

### 3.3 Annotation

**Annotation conventions.** To ensure consistency throughout the annotation process, a unified gloss annotation convention was established prior to annotation. This convention was developed with reference to widely used large-scale sign language datasets including RWTH-PHOENIX-Weather 2014T [24] and How2Sign [15]. Each gloss is written in uppercase Vietnamese corresponding most closely to the meaning of the sign. Compound signs are connected with hyphens (e.g., HUYẾT-ÁP). The start time of a gloss is defined as the frame at which the hand begins purposeful movement towards the sign configuration. The end time is defined as the frame at which the sign movement is completed before transitioning to the next sign or returning to a rest position.

**Annotation setup.** All data was annotated using ELAN [36], a tool widely used in sign language corpus construction. Each video was annotated on two parallel tiers: a *Gloss* tier recording each sign unit with detailed temporal boundaries, and a *Text* tier recording the corresponding Vietnamese sentence. Since frontal and lateral videos were recorded simultaneously and are perfectly synchronised in time, annotation was performed only on the 4,200 frontal-view videos; the resulting timestamps were then directly transferred to the corresponding 4,200 lateral-view videos.

**Annotation workflow and quality control.** Annotation was carried out in two phases. In *Phase 1*, 300 frontal-view videos of signer S01 were collaboratively annotated with a CED sign language interpreter, producing 300 reference `.eaf` files confirmed to meet the Ho Chi Minh City VSL standard, serving as templates for all subsequent annotation. In *Phase 2*, the remaining 3,900 frontal-view videos were annotated based on the Phase 1 reference files; all 4,200 resulting files were then reviewed by the CED interpreter to ensure consistency. Only 10 gloss labels required correction, indicating that the template-based annotation process achieved high accuracy. The total annotation time for the 4,200 frontal-view videos was approximately 160 hours.

**Storage format.** Annotation results are stored in `.json` format along with 300 reference `.eaf` files to serve related sign language research. The combination of temporal gloss boundaries and Vietnamese sentence translations provides a foundation for sign language recognition and translation research.

### 3.4 Keypoint extraction

Rather than using raw video directly, we extract body keypoint features for three main reasons: significantly reducing storage cost and accelerating training; removing irrelevant information such as clothing colour, lighting conditions, and background, allowing the model to focus on signing motion; and improving signer-independent generalisation through higher invariance to appearance differences across signers.

**Keypoint extraction.** Body keypoints were automatically extracted from all 8,400 videos using MediaPipe Holistic. From the 543-landmark output, we retained 137 landmarks directly relevant to sign language: 25 body pose keypoints (upper body), 70 facial landmarks, and 21 keypoints per hand. Each landmark is represented by `(x, y, z)` coordinates, yielding a feature vector of size 411 per frame and a tensor of size `(T × 411)` per video, where `T` is the number of frames. When a group of landmarks is not detected in a frame due to hand occlusion or the hand moving outside the field of view, the corresponding region is assigned a value of zero.

**Keypoint normalisation.** To remove variation caused by differences in standing position and body size across signers, all coordinates are relatively normalised: centred on the shoulder midpoint and scaled by the shoulder width, ensuring that the features reflect signing motion rather than individual physical characteristics.

### 3.5 Dataset statistics

Table 6 summarises the key statistics of VSL-GH. The dataset comprises 8,400 videos (4,200 frontal-view and 4,200 lateral-view) partitioned under a signer-independent protocol: S01–S04 are assigned to training (7,200 videos), S05 to validation (600 videos), and S06 to testing (600 videos). This split guarantees zero signer overlap across partitions, enabling a rigorous evaluation of model generalisation to unseen signers.

In terms of temporal characteristics, videos average 107.9 frames in length, ranging from 55 to 151 frames, corresponding to an average duration of 3.6 seconds at 30 fps. In terms of linguistic characteristics, at the Vietnamese sentence level, the dataset contains 1,861 word tokens with 520 unique vocabulary items, averaging 6.2 words per sentence (shortest 3 words, longest 13 words). At the gloss level, the dataset has 1,174 gloss tokens with 466 unique glosses, averaging 3.91 glosses per sentence (minimum 2 glosses, maximum 8 glosses).

To our knowledge, VSL-GH is the first Vietnamese sign language dataset to simultaneously provide continuous sentence-level data, multiview recordings, temporally aligned gloss annotations, and body keypoint features in a healthcare communication domain, dedicated to the Ho Chi Minh City variant of VSL.

### 3.6 Privacy, Bias and Ethical Considerations

**Privacy.** Since facial expressions are an essential component of sign language, recording the signer’s face is unavoidable. All signers provided written informed consent prior to recording, confirming permission to use their images for scientific research purposes and agreeing to the public release of their data. In the released dataset, signer identities are anonymised and referenced only by the codes S01–S06.

**Signer characteristics.** All 6 signers are deaf individuals using the Ho Chi Minh City variant of VSL in daily communication, ensuring the linguistic authenticity of the data. The group comprises 5 males and 1 female, born between 1992 and 2011, with diversity in handedness (5 right-handed, 1 left-handed). All signers were born and raised in Ho Chi Minh City, Vietnam.

**Geographic scope and language variety.** VSL-GH was developed specifically for the Ho Chi Minh City variant of VSL—one of the three major VSL variants comprising Hanoi, Hai Phong, and Ho Chi Minh City. Consequently, models trained on this dataset may not generalise directly to other VSL variants without additional adaptation data.

---

## 4. Experiments

We conduct comprehensive experiments to evaluate the performance of various SLT architectures on VSL-GH, analyse the impact of supervision strategies, examine the contribution of input features and multi-view data, assess signer generalisation, and present qualitative translation results.

### 4.1 Experimental Setup

**Research Questions.** We investigate the following research questions:
*   **RQ1 – Architecture Benchmark:** How do different encoder architectures perform on VSL-GH when trained under identical decoder and supervision settings?
*   **RQ2 – Gloss Supervision:** To what extent do different levels of gloss supervision (Gloss-Free, CTC, Frame-level Cross-Entropy) impact translation accuracy?
*   **RQ3 – Input Features & Multi-view:** What is the specific contribution of manual vs. non-manual features (face, velocity), and does incorporating a side-view camera stream improve performance?
*   **RQ4 – Signer Generalisation:** How well does the optimal model generalise to entirely unseen signers under a rigorous Leave-One-Subject-Out (LOSO) cross-validation?

**Data Split.** We adopt a signer-independent evaluation protocol, in which the test set consists of a signer who never appears during training. Signers S01–S04 are assigned to the training set, S05 to the validation set, and S06 to the test set, ensuring zero signer overlap across partitions.

**Baseline Models.** To ensure a fair comparison, all models share the same Transformer decoder (6 layers) and the same input features (Front-view + Face + Velocity, 822 dimensions) and are trained using Supervised Frame Cross-Entropy (weight 0.3). The only variable is the encoder architecture. We benchmark five representative architectures: Sequence Attention (**Transformer**), Graph Convolution (**ST-GCN + Transformer**) [10], State Space Model (**Mamba**), Convolution + Self-Attention (**Conformer**), and Pure Convolution (**TCN**). 

**Evaluation Metrics.** We report four standard metrics for sign language translation: BLEU-4 (↑), Word Error Rate (WER) (↓), chrF (↑), and ROUGE-L (↑). chrF operates at the character level, making it particularly well-suited for handling diacritics and tonal markers in Vietnamese.

**Implementation Details.** All models are implemented in PyTorch and trained with identical configurations. To ensure the reliability of the results, all evaluation metrics are reported as **mean ± std** across three runs with different random seeds.

### 4.2 Architecture Benchmark

Table 7 presents the performance of all encoder architectures on the test set (S06).

*Table 7: Architecture Benchmark on VSL-GH. Best results are in bold. All models use a 6-layer Transformer decoder. Metrics are reported as mean ± std over 3 runs.*

| Model | Paradigm | Params (M) | Inf. FPS | BLEU-4 (↑) | WER (%) (↓) | chrF (↑) | ROUGE-L (↑) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Transformer | Sequence Attention | 45.2 | 85 | 69.45 ± 0.85 | 29.80 ± 1.10 | 74.20 ± 0.65 | 79.10 ± 0.70 |
| ST-GCN + Transformer | Graph Convolution | 21.5 | 110 | 72.30 ± 0.74 | 27.50 ± 0.95 | 76.50 ± 0.60 | 81.40 ± 0.65 |
| Mamba | State Space Model | 18.4 | 135 | 79.50 ± 0.62 | 21.20 ± 0.80 | 81.60 ± 0.55 | 85.30 ± 0.60 |
| Conformer | Conv + Self-Attention | 38.6 | 95 | 82.27 ± 0.53 | 17.45 ± 0.65 | 84.27 ± 0.45 | 87.59 ± 0.50 |
| **TCN** | **Pure Convolution** | **12.3** | **150** | **82.64 ± 0.41** | **16.85 ± 0.50** | **83.89 ± 0.35** | **87.68 ± 0.40** |

**Analysis.** Under supervised training conditions, the Temporal Convolutional Network (TCN) achieves the highest translation performance (82.64 ± 0.41 BLEU-4). Although the Conformer yields highly competitive results (82.27 ± 0.53), the lower standard deviation of the TCN indicates a more stable optimisation process. Crucially, the TCN demonstrates superior practical viability: it requires only 12.3 million parameters and achieves an inference speed of 150 FPS (approximately 1.5 times faster than the Conformer). This demonstrates that for continuous time-series skeleton data, modelling local dynamics through pure convolution is not only optimal in terms of accuracy but also highly computationally efficient.

### 4.3 Gloss Supervision Ablation

To investigate the effectiveness of different supervision strategies (RQ2), we conduct an ablation study using the strongest architecture (TCN). Table 8 presents the results.

*Table 8: Gloss Supervision Ablation. Evaluated using the TCN architecture. Metrics are reported as mean ± std over 3 runs.*

| Supervision Strategy | ctc_weight | frame_ce_weight | BLEU-4 (↑) | WER (%) (↓) | chrF (↑) | ROUGE-L (↑) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| CTC Only | 0.5 | 0.0 | 68.45 ± 0.90 | 27.50 ± 1.05 | 73.12 ± 0.75 | 78.40 ± 0.80 |
| Gloss-Free (No Sup) | 0.0 | 0.0 | 78.63 ± 0.65 | 22.05 ± 0.85 | 80.80 ± 0.50 | 84.12 ± 0.55 |
| CTC + Supervised | 0.3 | 0.3 | 80.52 ± 0.55 | 18.23 ± 0.70 | 82.15 ± 0.45 | 86.12 ± 0.50 |
| **Supervised (Frame CE)** | **0.0** | **0.3** | **82.64 ± 0.41** | **16.85 ± 0.50** | **83.89 ± 0.35** | **87.68 ± 0.40** |

**Analysis.** The results demonstrate a clear monotonic trend: Supervised (Frame CE) > CTC + Supervised > Gloss-Free > CTC Only. The Supervised (Frame CE) approach achieves the highest performance, confirming the necessity and value of manual gloss annotation at the frame level. While combining CTC and Supervised Frame CE yields good results (80.52) compared to the Gloss-Free approach (78.63), the inherent difference between the sparse distribution of CTC and the dense distribution of Frame CE prevents it from surpassing pure Frame CE. The pure CTC loss yields the lowest results (68.45), indicating its limitations when applied to continuous spatial data streams like skeletons.

### 4.4 Input Feature & Fusion Ablation

To address RQ3, we analyse the contribution of input feature groups and the multi-view strategy using the TCN Supervised model. Table 9 reports the results.

*Table 9: Input Feature and Fusion Ablation. Evaluated using the TCN Supervised model. Metrics are reported as mean ± std over 3 runs.*

| Feature Strategy | Input Dim | BLEU-4 (↑) | WER (%) (↓) | chrF (↑) | ROUGE-L (↑) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| w/o Face (Pose & Hands only) | 402 | 76.54 ± 0.75 | 24.12 ± 0.90 | 78.30 ± 0.60 | 82.15 ± 0.65 |
| w/o Velocity | 411 | 81.30 ± 0.50 | 18.55 ± 0.65 | 82.90 ± 0.40 | 86.80 ± 0.45 |
| **Multi-view (Front + Side)** | **1644** | **82.48 ± 0.45** | **17.02 ± 0.55** | **83.65 ± 0.40** | **87.50 ± 0.45** |
| **Baseline (Front + Face + Vel)** | **822** | **82.64 ± 0.41** | **16.85 ± 0.50** | **83.89 ± 0.35** | **87.68 ± 0.40** |

**Analysis.** Incorporating side-view information (Multi-view) yields results comparable to the baseline (82.48 vs. 82.64). Although it does not produce a significant performance breakthrough—likely due to overlapping geometric information between the two camera angles—this result demonstrates that the late fusion mechanism is highly stable, preventing performance degradation despite the dimensionality expansion of multi-view data. Furthermore, removing facial expressions causes a severe drop in performance (76.54), whereas removing velocity features only slightly reduces the score (81.30). This confirms that in sign language, mouth patterns and facial expressions carry substantially more vital semantic and grammatical information than pure physical motion derivatives.

### 4.5 Generalisation to Unseen Signers

To evaluate how well the model generalises to signers unseen during training (RQ4), we perform a Leave-One-Subject-Out (LOSO) cross-validation using the TCN Supervised model. Table 10 reports the results.

*Table 10: Generalisation to Unseen Signers (Leave-One-Subject-Out). Evaluated using the TCN Supervised model. Metrics are reported as mean ± std over 3 runs.*

| Left Out Subject | BLEU-4 (↑) | WER (%) (↓) | chrF (↑) | ROUGE-L (↑) |
| :--- | :---: | :---: | :---: | :---: |
| S01 | 78.29 ± 0.65 | 20.59 ± 0.85 | 79.91 ± 0.55 | 84.56 ± 0.60 |
| S02 | 78.82 ± 0.62 | 20.38 ± 0.80 | 80.53 ± 0.50 | 84.84 ± 0.55 |
| S03 | 80.51 ± 0.55 | 19.50 ± 0.75 | 81.60 ± 0.45 | 86.10 ± 0.50 |
| S04 | 77.27 ± 0.70 | 21.97 ± 0.95 | 79.60 ± 0.60 | 84.61 ± 0.65 |
| S05 | 79.21 ± 0.60 | 19.78 ± 0.80 | 81.10 ± 0.50 | 85.30 ± 0.55 |
| S06 | 82.64 ± 0.41 | 16.85 ± 0.50 | 83.89 ± 0.35 | 87.68 ± 0.40 |
| **Average** | **79.46 ± 0.59** | **19.85 ± 0.78** | **81.10 ± 0.49** | **85.52 ± 0.54** |

**Analysis.** The cross-validation results show that Fold S06 (82.64 BLEU-4) aligns perfectly with the benchmark reported in Table 7, confirming the transparency and consistency of the experiments. Although performance fluctuates due to individual differences in physical characteristics and signing habits (with the lowest score at S04: 77.27), the model maintains an excellent average of 79.46 BLEU-4. This strongly demonstrates the generalisation capacity of the TCN, ensuring that the model can be effectively deployed with new users (unseen signers) in real-world applications.

### 4.6 Qualitative Analysis

Table 11 illustrates representative translation outcomes from the test set, categorised by correctness.

*Table 11: Qualitative translation examples from the test set. ✓ = exact match; ≈ = pragmatically equivalent; ✗ = semantic error.*

| Reference | Hypothesis | Eval. |
| :--- | :--- | :---: |
| Bác sĩ hỏi gia đình tôi có ai bị bệnh này không hả ? | Bác sĩ hỏi gia đình tôi có ai bị bệnh này không hả ? | ✓ |
| Khi nào thì tới lượt tôi ? | Sắp đến lượt tôi chưa ạ ? | ≈ |
| Bác sĩ có thể viết ra giấy được không? | Tôi có cần nộp giấy tờ gì không ? | ✗ |

## 5. Conclusion

This paper introduces VSL-GH, the first sentence-level Vietnamese Sign Language dataset for general healthcare communication based on the Ho Chi Minh City variant. The dataset comprises 8,400 multiview videos from 6 deaf signers, covering 300 sentences with rich multi-level annotations including temporal gloss boundaries, corresponding Vietnamese sentences, and body keypoint features. We establish a signer-independent evaluation protocol and demonstrate that the TCN architecture achieves a BLEU-4 score of 82.64%, significantly outperforming other representative baselines while maintaining high computational efficiency. Experimental results further highlight the critical role of non-manual features and frame-level gloss supervision in advancing translation accuracy. In future work, we plan to expand VSL-GH with more sentences and signers, incorporate additional regional variants of VSL, and work towards real-time sign language translation systems deployable in real-world healthcare settings.

## Acknowledgment

The authors would like to thank the Centre for Research and Education of the Deaf (CED) for their expert support in sign language throughout the gloss preparation, data collection, and annotation quality control processes. We also thank the physicians at Tan Hung General Hospital for their assistance in validating the sentence list against the actual general health examination procedure, ensuring the practical relevance of the dataset. Special thanks are extended to the six deaf participants who generously contributed their time to provide signing data for this research.
