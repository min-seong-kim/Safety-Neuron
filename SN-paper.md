# Safety-Specific Neurons in LLMs: Methodology-Centric Summary
(ICLR 2025)

## 1. 핵심 문제 설정

기존 LLM 안전성 연구의 근본적 한계는 다음과 같다.

- 안전성(safety)을 **모델 전체의 분산된 성질**로 취급
- layer-level, module-level 분석 → 너무 거칠어 실질적 제어 불가
- 안전 파인튜닝은 대규모 데이터·연산 비용 요구
- downstream fine-tuning 시 안전성 붕괴 현상에 대한 구조적 설명 부족

본 논문은 다음 질문을 중심으로 접근한다.

> **“LLM의 안전성은 실제로 어디에, 어떤 단위로 구현되어 있는가?”**

이에 대한 답으로 **Safety-Specific Neuron** 개념과 이를 조작하는 두 가지 기법  
**SN-Tune** 및 **RSN-Tune**를 제안한다.

---

## 2. Safety Neuron 탐지 기법 (Core Method)

### 2.1 Neuron 단위 정의

- Neuron = Transformer 파라미터 행렬의 **단일 row 또는 column**
  - FFN: W_up, W_down의 column
  - Attention: W_Q, W_K, W_V의 column
- 즉, activation space가 아니라 **parameter-space neuron**

→ 기존 activation clustering이나 feature attribution보다 **훨씬 미시적인 단위**

---

### 2.2 Neuron 중요도 정의 (Query-conditional)

유해 질의 \(x\)에 대해,  
레이어 \(i\)의 뉴런 \(N_i^{(l)}\) 중요도를 다음과 같이 정의:

\[
\text{Imp}(N_i^{(l)} \mid x)
=
\|h_i(x) - h_{i \setminus N_i^{(l)}}(x)\|_2
\]

- 해당 뉴런을 제거했을 때 **중간 표현 변화량**
- 출력 token이 아니라 **hidden representation 변화** 기준
- 라벨 불필요 (unsupervised)

중요한 점:
- “유해 답변을 했는가?”가 아니라
- “유해 질의를 처리하는 과정에서 이 뉴런이 필수적인가?”

---

### 2.3 Safety Neuron 집합 정의 (Consistency Criterion)

유해 질의 집합 \(X = \{x_1, \dots, x_n\}\)에 대해:

1. 각 질의별 활성 뉴런 집합:
\[
N_x = \{ N \mid \text{Imp}(N \mid x) \ge \epsilon \}
\]

2. **모든 유해 질의에서 공통으로 중요한 뉴런**만 선택:
\[
N_{\text{safe}} = \bigcap_{x \in X} N_x
\]

핵심 철학:
- 특정 jailbreak에만 반응하는 뉴런 제거
- **“안전 메커니즘 자체”를 구성하는 뉴런만 남김**

---

### 2.4 병렬 가속화 기법 (Practical Contribution)

#### FFN
- 마스킹 행렬을 이용해 모든 neuron importance를 **한 번의 forward로 계산**
- W_up / W_down 뉴런 제거 효과의 수학적 동치성 증명

#### Self-Attention
- W_Q, W_K 뉴런 제거가 attention score에 미치는 영향을 근사
- softmax 이전 attention logit 차이를 기반으로 importance 계산

→ 기존 neuron ablation 대비 **수백 배 이상 효율적**

---

## 3. SN-Tune (Safety Neuron Tuning)

### 3.1 설계 철학

관찰된 사실:
- Safety Neuron은 전체 파라미터의 **< 1%**
- 안전성 붕괴는 이 뉴런들만 건드려도 발생
- 일반 성능은 대부분 **다른 뉴런에 저장**

→ 그렇다면:
> **“안전성은 safety neuron만 학습시키면 충분하다.”**

---

### 3.2 SN-Tune 알고리즘

1. Safety Neuron 탐지 (Section 2)
2. 파인튜닝 시:
   - Safety Neuron 파라미터: **gradient 허용**
   - 나머지 파라미터: **gradient = 0 (완전 고정)**

수식적으로:
\[
\nabla \theta_j =
\begin{cases}
\nabla \theta_j & \text{if } \theta_j \in N_{\text{safe}} \\
0 & \text{otherwise}
\end{cases}
\]

---

### 3.3 학습 설정의 특징

- 데이터: 약 **50개 안전 문서**
- Epoch: **1**
- Learning rate: **1e-6**
- Chat template 불필요 (base model도 가능)

→ “계속학습(continue training)” 수준의 미세 조정

---

### 3.4 효과 및 의미

- Instruction-tuned model:
  - Harmful score 60~90 → **2~5**
  - MMLU / GSM8K 성능 유지
- Base model:
  - 기존에 없던 safety mechanism을 **후설치**

중요한 해석:
- 안전성은 **모델 전체의 emergent property가 아니라**
- **극소수 뉴런에 국소화된 기능적 회로**

---

## 4. RSN-Tune (Robust Safety Neuron Tuning)

### 4.1 문제 인식

기존 연구(Qi et al., 2024):
- GSM8K 같은 “무해한” 데이터로 fine-tuning해도
- LLM 안전성은 심각하게 붕괴됨

본 논문의 해석:
- 원인은 **Safety Neuron ↔ Foundation Neuron 중첩**

---

### 4.2 Foundation Neuron 개념

- Foundation Neuron:
  - 일반적인 질의 처리 능력의 핵심 뉴런
  - Wikipedia corpus 기반 동일한 neuron detection 방식으로 추출

즉:
- Safety Neuron: “유해 질의 대응”
- Foundation Neuron: “모든 질의 처리의 골격”

---

### 4.3 RSN-Tune 핵심 아이디어

Safety Neuron 집합 분해:
\[
N_{\text{safe}} =
(N_{\text{safe}} \cap N_{\text{foundation}})
\;\cup\;
(N_{\text{safe}} \setminus N_{\text{foundation}})
\]

RSN-Tune는 다음만 학습:
\[
N_{\text{robust}} = N_{\text{safe}} \setminus N_{\text{foundation}}
\]

→ **task learning에 필수적인 safety neuron은 보호**
→ **downstream fine-tuning 시 안전성 유지**

---

### 4.4 RSN-Tune 절차

1. Safety Neuron 탐지
2. Foundation Neuron 탐지
3. 두 집합의 차집합 계산
4. SN-Tune을 **해당 subset에만 적용**

---

### 4.5 효과 및 한계

- GSM8K fine-tuning 후:
  - Harmful score 증가 폭 **대폭 완화**
- 완전한 안전성 유지(0.0)는 불가
  - non-overlapping safety neuron 수가 제한적

→ 중요한 기여:
- **안전성 붕괴의 구조적 원인을 처음으로 뉴런 수준에서 설명**
- 안전성–성능 trade-off를 정량적으로 분해 가능

---

## 5. 기법 간 관계 요약

| 구성 요소 | 역할 |
|---------|-----|
| Safety Neuron Detection | 안전 메커니즘의 최소 단위 식별 |
| SN-Tune | 안전성 강화 / 안전성 설치 |
| RSN-Tune | downstream fine-tuning 시 안전성 보존 |

---

## 6. 핵심 메시지 (Methodological Takeaway)

1. LLM 안전성은 **희소하고(localized)**, **조작 가능**
2. 안전성 튜닝은 **대규모 데이터가 필요하지 않음**
3. “안전성 붕괴”는 우연이 아니라 **뉴런 중첩의 필연적 결과**
4. Safety tuning은 **parameter-efficient fine-tuning의 새로운 축**

---

## 7. 연구적 확장 포인트

- Safety neuron 기반 **LoRA / Adapter 분리**
- Safety neuron을 invariant constraint로 둔 continual learning
- 공격 유형별 safety neuron clustering
- ICS / Text-to-SQL 등 도메인 안전성으로의 확장
