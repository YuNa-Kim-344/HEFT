# HEFT 기반 모델 학습 최적화 실험

### HEFT 란?

**HEFT (Heterogeneous Earliest Finish Time)**는 통신 비용을 고려하여 이기종 환경에서 종속 작업 집합을 스케줄링하는 휴리스틱 알고리즘이다.
Input 으로 DAG(directed acyclic graph), Execution time, Communication Cost (통신 시간)을 받는다.


HEFT 알고리즘은 2단계로 나뉜다.
   1.  Prioritizing tasks : 각 작업에 우선순위가 부여되는 작업한다.
    <img width="1030" height="446" alt="스크린샷 2025-08-07 오후 2 16 46" src="https://github.com/user-attachments/assets/f6bbbe61-5e41-45a0-9230-af2534a77419" />

   2.  Assigning tasks to workers : 설정된 우선순위부터 작업을 최적의 프로세서에 할당한다.
    <img width="862" height="463" alt="스크린샷 2025-08-07 오후 2 17 04" src="https://github.com/user-attachments/assets/4946a0e1-7d0f-4bc5-adc1-15444a1032e3" />


---
### GPU 환경 및 성능

본 프로젝트는 다음과 같은 GPU 환경에서 HEFT을 적용하였다.

- 각 GPU별 연산 성능

    | GPU   | 모델   | Single-Precision (TFLOPs) | Tensor Core (TFLOPs) |
    |:-----:|:------:|:--------------------------:|:---------------------:|
    | GPU 0 | 6000   | 911.1                      | 1457                  |
    | GPU 1 | 3090   | 35.6                       | 285.7                 |
    | GPU 2 | 3090   | 35.6                       | 285.7                 |

- GPU 간 데이터 전송 속도

    | From → To     | 전송 속도 |
    |:-------------:|:---------:|
    | GPU 0 ↔ GPU 1 | 32 GB/s   |
    | GPU 0 ↔ GPU 2 | 32 GB/s   |
    | GPU 1 ↔ GPU 2 | 32 GB/s   |
    
Single-precision 기준으로 연산 성능이 6000이 3090보다 약 2.5배 더 빠르다.
Tensor Core 기준으로는 6000이 3090보다 약 5.1배 더 빠르다.
GPU 간 또는 GPU-CPU 간 데이터 전송 속도는 32 GB/s이다.

---
### 모델 구조  및 연산량

본 시스템은 Llama3-8B-Instruct 모델에 **QLoRA** 및 **FlashAttention**을 적용한 구조를 기반으로 하며, 전체 연산 흐름은 아래의 DAG(Task Graph)로 표현됩니다.

해당 구조는 다음과 같은 주요 레이어로 구성됩니다:
- `Embed`, `Layer0 ~ Layer31`, `Norm`, `Lm_head` 로 이어지는 순차적 흐름
- 각 레이어 간에는 통신량이 명시되어 있으며, 예를 들어 `Layer0 → Layer1` 구간의 통신량은 `5,236,162,560 × 4 bytes`입니다.
- 출력 레이어인 `Lm_head`의 파라미터 수는 약 256,300개로 모델의 최종 출력 단계에 해당합니다.
<img width="1714" height="714" alt="스크린샷 2025-08-07 오후 2 13 45" src="https://github.com/user-attachments/assets/a02551f5-e103-468c-924f-bab01296d9bd" />


각 GPU의 1초당 계산 가능한 연산량을 상대적 값으로 환산하여 표시하였습니다.
- 단정밀도 기준: 6000의 성능을 1, 3090의 성능을 0.4로 설정
→ 해당 설정은 Embed, Norm, Lm_head 연산에 적용됨
- Tensor Core 기준: 6000의 성능을 1, 3090의 성능을 0.2로 설정
→ 해당 설정은 Layer 연산에 적용됨

| 연산항목 (샘플값) | GPU 0: Ada 6000 | GPU 1: RTX 3090 | GPU 2: RTX 3090 |
|:------------------:|:----------------:|:----------------:|:----------------:|
| Embed              | 1                | 0.4              | 0.4              |
| Layer 0            | 106500           | 21300            | 21300            |
| ...                | ...              | ...              | ...              |
| Layer 31           | 106500           | 21300            | 21300            |
| Norm               | 2                | 0.8              | 0.8              |
| Lm_head            | 256300           | 102520           | 102520           |

---
### HEFT 알고리즘 실험

##### HEFT 적용하지 않고 Auto로 사용할 경우 실제 학습 시간 
- Rank, Alpha = 32, epoch = 6, batch size = 5, Quantization(o)
- GPU 3개에 Auto 로 할당할 경우, `5시간 29분 30초`

    - Embed ~ layer1 -> cuda : 0 (Ada 600) 
    - layer2 ~ layer15 -> cuda : 1 (RTX 3090)
    - layer16 ~ lm_head -> cuda : 2 (RTX 3090)

##### HEFT 알고리즘을 적용한 학습 시간

HEFT 적용 시, 모든 레이어는 성능이 가장 우수한 GPU 0(Ada 6000)에 배정되었다.
<img width="335" height="343" alt="image" src="https://github.com/user-attachments/assets/7306f0fd-b256-4c21-801b-a1440a5465b9" />
<img width="333" height="330" alt="image" src="https://github.com/user-attachments/assets/5ef943a1-7e3b-4fdb-8bea-657ecbc575eb" />




알고리즘을 통해 나온 결과로 Layer를 할당하였을 때,`2시간 49분 22초`가 걸렸다.

---
### 결과

결과적으로 HEFT 알고리즘을 적용하지 않고 AUTO로 Layer를 할당하여 학습 하는 것보다 HEFT 알고리즘을 통한 Layer 할당이 약 3시간 정도 빠른 것으로 확인 하였다.

---
### 코드 검증
본 연구에서 구현한 HEFT 알고리즘 코드의 정확성을 검증하기 위해,
HEFT 알고리즘 논문 *“Performance-Effective and Low-Complexity Task Scheduling for Heterogeneous Computing”*에 제시된 예제를 기준으로 테스트를 진행하였다.

논문의 예제의 Layer는 다음 이미지 와 같다.
<img width="907" height="414" alt="스크린샷 2025-08-07 오후 2 12 59" src="https://github.com/user-attachments/assets/08e4a64f-991e-4ce8-883c-0ea94b724fcd" />


결과로는 다음과 같이 할당 되는 것을 확인할 수 있다.
- P1 -> Task 2, Task 8
- P2 -> Task 4, Task 6, Task 9, Task 10
- P3 -> Task 1, Task 3, Task 5, Task 7
<img width="199" height="319" alt="image" src="https://github.com/user-attachments/assets/a88c0df5-b973-459d-ada6-5f69cffcf24e" />


본 연구의 HEFT 알고리즘 코드에 동일한 예제를 적용하였을때, 다음과 같이 동일한 결과가 출력되는 것을 확인하였다.
이에 본 연구의 HEFT 알고리즘 코드가 적절하다는 것을 확인하였다.
<img width="288" height="326" alt="image" src="https://github.com/user-attachments/assets/5fe3618b-6a1e-4ae7-990c-ad5dab917ca2" />















