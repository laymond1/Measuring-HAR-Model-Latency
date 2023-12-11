# 실험 코드 정리

## PPT 실험 파트 구성 순서에 따라 코드 설명

### Efficiency 관련 코드
- Model 종류: 
  - Vision / HAR / HARNAS
- Metric 종류:
  - Size(MB) / FLOPs(M) / PC:GPU(ms) / Nano:CPU(s) / Nano:GPU(ms) / Smartphone:A31-CPU(ms)
- Device 종류:
  - PC / Nano / Smartphone
<!-- table -->
| Model  | Size(MB) | FLOPs(M) | PC:GPU(ms) | Nano:CPU(s) | Nano:GPU(ms) | Smartphone:A31-CPU(ms) |
| :----: | :------: | :------: | :--------: | :---------: | :----------: | :--------------------: |
| Vision |    -     |    -     |     -      |      -      |      -       |           -            |
|  HAR   |    -     |    -     |     -      |      -      |      -       |           -            |
| HARNAS |    -     |    -     |     -      |      -      |      -       |           -            |

:sparkles: 대부분의 실험은 쥬피터 노트북에 정리되어 있음.

1. measure metrics: 

  ```python 
>>> python ./ea_harnas/measure_metrics.py --dataset $data --arch EANAS
>>> python ./rl_harnas/measure_metrics.py --dataset $data --arch RLNAS
>>> python ./dnas_harnas/measure_metrics.py --dataset $data --arch OPPA31
```

2. measure latency: 

  ```python 
>>> python ./ea_harnas/measure_latency.py --dataset $data --arch EANAS --num-runs 100 --hardware pc --device gpu
>>> python ./rl_harnas/measure_latency.py --dataset $data --arch RLNAS --num-runs 100 --hardware pc --device gpu
>>> python ./dnas_harnas/measure_latency.py --dataset $data --arch OPPA31 --num-runs 100 --hardware pc --device gpu
```

3. convert the model(or blocks) for smartphone


  ```python 
>>> python ./convert_models/convert_har_mobile.py --dataset $data --arch EANAS
>>> python ./convert_models/convert_harblock_mobile.py --dataset $data --arch EANAS
>>> python ./convert_models/convert_vision_mobile.py --dataset $data --arch RLNAS
>>> python ./convert_models/convert_visblock_mobile.py --dataset $data --arch RLNAS
>>> python ./convert_models/convert_harnas_mobile.py --dataset $data --arch RLNAS
```

---

### Performance(F1-Score) 관련 코드
- Model 종류: 
  - Vision / HAR / HARNAS
- Dataset 종류: 
  - UCI-HAR / WISDM / UniMiB-SHAR / OPPORTUNITY / KU-HAR
<!-- table -->
| Model  | UCI-HAR | WISDM | UniMiB-SHAR | OPPORTUNITY | KU-HAR |
| :----: | :-----: | :---: | :---------: | :---------: | :----: |
| Vision |    -    |   -   |      -      |      -      |   -    |
|  HAR   |    -    |   -   |      -      |      -      |   -    |
| HARNAS |    -    |   -   |      -      |      -      |   -    |


1. HAR Model train
   - run_har 디렉토리의 bash 실행

2. Vision Model train
   - run_vis_{dataset} 디렉토리의 bash 실행
   - 각 데이터셋 별로 따로 디렉토리 구성

3. HARNAS Model train
   - run_harnas 디렉토리의 bash 실행

---

### HARNAS Model
HARNAS의 model들은 기본적으로 OPPORTUNITY 데이터셋에 최적인 모델 활용.

* RL-based NAS
  * OPPORTUNITY MODEL
  * Dataset이 OPPORTUNITY만 활용됨.
  * Pellatt, Lloyd, and Daniel Roggen. “Fast Deep Neural Architecture Search for Wearable Activity Recognition by Early Prediction of Converged Performance.” In 2021 International Symposium on Wearable Computers, 1–6, 2021.

* EA-based NAS
  * OPPORTUNITY MODEL
  * OPPORTUNITY dataset에 최적인 model을 다른 dataset에 directly 적용했다고 논문들에서 언급하고 있음.
  * Wang, Xiaojuan, Xinlei Wang, Tianqi Lv, Lei Jin, and Mingshu He. “HARNAS: Human Activity Recognition Based on Automatic Neural Architecture Search Using Evolutionary Algorithms.” Sensors 21, no. 20 (October 19, 2021): 6927. https://doi.org/10.3390/s21206927.

* DNAS-based NAS
  * OPPORTUNITY A31 MODEL
  * 위 논문들을 따라 OPPORTUNITY을 활용하여 탐색한 모델 제시.
  * Lim, Won-Seon, Wangduk Seo, Dae-Won Kim, and Jaesung Lee. “Efficient Human Activity Recognition Using Lookup Table-Based Neural Architecture Search for Mobile Devices.” IEEE Access 11 (2023): 71727–38. https://doi.org/10.1109/ACCESS.2023.3294564.
