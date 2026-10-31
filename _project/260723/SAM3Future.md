---
layout: default
title: "Predicting future frames using object-centric video representations"
permalink: /260723/SAM3Future/
nav: false
---

<link rel="stylesheet" href="{{ '/assets/css/segjoint.css' | relative_url }}">

<article class="segjoint-page" markdown="1">

# Predicting future frames using object-centric video representations

\[학회명: International Conference on Advanced Machine Learning and Data Science (AMLDS) / 게재일: 2026.07.21\]

- SAM3Future: Predicting future frames using object-centric video representations
- 단독 1저자 학술대회 포스터 발표 진행
- [Paper](/assets/pdf/SAM3Future.pdf){:target=\"_blank\" rel=\"noopener noreferrer\"} / [Code](https://github.com/onebom)

---

## Abstract

Video prediction은 과거 영상으로부터 미래 프레임을 예측하는 과제로, 미래 변화의 불확실성으로 인해 예측에 어려움이 있다. 
이를 다루기 위해 최근 일부 방법은 optical flow와 같은 보조 표현을 활용하여 미래 움직임 예측과 움직임에 조건화된 영상 생성을 분리하여 학습한다. 
그러나 미래 객체 영역을 잘못 예측하거나 생성 과정에서 서로 다른 객체의 정보가 혼합되면, 객체의 외형이 왜곡되거나 사라지는 다중 객체 왜곡이 발생할 수 있다. 
본 연구에서는 이러한 문제를 완화하기 위해 사전 학습된 SAM3의 객체 표현을 활용하여 객체별 미래 영역 예측과 미래 프레임 생성을 공동으로 학습하는 SAM3Future 프레임워크를 제안한다. 
SAM3Future는 예측한 미래 객체 embedding을 cross-attention으로 주입하고 해당 attention score에 segmentation supervision을 부여하며, 예측 영역을 기반으로 디코딩 과정에 instance-masked attention을 적용하여 서로 다른 객체의 정보 혼합을 억제한다. 
Cityscapes와 KTH에서 기존 방법 대비 경쟁력 있는 예측 성능을 보였으며, ablation study를 통해 영역 supervision과 마스크 기반 디코딩의 추가적인 기여를 확인하고 정성적 비교에서 객체의 형태와 위치가 더 잘 유지되는 양상을 관찰하였다.

## Introduction

Video prediction은 과거에 관측된 프레임들로부터 이후에 나타날 미래 프레임들을 예측하는 과제다. 
동일한 과거 동영상이 주어져도 여러 미래가 가능하므로, 모델은 미래의 불확실성이 반영된 복잡한 조건부 영상 분포를 학습해야 한다. 
이러한 분포를 효과적으로 학습하기 위해 최근 video prediction 방식은 학습분포를 미래 움직임의 분포와 해당 움직임에 조건화된 영상 분포를 분리하여 학습한다. 
구체적으로 optical flow와 같은 장면 단위의 미래 움직임을 나타내는 보조 표현을 먼저 예측하고, 이를 바탕으로 미래 프레임들을 생성한다. 
다만 이 구조에서는 보조 표현의 예측 오류가 최종 영상 생성 품질에 영향을 미칠 수 있다.

### <b>Problem statement</b>

1. <b>Multi object distortion</b>

   \[figure/동영상\]

   기존 방식은 미래 움직임을 나타내는 보조 표현을 학습하지만, 이를 객체 단위로 분리하여 예측하지 않으며 미래 동영상 생성 과정에서 객체별 공간 영역을 명시적으로 구분하도록 하는 제약도 부족하다. 
   따라서 여러 객체의 외형과 상대적 위치가 복잡하게 변하는 장면에서는 객체별 미래 움직임을 정확하게 추정하기 어려울 수 있다. 
   또한 객체들의 영역이 근접한 경우, 생성된 영상에서 서로 다른 객체의 외형이 혼합되거나 일부 객체가 사라지는 <b>다중 객체 왜곡</b>이 발생할 수 있다. 
   1. 미래 객체 영역을 잘못 예측하면 서로 다른 객체의 표현이 동일한 공간에 반영되어, 객체의 외형이 혼합되거나 왜곡될 수 있다. 
   2. 미래 객체 영역이 정답에 가깝게 예측된 경우에도, 동영상 생성 과정에서 해당 영역에 다른 객체의 표현이 함께 반영되면 다중 객체 왜곡이 발생할 수 있다. 

## Method

본 연구에서는 대규모 데이터로 사전 학습된 SAM3의 객체 표현을 활용하여 객체별 미래 영역을 예측하고, 이를 기반으로 디코딩 과정의 attention을 제한하여 각 영역에 해당하지 않는 객체의 표현이 혼합되는 것을 억제하는 SAM3Future 프레임워크를 제안한다.

### Background

본 연구에서는 EDM 프레임워크를 통해 denoising 모델을 최적화했고, 모델의 기본 구조는 Hourglass Diffusion Transformer(HDiT)를 참고하여 동영상 대상의 denoiser를 구현했다. 
또한 객체 영역 예측과 프레임 간 동일 객체 추적을 사전 학습한 표현을 활용하기 위해 SAM3의 object pointer embedding을 사용하였다.

- SAM3

  SAM3는 동영상에서 객체를 검출·분할·추적하는 통합 모델이다. 
  detector가 프레임별 객체를 검출하고, 메모리 기반 tracker가 과거 객체 정보를 활용하여 현재 프레임의 마스크를 예측한다.

  ![SAM3 overview](/assets/img/sam3future/sam3-overview.png)

  Tracker 구조는 SAM2를 계승하여 현재 프레임의 feature가 과거의 spatial memory와 object pointer를 참조하도록 한 뒤, mask decoder로 현재 객체 영역을 예측한다. 

  ![SAM3 tracker and object pointer](/assets/img/sam3future/sam3-tracker.png)

  Object pointer는 mask decoder에서 프레임 feature 및 prompt 정보와의 attention을 통해 얻어진 객체 수준의 백터 표현이다. 
  해당 표현은 예측 마스크에 대응하는 객체 정보를 담으며, spatial memory와 함께 memory bank에 저장되어 이후 프레임에서 동일 객체의 마스크를 예측하는 데 활용된다. 

### SAM3Future

SAM3Future는 사전 학습된 SAM3의 객체 표현을 활용하여 객체별 미래 영역 예측과 미래 프레임 생성을 공동으로 학습한다. 미래 프레임의 디코딩 과정에서는 예측한 마스크를 기반으로 instance-masked attention을 적용하여, 객체별 공간 영역에 따라 feature 간 참조를 제한한다.

\[overview 그림 : \]

구체적으로 SAM3를 이용한 과거 영상의 segmentation 과정에서 객체별 pointer embedding을 추출하고, 이를 외삽하여 미래 프레임별 객체 embedding을 추정한다. 
추정한 embedding은 미래 프레임의 denoising feature와의 cross-attention을 통해 생성 과정에 객체별 조건 정보로 반영한다. 
동시에 두 표현 간의 attention score로 미래 객체 영역을 예측하고 segmentation supervision을 부여한다. 
예측한 객체별 영역 확률 맵은 이진 마스크로 변환하여 디코더 초기 단계의 instance-masked attention에 사용한다. 
이를 통해 각 객체 영역의 feature가 동일 객체에 할당된 영역의 feature만 참조하도록 제한하여, 해당 영역에 속하지 않는 객체의 정보가 혼합되는 것을 억제한다. 

<details markdown="1">
<summary><b>SAM3 object representation conditioned Denoising</b></summary>

1. <b>Instance Mask and Embedding Extraction</b>

   사전학습된 SAM3를 활용하여 첫 프레임에서 객체를 검출하여 bounding box를 얻고, 프레임별 instance mask와 그에 대응되는 객체 임베딩을 추출한다. 
   미래 프레임의 마스크는 학습 시 segmentation supervision을 제공하는 라벨로만 사용한다. 
   또한 과거 객체 embedding에 학습 가능한 배경 embedding을 추가하여, 객체로 정의되지 않은 영역(배경)도 함께 표현하도록 한다.

2. <b>Future Instance Embedding Estimation</b>

   \[figure\]

   객체와 배경 embedding으로 구성된 과거 시퀀스를 다단계 Transformer 기반 예측기에 입력한다. 
   각 단계는 temporal attention과 object-wise attention을 통해 시간적 변화와 객체 간 관계를 반영하며, 입력 시퀀스와 동일한 길이의 후속 embedding을 예측한다. 
   예측한 embedding을 기존 시퀀스 뒤에 시간축으로 연결하여 시퀀스 길이를 두 배로 확장한다. 필요한 미래 구간을 확보할 때까지 이 과정을 반복한 뒤, 과거 구간을 제외하고 목표 미래 프레임에 대응하는 embedding을 추출한다.

3. <b>Instance Embedding Conditioning</b>

   예측한 미래 객체 및 배경 embedding은 residual cross-attention block을 통해 denoising feature에 반영한다. 먼저 중간 denoising feature를 2D convolution 기반 refiner에 통과시켜, 공간 위치와 객체 embedding 사이의 대응 관계를 학습하는 데 적합한 표현으로 변환한다. 
   이후 각 미래 프레임에서 정제된 공간 feature를 query로, 해당 프레임의 객체 및 배경 embedding을 key와 value로 사용하는 single-head cross-attention을 수행한다. Attention 출력을 MLP에 통과시킨 뒤 원래의 denoising feature에 residual 방식으로 더하여 객체별 조건 정보를 반영한다.

</details>

<details markdown="1">
<summary><b>Segmentation-guided Joint Learning</b></summary>

- <b>Future Instance Mask Prediction</b>

  미래 객체 영역은 cross-attention의 softmax 이전 attention score로부터 예측한다. Query인 정제된 denoising feature와 key인 객체 embedding 사이의 correlation map을 마스크 해상도로 upsampling하고, sigmoid를 적용하여 객체별 영역 확률 맵을 얻는다. 
  예측한 확률 맵에 미래 instance mask를 학습용 라벨로 사용하는 segmentation supervision을 부여하여, 미래 프레임의 공간 위치와 객체 embedding 사이의 대응 관계를 학습하도록 한다.

- <b>Joint Training Objective</b>

  학습에는 미래 프레임 생성을 위한 diffusion loss와 미래 객체 영역 예측을 위한 segmentation loss를 함께 사용한다. Segmentation loss는 focal loss와 Dice loss로 구성하며, 예측한 객체별 영역 확률 맵과 해당 instance mask 라벨 사이에서 계산한다. 
  이러한 공동 학습을 통해 객체 embedding이 생성 조건으로 활용되는 동시에 해당 객체의 미래 공간 영역과 연결되도록 유도한다.

</details>

<details markdown="1">
<summary><b>Instance-masked Decoding</b></summary>

1. <b>Binary Mask Construction</b>

   객체별 attention score 맵에 Gumbel-Sigmoid를 적용한다. 그 출력을 0.5 기준으로 이진화하여 instance-masked attention에 사용할 마스크를 생성한다. 
   학습 시에는 straight-through estimator를 사용하여 forward에서는 이진 마스크를 적용하고, backward에서는 연속적인 Gumbel-Sigmoid 출력을 통해 gradient를 전달한다.

2. <b>Instance-masked Attention</b>

   예측한 객체별 이진 마스크로 동영상 token 간 attention mask를 구성하고, 디코더 초기 단계에 해당하는 middle block에만 적용한다. 
   <br />각 query token은 자신과 하나 이상의 객체 영역을 공유하는 key token만 참조하도록 제한한다.<br /> 
   이 제약은 spatial attention에서는 동일 프레임 내에, spatio-temporal attention에서는 해당 block이 처리하는 프레임들에 걸쳐 적용한다.

</details>

## Result

### <b>Quantitative Results</b>

   <b>\[Comparison between baselines\]</b>

   ![Benchmark comparison results](/assets/img/sam3future/benchmark-results.png)

   SAM3Future는 제시된 비교 방법 중 Cityscapes에서 PSNR, LPIPS, FVD가 가장 우수했으며, KTH에서는 네 지표 모두에서 가장 좋은 성능을 보였다. 이는 객체 표현과 미래 영역 예측을 활용하는 제안 방법이 다양한 장면에서 미래 프레임의 예측 품질을 개선할 수 있음을 보여준다.

   ![Additional benchmark results](/assets/img/sam3future/benchmark-details.png)

   <b>\[ablation study 결과 on Cityscapes dataset\]</b>

   | SAM3 embedding conditioning | Segmentation objective | Instance-masked attention | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FVD ↓ |
   | --- | --- | --- | --- | --- | --- | --- |
   | ✗ | ✗ | ✗ | 23.52 | 0.713 | 0.090 | 85.56 |
   | ✓ | ✗ | ✗ | 23.61 | 0.718 | 0.091 | 84.53 |
   | ✓ | ✓ | ✗ | 24.19 | 0.739 | 0.084 | 80.00 |
   | ✓ | ✓ | ✓ | <b>24.31</b> | <b>0.747</b> | <b>0.082</b> | <b>78.4</b> |

   SAM3 embedding conditioning만 적용한 경우 PSNR, SSIM, FVD는 개선되었지만 LPIPS는 소폭 악화되었다. 여기에 segmentation objective를 추가하자 모든 지표가 개선되었으며, instance-masked attention까지 적용한 최종 모델이 가장 좋은 성능을 보였다. 이는 객체 표현을 조건으로 제공하는 것에 더해, 객체별 미래 영역을 학습하고 이를 디코딩 과정의 참조 제한에 활용하는 것이 추가적인 예측 품질 개선에 기여함을 보여준다.

### <b>Qualitative Results</b>
   ![Cityscapes qualitative comparison](/assets/img/sam3future/cityscapes-results.png)

   Cityscapes에서는 비교 모델에서 나타나는 차량 주변의 형태 왜곡과 도로 영역의 흐림이 상대적으로 완화되었으며, 차량의 외형과 차선 구조가 더 선명하게 유지되는 양상을 관찰하였다.

   ![KTH qualitative comparison](/assets/img/sam3future/kth-results.png)

   KTH에서는 인물의 위치와 보행 자세가 정답 프레임에 더 가깝게 나타났다. 이러한 결과는 제안 방법이 객체의 외형과 움직임을 유지하는 데 도움이 될 수 있음을 시사한다. 

### Conclusion

본 연구에서는 SAM3의 객체 표현을 활용하여 미래 객체 영역 예측과 프레임 생성을 공동으로 학습하는 SAM3Future를 제안하였다. 예측한 영역을 기반으로 instance-masked attention을 적용하여, 해당 영역에 속하지 않는 객체의 정보가 혼합되는 것을 억제하고자 하였다. 
여러 데이터셋의 정량적 비교에서 경쟁력 있는 성능을 보였으며, ablation study를 통해 영역 supervision과 마스크 기반 디코딩의 추가적인 기여를 확인하였다. 정성적 결과에서도 객체의 형태와 위치가 더 잘 유지되는 양상을 관찰하였다. 이러한 결과는 객체별 영역 학습과 attention 제약을 결합하는 접근이 미래 동영상 예측의 품질을 개선하는 유효한 방향임을 보여준다.


</article>
