---
layout: default
title: "Learning to Segment for Better Diffusion-Based Video Prediction"
permalink: /260204/SegJoint/
nav: false
---

<link rel="stylesheet" href="{{ '/assets/css/segjoint.css' | relative_url }}">

<article class="segjoint-page" markdown="1">

# Learning to Segment for Better Diffusion-Based Video Prediction

\[학회명: Workshop on Image Processing and Image Understanding (IPIU) / 게재일: 2026.02.04 \]

- <b>“SegJoint: Motion-Free Segmentation Conditioning for Video Prediction”</b>
- 공동 1저자로 학술대회 포스터 발표 진행
- [Paper](/assets/pdf/SegJoint.pdf){:target=\"_blank\" rel=\"noopener noreferrer\"} / [Code](https://github.com/onebom)

---

## Abstract

Video prediction은 과거 영상으로부터 미래 프레임을 예측하는 과제로, 미래 불확실성으로 예측에 어려움이 있다. 이를 다루기 위해 최근 방식은 optical flow와 같은 보조 표현을 활용하여 미래 움직임 예측과 움직임에 조건화된 영상 생성을 분리하여 학습한다. 
그러나 객체별로 움직임을 예측하지 않고 공간 영역을 명시적으로 구분하는 제약이 부족하여, 여러 객체가 근접하거나 복잡하게 움직이는 장면에서 객체의 외형이 혼합되거나 사라지는 다중 객체 왜곡이 발생할 수 있다.
본 연구에서는 이러한 문제를 완화하기 위해 미래 프레임 생성과 객체별 미래 영역 예측을 공동으로 학습하여, 각 객체의 표현이 예측한 객체 영역에 반영되도록하는 SegJoint 프레임워크를 제안한다.
SegJoint는 과거 instance mask를 기반으로 추출한 객체 임베딩을 cross-attention으로 주입하고, 해당 attention score에 segmentation supervision을 부여하여 객체별 공간 구분을 유도한다.
Cityscapes에서 baseline 대비 PSNR·SSIM·LPIPS가 모두 개선되었으며, ablation study와 정성적 비교를 통해 제안 구성 요소의 성능 기여와 다중 객체 왜곡의 완화 양상을 확인하였다.

## Introduction

Video prediction은 과거에 관측된 동영상으로부터 이후에 나타날 미래 동영상을 예측하는 과제다. 
동일한 과거 동영상이 주어져도 여러 미래가 가능하므로, 모델은 미래의 불확실성이 반영된 복잡한 조건부 동영상 분포를 학습해야 한다. 
이러한 분포를 효과적으로 학습하기 위해 최근 video prediction 방식은 미래 움직임의 분포와 해당 움직임에 조건화된 동영상 분포를 분리하여 학습한다. 
구체적으로 optical flow와 같은 장면 단위의 미래 움직임을 나타내는 보조 표현을 먼저 예측하고, 이를 바탕으로 미래 프레임들을 생성한다. 
다만 이 구조에서는 보조 표현의 예측 오류가 최종 영상 생성 품질에 영향을 미칠 수 있다.

### Problem statement

1. <b>Multi object distortion</b>

   \[figure/동영상\]

   기존 방식은 미래 움직임을 나타내는 보조 표현을 학습하지만, 이를 객체 단위로 분리하여 예측하지 않으며 미래 프레임 생성 과정에서 공간 영역 별 특징이 객체 별로 구분하도록 하는 제약이 없다. 
   따라서 여러 객체의 외형과 상대적 위치가 복잡하게 변하는 장면에서는 객체별 미래 움직임을 정확하게 추정하기 어려울 수 있다. 
   또한 생성 과정에서 객체들의 영역이 근접한 경우, 서로 다른 객체의 외형이 혼합되거나 일부 객체가 사라지는 <b>다중 객체 왜곡</b>이 발생할 수 있다.

## Method

따라서 본 연구에서는 객체별 미래 영역 예측과 미래 프레임 생성을 공동으로 학습하여, 각 객체의 표현이 해당 예측 영역에만 반영되도록 유도하는 SegJoint 프레임워크를 제안한다.

### Background

- <b>Conditional Diffusion based video prediction model</b>

  조건부 video prediction에서는 diffusion model을 사용하여 과거 프레임을 조건으로 denoising 과정에서 이 관측 정보와 부합하는 미래 프레임을 생성하도록 학습한다.

  본 연구에서는 EDM 프레임워크를 통해 denoising 모델을 최적화했고, 모델의 기본 구조는 Hourglass Diffusion Transformer(HDiT)를 참고하였다.

  ![HDiT architecture](/assets/img/segjoint/hdit-architecture.png)

  HDiT는 Transformer block을 U-Net과 유사한 계층 구조로 구성한다. 인코더에서는 공간적으로 인접한 토큰을 병합하여 해상도를 낮추고, 디코더에서는 토큰을 분할하여 해상도를 복원한다. 또한 대응하는 해상도의 인코더와 디코더 사이에 skip connection을 두어, 인코더의 feature를 디코더에 전달한다.

  본 연구에서는 이미지 생성을 위한 HDiT 구조로 동영상의 시간적 관계도 학습하기 위해 neighborhood attention과 global attention을 사용하는 block에 spatio-temporal attention module을 추가하였다.

### SegJoint

SegJoint는 객체 표현을 생성 과정에 주입하는 cross-attention에 segmentation objective를 적용하여, 객체별 미래 영역을 예측하고 해당 영역에만 객체표현이 적용되도록 유도하는 프레임워크다.

\[overview 그림 : \]

구체적으로 과거 프레임의 segmentation mask를 기반으로 객체별 embedding을 추출하고, 이를 key와 value로, 미래 프레임의 denoising feature를 정제한 표현을 query로 사용하여 cross-attention을 수행한다. 

이때 softmax 이전 attention score로 미래 객체 영역을 예측하고 segmentation supervision을 부여함으로써, 객체별 조건 정보를 반영하는 attention이 미래 영역 예측에도 활용되도록 공동으로 학습한다.

<details markdown="1">
<summary><b>Instance-conditioned Denoising</b></summary>

1. <b>Instance Mask Extraction</b>

   사전학습된 YOLO와 UniDVPS를 활용하여 첫 프레임에서 객체를 검출하여 bounding box를 얻고, 프레임별 instance mask를 추출한다.

   과거 프레임의 마스크는 객체 embedding을 인코딩하는 데 사용하고, 미래 프레임의 마스크는 학습 시 segmentation supervision을 제공하는 라벨로 사용한다.

2. <b>Mask-based Instance Embedding</b>

   \[algorithm\]

   과거 프레임들의 instance mask를 denoising feature 해상도로 bilinear downsampling하고, 마스크 값을 가중치로 공간축 pooling을 수행한 뒤 MLP를 적용하여 프레임별 embedding을 추출한다. 이후 시간축 average pooling과 MLP를 통해 프레임별 embedding을 통합하여 최종적으로 instanace 별 하나의 vector embedding을 얻는다.

3. <b>Instance Conditioning Module</b>

   추출한 instance embedding은 residual cross-attention block을 통해 denoising feature에 반영한다.

   먼저 중간 denoising feature를  2D convolution 기반 refiner에 통과시킨다.

   정제된 feature를 query로, instance embedding을 key와 value로 사용하여 single-head cross-attention을 수행한다. 

   이후 attention 출력을 MLP에 통과시키고 원래의 denoising feature에 residual 방식으로 더하여 객체별 조건 정보를 반영한다.

   이때, denoising feature를 segmentation 학습에 적합한 feature 공간으로 변환하기 위해 refiner를 설계했다.

</details>

<details markdown="1">
<summary><b>Segmentation-guided Joint Learning</b></summary>

- <b>Future Instance Mask Prediction</b>

  미래 객체 영역은 cross-attention의 softmax 이전 attention score로부터 예측한다. 

  Query인 정제된 denoising feature와 key인 객체 임베딩 사이의 correlation map을 마스크 해상도로 upsampling하고 sigmoid를 적용하여 객체별 영역 확률 맵을 얻는다. 

  예측한 마스크에 미래 instance mask를 정답으로 사용하는 segmentation supervision을 부여하여, 공간 위치와 객체 embedding 사이의 대응 관계를 학습하도록 한다.

- <b>Joint Training Objective</b>

  학습에는 미래 프레임 생성을 위한 diffusion loss와 미래 객체 영역 예측을 위한 segmentation loss를 동시에 사용한다.  

  Segmentation loss는 focal loss와 Dice loss로 구성하며, 예측한 미래 마스크와 해당 instance mask 라벨 사이에서 계산한다. 

  이러한 공동 학습을 통해 객체 embedding이 생성 조건으로 활용되는 동시에 해당 객체의 미래 영역과 연결되도록 유도한다.

</details>

## Result
### <b>Quantitative Results</b>

   <b>\[Comparison between baselines\]</b>

   | Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
   | --- | --- | --- | --- |
   | WALDO | 22.40 | 0.653 | 0.147 |
   | SegJointVP | <b>23.76</b> | <b>0.719</b> | <b>0.093</b> |

   실험은 Cityscapes 데이터셋에서 128 × 128 해상도의 과거 4프레임을 입력으로 사용하여 미래 16프레임을 예측하는 설정으로 수행하였다.

   동일한 benchmark 설정에서 WALDO와 비교한 결과, SegJointVP는 PSNR, SSIM, LPIPS 모두에서 높은 성능 개선을 보였다.

   <b>\[ablation study 결과\]</b>

   | Instance conditioning | Future segmentation loss | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FVD ↓ |
   | --- | --- | --- | --- | --- | --- |
   | ✗ | ✗ | 23.34 | 0.708 | 0.097 | 106.5 |
   | ✓ | ✗ | 23.51 | 0.713 | 0.096 | 105.6 |
   | ✓ | ✓ | <b>23.76</b> | <b>0.719</b> | <b>0.093</b> | <b>100.62</b> |

   Instance conditioning과 future segmentation loss의 기여를 확인하기 위해 두 요소를 단계적으로 추가하여 성능을 비교하였다.

   Instance conditioning만 적용한 경우에도 baseline 대비 모든 지표에서 소폭의 성능 개선이 나타났다. 이는 과거 영상에서 추출한 객체별 표현이 미래 프레임 생성에 유용한 조건 정보로 활용될 수 있음을 보여준다.

   여기에 future segmentation loss를 추가한 최종 모델은 PSNR 23.76 dB, SSIM 0.719, LPIPS 0.093, FVD 100.62를 기록하여 비교한 구성 중 모든 지표에서 가장 좋은 성능을 보였다. 이러한 결과는 객체 표현을 생성 조건으로 제공하는 것에 더해, 해당 표현과 미래 공간 영역 사이의 대응 관계를 사용해 객체 별 미래 영역을 예측하도록 유도하는 supervision이 추가적인 예측 품질 개선에 기여함을 보여준다.

### <b>Qualitative Results</b>

![SegJoint qualitative comparison](/assets/img/segjoint/qualitative-results.png)

정성적 비교에서는 segmentation loss의 유무에 따른 생성 결과를 살펴보았다. 

Segmentation loss를 적용한 경우, 객체의 미래 위치와 모양이 정답 영상에 더 가깝게 예측되는 양상을 관찰하였다. 따라서, 잘못된 영역 예측으로 객체들이 중첩되는 부분에서 서로 다른 객체의 외형이 혼합되거나 왜곡되는 현상이 완화되었다.

이러한 관찰은 객체의 미래 영역의 예측 오류는 서로 다른 객체의 표현을 동일한 영역에 반영하게 되고, 객체 간 경계가 흐려지거나 형태가 불분명해지는 왜곡이 심화될 수 있음을 시사한다.

또한 미래 객체 영역에 대한 supervision이 더 정확한 객체별 영역 예측과 공간적 구분을 유도함으로써, 다중 객체 왜곡을 완화하는 데 도움이 될 수 있음을 시사한다.

### Conclusion & Futurework

본 연구에서는 객체의 미래 영역을 예측을 미래 프레임 생성 과정에 결합하여 객체의 특징이 예측한 영역에만 반영되도록 제약하는 SegJoint 프레임워크를 제안하였다. Cityscapes benchmark에서 WALDO 대비 PSNR, SSIM, LPIPS가 모두 개선되었으며, ablation study를 통해 instance conditioning과 future segmentation loss가 각각 예측 품질 개선에 기여함을 확인하였다. 정성적 비교에서도 객체의 위치와 영역이 예측 정확도가 높아지고, 객체 간 외형 혼합과 경계의 흐림이 완화되는 양상을 관찰하였다.

다만 객체의 미래 영역이 정답에 가깝게 예측되더라도, 해당 영역 내부에 인접한 다른 객체의 외형이 섞이는 현상은 여전히 남아 있다. 이는 정확한 미래 영역 예측만으로는 객체 간 외형 혼합을 완전히 방지하기 어려우며, 각 영역에서 서로 다른 객체의 표현이 혼합되지 않도록 하는 추가적인 제약이 필요함을 시사한다.

또한 객체 임베딩과 미래 영역 예측이 중간 denoising feature에 의존하므로, encoding된 denoising feature의 품질에 따라 영역 예측의 성능이 불안정하다.

향후에는 각 영역 내부에서 객체 표현의 혼합을 억제하는 학습 방법을 연구할 필요가 있고, 아울러 중간 feature의 품질에 대한 의존도를 낮춰 안정적인 객체 표현을 확보한다면 다중 객체 장면에서의 예측 안정성을 높일 수 있을 것으로 기대한다.

---

1. → masked attention decoding using gumble softmax
2. → video segmentation Large model인 SAM3의 prior를 사용하여 object representation 안정화



</article>
