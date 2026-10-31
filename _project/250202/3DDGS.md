---
layout: default
title: "Generalizable 3D Gaussian Reconstruction via Latent Diffusion"
permalink: /250202/3DDGS/
nav: false
---

<link rel="stylesheet" href="{{ '/assets/css/segjoint.css' | relative_url }}">

<article class="segjoint-page" markdown="1">

# Generalizable 3D Gaussian Reconstruction via Latent Diffusion

**Industry–Academia Collaborative Project**  
**Period:** August 2024 – February 2025  
**Organization:** Sungkyunkwan University × AIM FUTURE, Inc.  
**Advisor:** Prof. Il Yong Chun

## 프로젝트 개요

본 프로젝트는 단일 이미지로부터 3D Gaussian Splatting(3DGS) 기반의 3차원 객체 표현을 생성하고, 이를 활용해 입력 시점에서 관측되지 않은 새로운 시점의 이미지를 렌더링하는 방법을 연구했다. 
기존의 장면별 최적화 방식에서 벗어나 다양한 객체에 적용할 수 있는 일반화된 3D 복원을 목표로, **Variational Autoencoder(VAE)**와 **Latent Diffusion Model(LDM)**을 결합한 2단계 프레임워크를 설계하고 평가했다.

## 연구 배경과 문제 정의

3D Gaussian Splatting은 3차원 장면을 Gaussian primitive의 집합으로 표현하여 효율적으로 렌더링할 수 있다. 하지만 제한된 시점의 이미지만 주어졌을 때 보이지 않는 영역까지 일관되게 복원하는 것은 어렵다. 
초기에는 diffusion model이 3D Gaussian의 속성을 직접 생성하는 방식을 검토했다. 그러나 위치, 색상, 불투명도, 크기, 회전 등 각 속성의 값 범위가 서로 달라 하나의 정규화 방식으로 처리하기 어려웠다. 또한 baseline 모델인 **Splatter Image**에서 Gaussian 파라미터를 미리 추출해야 하는 부담이 있었다. 
이에 Gaussian 파라미터를 직접 생성하는 대신, **학습된 잠재 공간에서 diffusion을 수행한 후 디코더를 통해 Gaussian 파라미터를 생성하는 방식**으로 접근을 변경했다.

## 방법

### 1. Gaussian VAE 학습

입력 이미지를 VAE encoder로 잠재 표현으로 변환하고, decoder가 3D Gaussian primitive의 파라미터를 예측하도록 학습했다. 예측한 Gaussian을 목표 카메라 시점에서 렌더링한 뒤, 실제 목표 이미지와의 차이를 이용해 복원 품질을 학습했다. 
학습에는 렌더링 이미지에 대한 **reconstruction loss**와 잠재 공간을 정규화하기 위한 **KL divergence loss**를 사용했다.

### 2. Gaussian Latent Diffusion 학습

학습된 VAE의 잠재 표현에 노이즈를 추가하고, 입력 이미지를 조건으로 노이즈를 예측하는 latent diffusion model을 학습했다. 생성된 잠재 표현을 VAE decoder에 전달해 Gaussian 파라미터를 얻고, 이를 새로운 시점의 이미지로 렌더링했다. 
이 구조는 서로 다른 범위를 갖는 Gaussian 속성을 diffusion model이 직접 정규화해 생성해야 하는 문제를 줄이기 위한 것이다.

## 실험

- **데이터셋:** ShapeNet-SRN의 자동차 객체
- **입력 및 평가:** 단일 입력 이미지로 객체를 복원하고, 새로운 시점에서 렌더링한 결과를 정답 이미지와 비교
- **평가 지표:** PSNR, SSIM, LPIPS
- **샘플링:** DDIM, 100 steps
- **학습 장비:** NVIDIA GeForce RTX 4090

| 구성 요소 | Batch size | Optimizer | Learning rate | 학습 반복 |
|---|---:|---|---:|---:|
| Gaussian VAE | 8 | Adam | 5×10⁻⁵ | 800,000 |
| Gaussian LDM | 32 | Adam | 5×10⁻⁵ | 300,000 |

## 결과 및 분석

단일 입력 이미지로부터 자동차 객체의 3D Gaussian 표현을 생성하고, 여러 새로운 시점의 이미지를 렌더링했다. 최종 보고서에 따르면 Gaussian LDM은 학습이 진행됐으며, 정량·정성 평가에서 **Gaussian VAE와 비슷하거나 소폭 개선된 결과**를 보였다. 
다만 최종 성능은 1단계 Gaussian VAE의 복원 품질에 크게 영향을 받았다. 또한 LDM 학습 과정에서 loss가 크게 감소하지 않아, diffusion 모델과 학습 방식의 추가 개선이 필요한 것으로 분석했다.

## 한계와 향후 방향

- 최종 실험은 **합성 자동차 객체**에 한정되어 있어, 실제 도로 환경이나 주행 장면 전체에 대한 일반화 성능은 확인하지 못했다.
- 최종 복원 품질이 선행 VAE의 표현 능력과 복원 정확도에 의존했다.
- LDM의 개선 폭이 제한적이었으며, unseen 영역의 복원 향상을 별도로 입증하기 위한 추가 실험이 필요하다.

향후에는 Gaussian VAE의 복원 품질과 LDM 학습 안정성을 개선하고, 더 다양한 객체와 실제 주행 장면으로 평가 범위를 확장할 수 있다.

</article>
