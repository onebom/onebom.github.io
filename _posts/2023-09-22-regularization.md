---
title: L1&L2 Regularization + R1 regularization 맛보기
author: onebom
date: 2023-09-08 18:15:00 +0800
categories: [Graphics,3D Geometry]
tags: [3D]
toc: true
math: true
---

Regularization은 DL model의 gradient를 penalizing함으로써 학습을 제어하는 기법이다.
- weight이 지나치게 큰 값을 가지지 않도록 하면서 모델의 복잡도를 낮춘다.

## 1. Bias,Variance trade-off

Model Learning에서 Error(loss)를 계산할 때 다음과 같다;   
Error(X)=noise(X)+bias(X)+variance(X)
- noise는 데이터가 가지는 본질적인 한계치이기에, ireeducible error라고도 부른다.
- bias/variance는 모델에 따라 변하므로, reducible error라고 한다.

이미지
- bias가 작다는 것은 과녁에 중앙에 잘 편향되어있다는 뜻이다.
- variance가 작다는 것은 흩어져있고 잘 모여있다는 것을 의미한다.

여기서 Bias는 데이터 내 모든 정보를 고려하지 않음으로 인해, 지속적으로 잘못된 것들을 학습하는 경향을 말하며 underfitting이라고도 한다.   

반대로 Variance는 데이터 내 에러나 노이즈까지 잘 잡아내는 model에 데이터를 fitting시킴으로써, 실제 현상과 관계없는 random한 것까지 학습하는 경향을 의미하며, overfitting과 관계가 있다.     

둘 다 줄이는 것이 가장 좋은데, 둘의 관계는 trade-off관게다.   
**따라서 "bias가 좀 증가하더라도 variance를 감소시키자"를 구현한 것이 regularization이다.**

모델의 dimension이 커지면 복잡도가 높아지면서 variance가 커지므로, test data에 대해 잘못 예측할 가능성이 높아진다.    
따라서 복잡도만 높이고 성능에 영향을 주지 않는 feature를 없애므로써 local noise가 학습에 영향을 끼치지 않게하여 모델 성능을 높일 수 있다.
## 2. Lasso(L1) regularization
l1 norm을 이용하여 가중치를 계산한다.

$$C=C_0+{\lambda\over{n}}\sum_{w}\left\vert w \right\vert$$
- C_0 : 기존 cost function
- n : train data 수
- lambda : regularization 변수
- w: 가중치

regularization에서는 C_0이 작아지는 방향으로 단순하게 학습되는 것이 아닌, w값 역시 최소가 되는 방향으로 진행하게 된다.   
따라서 w에 대해 편미분을 수행하여 새로운 가중치를 얻는다;
$$w\rightarrow w' = w - {\eta\lambda \over{n}} \sgn w-\eta{\delta C_0\over{\delta w}}$$
- w의 부호에 따라 상수값을 빼준다.
- 이때 w'은 0이 될 수 있으며, 주요 가중치만 남길 수 있다.

## 3. Ridge(L2) regularization

Lasso와 달리 Ridge는 L2 norm을 사용하여 가중치를 준다.

$$C=C_0+{\lambda\over{2n}}\sum_{w}\left\vert w^2 \right\vert$$

w에 대해 편미분을 수행하면 다음과 같다;
$$w\rightarrow w' = w - {\eta\lambda \over{n}}w-\eta{\delta C_0\over{\delta w}} = (1-{\eta\lambda \over{n}})w- -\eta{\delta C_0\over{\delta w}}$$
- ()안의 값이 작아지는 방향으로 학습을 진행하게된다. 이를 weight decay라 한다
  - weight decay에 의해 특정 가중치가 비이상적으로 커지는 것을 방지한다.
- L1 regularization과 달리 w가 0이 되진 않고, 0에 가깝게 수렴하기 때문에 L1보다 둔감하다.
- => Outliner에 대해 신경써야하는 경우 L2가 효과적이다.

## (+) R1 Regularization 

GAN에서 regularization 기법은 일반적으로 discriminator가 [Nash Equilibrium](https://ko.wikipedia.org/wiki/%EB%82%B4%EC%8B%9C_%EA%B7%A0%ED%98%95)에서 벗어나지 않도록 penalty를 주기 위해 사용한다.   

R1 regularization이란 real data에 대해서만 gradient에 penalty를 줌으로써, generator는 실제 분포와 유사한 데이터를 생성할 수 있고 discriminator는 [data manifold](https://velog.io/@xuio/TIL-Data-Manifold-%ED%95%99%EC%8A%B5%EC%9D%B4%EB%9E%80#:~:text=%EA%B0%80%20%EB%93%B1%EC%9E%A5%ED%95%9C%EB%8B%A4.-,%F0%9F%9B%A0%20Data%20Manifold%EB%9E%80%3F,%EC%9E%98%20%ED%91%9C%ED%98%84%ED%95%A0%20%EC%88%98%20%EC%9E%88%EA%B2%8C%20%EB%90%9C%EB%8B%A4.)에서 zero gradient를 유지하게 된다.


GAN의 경우, R1 regularization 항이 추가된 목적 함수는 다음과 같다;
$$ R\_{1}\left(\psi\right) = \frac{\gamma}{2}E\_{p\_{D}\left(x\right)}\left[||\nabla{D\_{\psi}\left(x\right)}||^{2}\right] $$
- $\psi$ : 최적화 중인 판별기 모델
- $\gamma$ : regularization 강도를 제어하는 매개 변수    
- $$E\_{p\_{D}\left(x\right)}\left[||\nabla{D\_{\psi}\left(x\right)}||^{2}\right]$$ : real 데이터에서 discriminator gradient norm