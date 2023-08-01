---
title: 인공지능을 위한 선형대수:Least Square
author: onebom
date: 2021-07-31 18:15:00 +0800
categories: [DL, LinearAlgebra]
tags: [DL, LinearAlgebra]
toc: true
math: true
---

해당 포스트는 주재걸교수님의 "인공지능을 위한 선형대수" 강의를 듣고 작성하였습니다.
오늘의 글을 통해 꼭 알아야 하는 내용은 다음과 같습니다.
> [summary]
> 1.

---

## 1. Least Sqaure
몸무게,키,흡연여부를 통해 수명을 예측하는 예제에 대해서 생각해보자.
[이미지]
일반적으로 우리는 많은 수의 데이터셋을 가지게 될것이다. 이때 linear system에서 나온 내용에서 처럼 m>>n 이되면서 우리는 해당 데이터에 딱 맞는 solution을 가질 수 없게 된다. 이 경우를 over-determined system이라 부른다.    

따라서, Least sqaure의 근간은 이러한 over-determined system에서 모든 데이터에 대해 가장 근접하게 맞출수 있는 solution을 찾고자 하는 목표에서 출발한다.    

가장 근접한지에 대한 척도는 두 벡터간의 거리로 판단한다.
vector norm
unit vector
distance -> inner product and angle

문제 예시
least squares problem 정의
기하학적으로 바라보기

## 2. Normal Equation
least squares prolem을 푸는 방법중 하나

## 3. Orthogonal Projection
least squares prolem을 푸는 방법중 두번째

orthogonal, orthonormal set이란?
orthogonal porjection

문제풀이에 적용

## 4. Gram-Schmidt Orthogonalization
이점이

## QR Factorization