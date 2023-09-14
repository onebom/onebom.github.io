---
title: 인공지능을 위한 선형대수:Eigendecomposition(1)
author: onebom
date: 2023-08-18 18:15:00 +0800
categories: [DL, LinearAlgebra]
tags: [DL, LinearAlgebra]
toc: true
math: true
---

해당 포스트는 주재걸교수님의 "인공지능을 위한 선형대수" 강의를 듣고 공돌이의 수학정리 블로그를 참조하여 작성하였습니다.
오늘의 글을 통해 꼭 알아야 하는 내용은 다음과 같습니다.
> [summary]
> 1. 

---

이 포스터에 다룰 고유벡터와 고윳값은 다음 2가지를 물어본다;    
“벡터 x에 어떠한 선형변환 A를 했을 때, 그 크기만 변하고 원래 벡터와 평행한 벡터 x는 무엇인가?”    
“그렇다면, 그 크기는 얼마만큼 변했나요?”

이 질문의 답을 가지고 우리는 벡터 연산을 효율적으로 할 수 있게 된다.  

## 1. Eigenvectors and Eigenvalues
벡터에 행렬 연산을 취하면 보통 원래와 다른 벡터를 출력한다. 그런데 위의 내용처럼 특정한 벡터와 행렬은 linear transformation을 취해줬을때, 크기만 바뀌고 방향을 바뀌지 않는 경우가 있다. 

> [Defn]
> 정방행렬(square matrix) A가 주어졌을때,      
> 
> $A\vec{x}=\lambda \vec{x}$
> 
> 에 대하여 vector x가 0이 아닌(nonzero) solution이라면,   
> 상수 $\lambda$는 eigenvalue, $\vec{x}$는 $\lambda$에 대응하는 eigenvector 라고 한다.   
>
> Ex)   
> $$\begin{bmatrix} 2 && 6 \\ 5 && 3 \end{bmatrix}\begin{bmatrix} 1 \\ 1 \end{bmatrix}=8\begin{bmatrix} 1 \\ 1 \end{bmatrix}$$

위의 간단한 예시를 살펴보면,
$$\begin{bmatrix} 2 && 6 \\ 5 && 3 \end{bmatrix}\begin{bmatrix} 1 \\ 1 \end{bmatrix}$$의 경우, 총 6번의 연산이 일어난다.   
그러나 $$8\begin{bmatrix} 1 \\ 1 \end{bmatrix}$$의 경우, 연산횟수는 2회이다.

따라서 우리는 고윳값 분해를 구해냄으로써, 더 적은 연산으로 결과를 구할 수 있게된다. 그렇기 때문에 대규모 데이터와 컴퓨터의 비트 연산 환경에서 실시간으로 빠른 연산을 위해 고윳값 분해는 필수적이다.

$A\vec{x}=\lambda \vec{x}$


## 2. 영공간(Null Space) & 직교여공간(Orthogonal Complement)

## 3. Characteristic Equation

## 4. Diagonalizable matrix

## 5. Eigendecomposition

## 6. Diagonalization

## 7. 대수 중복도와 기하 중복도(Algebraic multiplicity and geometric multiplicity)