---
title: 인공지능을 위한 선형대수:Linear System & Transformation(2)
author: onebom
date: 2021-07-28 23:15:00 +0800
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

## 4. Span & Subspace
지난 시간에 span에 대해서 알아보았다. span의 vector들의 집합을 나타내는 공간 개념이다.    
반면, 벡터들의 집합이 스칼라 곱과 덧셈(linear combination)에 *닫혀있다*는 조건을 만족하면 이 공간을 **부분공간(subspace)**이라 한다.
- *곱셈과 덧셈에 닫혀있다*라는 개념을 쉽게 말해보면, 하나의 벡터공간 상에 있는 두개의 벡터를 더하거나 해당 벡터에 스칼라 곱샘이 이뤄져도 같은 벡터공간에 포함된다는 것을 말한다.
- subspace는 span과 비슷한 개념이라고 생각하면 된다. 왜냐하면 벡터들의 집합이 주어졌을 때 Span{v1,...,vp}는 항상 부분공간이 된다.
- 이를 증명하자면,    
  $$u_1= a_1v_1+\cdots+a_pv_p , u_2= b_1v_1+\cdots+b_pv_p$$ 인경우,   
  $$cu_1+du_2= (ca_1+db_1)v_1+\cdots+(ca_p+db_p)v_2$$로 표현할 수 있기 때문에 => 곱셈과 덧셈에 닫혀있는 것을 볼 수 있다.

basis는 이러한 vector 공간을 대표하는 vector이다.
> **basis란**   
> subspace를 span할 수 있는 vector 집합이다.    
> 나아가, 각 vector끼리 Linearly independent해야한다.
>
> 예를 들어, H=Span{v1,v2,v3}이며 v3=2v1+3v2인 경우, $$v3 \in Span\{v1,v2\}$$가 된다.    
> 따라서 H의 basis는 {v1,v2}를 말한다.    
> 우리는 basis를 통해 해당 벡터공간의 주요 특징을 알아낼 수 있다.   

그렇담, 같은 subspace H의 linearly independent한 vector set은 하나만 존재하는가? 즉,
### point1) subspace H의 basis set은 하나만 존재하는가?
답은 아니다!
![figure1](/assets/img/posts/LinearAlgebra2/figure1.png)   
동일한 xy평면의 한 점을 표현한다고 했을떄, v1과 v2를 사용하여 점을 나타낼수도 있지만 다른 coefficient를 통해 a1과 a2를 사용해서 나타낼 수도 있다.   
이를 change of basis라 한다.      
이후에 배울 eigen decomposition에서도 행렬의 basis를 전환함으로써 행렬을 다른 형식으로 분해할 수 있다.    
- 다만 basis set의 vector 조합은 다양할 수 있으나, basis set의 vector 수는 같은 행렬의 basis라면 동일해야한다.
- 이러한 basis set의 vector수를 **dimension**이라한다-> $$dim H$$
  - 예로, 2차원 xy평면의 dimension은 어떤 basis를 뽑더라도 2개의 vector로 이뤄져있어야한다. 

$$A= \begin{bmatrix} 1 & 1 & 2 \\ 1 & 0 & 1 \\ 0 & 1 & 1\end{bmatrix} $$이 주어졌을 때, 
$$Col A = Span\begin{Bmatrix} \begin{bmatrix} 1\\1\\0 \end{bmatrix} , \begin{bmatrix} 1\\0\\1 \end{bmatrix} , \begin{bmatrix} 2\\1\\1\end{bmatrix} \end{Bmatrix} =  Span\begin{Bmatrix} \begin{bmatrix} 1\\1\\0 \end{bmatrix} , \begin{bmatrix} 1\\0\\1 \end{bmatrix} \end{Bmatrix}$$   

따라서 해당 Col A의 dimension은 2임을 알 수 있다.   
이때 행렬 A의 column space의 dimension을 우리는 **Rank**라 부른다.


## 5. Linear transformation
Linear transformation은 어려울게 없다. transformation은 다른 용어로 function, mapping이라 할수 있다.    
linear transformation은 그 중에서도 `T(cu+dv)=cT(u)+dT(v)`조건을 만족시키는 function이다.    
- 예를 들어, T(x)=3x에 대해서 x_1=1 x_2=2가 주어졌을 때,   
  T(4x_1+5x_2)=T(14)=42, 4T(x_1)+5T(x_2)=4x3+5x6=42    
  처럼 동일한 값이 나오는 function을 linear transformation이라 한다.

| Example) Suppose T is a linear transformation from $\mathbb{R}^2$ to $\mathbb{R}^3$ such that   
| $$T\begin{pmatrix} \begin{bmatrix} 1\\0 \end{bmatrix}\end{pmatrix}=\begin{bmatrix} 2\\-1\\1 \end{bmatrix}, T\begin{pmatrix} \begin{bmatrix} 0\\1 \end{bmatrix}\end{pmatrix}=\begin{bmatrix} 0\\1\\2 \end{bmatrix}$$.   
| with no additional information find a formula for image of an arbitary x in $\mathbb{R}^2$ 
   
를 풀어보자!   
$$x=\begin{bmatrix} x_1\\x_2 \end{bmatrix}=x_1\begin{bmatrix} 1\\0 \end{bmatrix}+x_2\begin{bmatrix} 0\\1 \end{bmatrix}$$와 같이 x를 표현할 수 있다.

$$T(x)=T(x_1\begin{bmatrix} 1\\0 \end{bmatrix}+x_2\begin{bmatrix} 0\\1 \end{bmatrix})=x_1T(\begin{bmatrix} 1\\0 \end{bmatrix})+x_2T(\begin{bmatrix} 0\\1 \end{bmatrix})=x_1\begin{bmatrix} 2\\-1\\1 \end{bmatrix}+x_2\begin{bmatrix} 0\\1\\2 \end{bmatrix} = \begin{bmatrix} 2&0\\-1&1\\1&2 \end{bmatrix}\begin{bmatrix} x_1\\x_2 \end{bmatrix}$$

우리는 2개의 transformation 출력값을 통해 2차원 input에 대한 transformation 식을 정형화할 수 있었다.    

여기까지 읽었다면, 드디어 지금껏 배운 내용을 기반으로 인공지능의 연산과정이 어떻게 이뤄지는 살짝 맛볼 수 있다. https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/   
 
우리가 배울 딥러닝 모델의 연산은 여러개의 다양한 연산 레이어를 거치면서 일어난다. 이때 거의 모든 레이어가 동일하게 선형변환(affine transformation) 후 point wise 활성화 함수를 거치는 순이다.    
tanh layer를 에를 들어 연속적인 레이어 연산의 결과를 시각해보면 아래와 같다.   
![figure2](/assets/img/posts/LinearAlgebra2/figure2.gif)  
tanh layer $$tanh(Wx+b)$$는 다음으로 구성되어있다;
1. linear transformation by "weight" matrix W
2. translation by the vector b
3. Point-wise application of tanh 

이 figure가 의미하는 바는,   
처음에는 basis vector와 축 벡터가 일련의 A행렬곱이 일어나는 T를 거치면서 vector 변환이 일어나, 모눈종이가 꾸겨지는 과정을 보여준다.   
옆으로 흐르는 이유는 bias로 인한 것이고, 이후 가장자리가 눌려지는 부분은 non-linear한 activation function의 연산으로 중앙에서 멀어지는 부분의 차이를 최소화하기 위해 압축하는 과정을 도식화한 부분이라고 한다.    

더욱 직접적으로 하나의 layer에서 일어나는 linear transformation 연산을 살펴보자.    
![figure3](/assets/img/posts/LinearAlgebra2/figure3.png) 

## 6. Onto & One-to-One
간단하게 인공지능 연산과정에 대해 이야기하자면, 알고있는 정보로 이뤄진 input data를 일련의 linear transformation을 거치게해서 우리가 얻고자하는 정보 domain의 output으로 전환하는 과정이라 말할 수 있을 것 같다.   
input을 output으로 전환하는 과정에서 우리는 2가지 조건을 고려해야한다; ONTO & ONE-TO-ONE
>**[ONTO란]**   
>
>ONE-TO-ONE이란