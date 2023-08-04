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

## 1. Least Sqaure intro
몸무게,키,흡연여부를 통해 수명을 예측하는 예제에 대해서 생각해보자.   
![figure1](/assets/img/posts/LinearAlgebra3/figure1.png)   
일반적으로 우리는 많은 수의 데이터셋을 가지게 될 것이다. linear system 글을 상기시켜보자. linear system의 equation 수 m이 variable 수 n보다 커지게 되면(m>>n), 우리는 해당 데이터들을 모두 맞출 수있는 solution을 가질 수 없다. 이 경우를 over-determined system이라 부른다.    

따라서, Least sqaure의 근간은 이러한 over-determined system에서 모든 데이터에 대해 가장 근접하게 맞출수 있는 solution을 찾고자 하는 목표에서 출발한다.    
over-determined system을 vector equation으로 표현해보면 다음과 같다;       
$$\begin{bmatrix}60\\65\\55\\ \vdots\end{bmatrix}x_1+\begin{bmatrix}5.5\\5.0\\6.0\\ \vdots\end{bmatrix}x_2+\begin{bmatrix}1\\0\\1\\ \vdots\end{bmatrix}x_3= \begin{bmatrix}66\\74\\78\\ \vdots\end{bmatrix}$$   
$a_1x_1+a_2x_2+a_3x_3=b$
- space $\mathbb{R}^n$에 대해서 $a_1,a_2,a_3,b \in \mathbb{R}^n$이라면, $Span{a_1,a_2,a_3}$는 정말 얇디얇은 hyperplane일 것이다. 따라서 $b \in Span{a_1,a_2,a_3}$일 확률은 거의 없다.

따라서 b와 가장 근접한 Span{a_1,a_2,a_3}에서의 $\hat{b}$를 찾고자한다. 바로 least square 문제를 다루지 않고 가장 기본적인 norm,distance,angle에 대해서 개념을 잡고갈것이다. 위의 개념에 대해 다 안다면 바로 2.normal equation으로 넘어가길 바란다.    

가장 근접한지에 대한 척도는 두 벡터간의 거리로 판단한다. 
### point1) $\mathbb{R}^n$에서의 vector간 거리는 어떻게 구하는 가?
같은 $\mathbb{R}^n$에 포함되는 u,v에 대해서 둘 사이 거리는 다음과 같이 구할 수 있다;
$dist(u,v)= \lVert{\mathbf{u}-\mathbf{v}}$

제대로 계산하기 앞서, 우리는 vector norm에 대해서 집고 넘어가야한다.    
> **Vector norm**이란,   
> 방향을 고려하지 않고 순수 vector의 길이에 대한 값이다.   
> non-nogative scalar $\lVert v$ 를 말하며 두벡터간의 내적값 $v\cdot v$에
> square root를 씌운 값이다.
> $$ \lVert v = \sqrt{ \mathbf{v}\cdot\mathbf{v}}=({v_1}^2+{v_2}^2+\cdots+{v_n}^2)^0.5, {\lVert v}^2=\mathbf{v}\cdot\mathbf{v}$$   
> 
> 여기서 vector norm이 1인 vector를 **Unit vector**라 한다.
> 따라서 이후 계속 언급될 **Normalizing**은 nonzero vector가 주어졌을때, 해당 vector를 unit vector로 만드는 과정이다; $\mathbf{u}={1\over{\lVert v}}\mathbf{v}$
> - 즉, u는 v와 같은 direction을 가진 길이가 1인 벡터다.

예제를 통해 vector dist를 구해보자면,
| $u=\begin{bmatrix} 6\\1 \end{bmatrix}, v=\begin{bmatrix} 3\\2 \end{bmatrix}$
| $u-v=\begin{bmatrix} 6\\1 \end{bmatrix}-\begin{bmatrix} 3\\2 \end{bmatrix}=\begin{bmatrix} 3\\-1 \end{bmatrix}$
| $\lVert{u-v}=\sqrt{3^2+(-1)^2=\sqrt{10}$
이를 그래프상에서 보자면,
![figure2](/assets/img/posts/LinearAlgebra3/figure2.png)    

다시 b와 Span{a_1,a_2,a_3}에 대해서 생각해보자.    
vector b가 space에 가장 근접하려면 어떻게 해야할까? 아까 배운 내용에서 추론하면, space에서의 vector와 거리가 가장 짧으면 된다. 여기서 거리가 가장 짧은 vector는 vector b가 space에 수직으로 선분을 내렸을 때 수선과의 교점을 지나는 vector일 것이다. 머리속으로 그림을 그려봐라.    
그렇다면 $\hat{b}$를 구하기 위해, 벡터간의 각도를 구할 수 있어야한다.
### point2) 벡터간의 각도는 어떻게 구하는가?
벡터 u,v에 대해서 내적을 하면, norm과 angle을 통해서 표현할 수 있다;
$u\cdot v = \lVert{u}\lVert{v}\cos{\theta}$   
![figure3](/assets/img/posts/LinearAlgebra3/figure3.png)    

이때 두벡터의 각이 90도일 경우, **orthogonal**하다고 하며 가장 큰 특징은 두벡터의 내적값이 0이라는 것이다;    
$u\cdot v=\lVert{u}\lVert{v}\cos{\theta}=0$   
$\cos{90^\circ}=0$이기 때문이다.    

이 성질을 통해 우리는 $\hat{b}$를 찾아낼 것이다.

다시 over-determined system으로 돌아와서,   
두 경우를 살펴보자.    
![figure4](/assets/img/posts/LinearAlgebra3/figure4.png)  
- 첫번째 경우의 error 수치를 계산해보면, $\sqrt{(-5.3)^2+1.8^2+(-1.9)^2+7.5^2}=9.55$
- 두번째 경우는, $\sqrt{0^2+0^2+0^2+(-12)^2}=12$
- 따라서 least squre 문제에서 가장 최적의 근접해는 얼마나 답을 맞췄는지가 아닌, 얼마나 error의 합이 적은지를 따른다.

## 2. Normal Equation
Least squares 문제를 정의해보겠다.
$A \in \mathbb{R}^{m\times n}, \mathbf{b} \in \mathbb{R}^n$이며 m>>n인, over-determinde system $Ax\simeq b$가 주어졌을 때, least squares solution $\hat{x}$는 아래와 같이 정의된다;   

$$ \hat{x} =\arg \underset{x}{\overset{\min}} \lVert{\mathbf{b}-Ax}$$

- 어떠한 vector x가 선택되든지 상관없이, vector Ax는 Col A에 포함된다.
- 결과적으로, Least squares 문제는 Ax가 b와 가장 근접해지는 x를 찾는 것이 목표다.

마찬가지로 기하학적 관점에서 생각해보자. 
![figure5](/assets/img/posts/LinearAlgebra3/figure5.png)   
- $\hat{x}$에 대해서 $\hat{b}=A\hat{x}$는 Col A의 모든 지점 중에 가장 b와 근접한 지점일 것이다.
- $b-A\hat{x}$ vector를 그려보자. 그럼 Col A와 수직한 벡터를 표현할 수 있다.

즉, $b-A\hat{x}$는 모든 Col A의 vector들과 수직해야한다;   

$$b-A\hat{x} \bot (x_1a_1+x_2a_2+\cdots+x_pa_n)$$

### point3) b와 가장 근접한 Col A의 $\hat{b}$를 어떻게 구하는가?
이 문제는 $\hat{x}$를 구하는 문제와 동일하다.
따라서,   

$$A^TA\hat{x}=A^Tb$$

$$\hat{x}=(A^TA)^{-1}A^Tb$$

를 전개함으로써 구할 수 있다.    
- 이때 $C=A^TA \in \mathbb{R}^{n\times n}$ 여야하고, $d=A^Tb \in \mathbb{R}^n$이여야 한다.
- 나아가 $C=A^TA$가 invertible해야지만, 항을 넘길 수 있다.

least squares 문제식에서 normal equation을 전개해보자.

$$ \hat{x} =\arg \underset{x}{\overset{\min}} \lVert{\mathbf{b}-Ax} = \arg \underset{x}{\overset{\min}} \lVert{\mathbf{b}-Ax}^2 $$

$$


## 3. Orthogonal Projection
least squares prolem을 푸는 방법중 두번째

orthogonal, orthonormal set이란?
orthogonal porjection

문제풀이에 적용

## 4. Gram-Schmidt Orthogonalization
이점이

## QR Factorization