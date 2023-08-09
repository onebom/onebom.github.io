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
일반적으로 인공지능을 학습시키기 위해서 우리는 많은 수의 데이터셋을 가지게 될 것이다. linear system 글을 상기시켜보자. linear system의 equation 수 m이 variable 수 n보다 커지게 되면(m>>n), 우리는 해당 데이터들을 모두 맞출 수있는 solution을 가질 수 없다. 이 경우를 over-determined system이라 부른다.    

따라서, Least sqaure의 근간은 이러한 over-determined system에서 모든 데이터에 대해 가장 근접하게 맞출수 있는 solution을 찾고자 하는 목표에서 출발한다.    
over-determined system을 vector equation으로 표현해보면 다음과 같다;       
$$\begin{bmatrix}60\\65\\55\\ \vdots\end{bmatrix}x_1+\begin{bmatrix}5.5\\5.0\\6.0\\ \vdots\end{bmatrix}x_2+\begin{bmatrix}1\\0\\1\\ \vdots\end{bmatrix}x_3= \begin{bmatrix}66\\74\\78\\ \vdots\end{bmatrix}$$   
$\mathbf{a_1}x_1+\mathbf{a_2}x_2+\mathbf{a_3}x_3=\mathbf{b}$
- space $$\mathbb{R}^n$$에 대해서 $$a_1,a_2,a_3,b \in \mathbb{R}^n$$이라면, $$Span\{a_1,a_2,a_3\}$$는 정말 얇디얇은 hyperplane일 것이다. 따라서 $$b \in Span\{a_1,a_2,a_3\}$$일 확률은 거의 없다.

따라서 b와 가장 근접한 Span{a_1,a_2,a_3}에서의 $\hat{b}$를 찾고자한다.   
바로 least square 문제를 다루지 않고 가장 기본적인 norm,distance,angle에 대해서 개념을 잡고갈것이다. 이 개념들에 대해 다 안다면 바로 2.normal equation으로 넘어가길 바란다.    

가장 근접한지에 대한 척도는 두 벡터간의 거리로 판단한다. 
### point1) $\mathbb{R}^n$에서의 vector간 거리는 어떻게 구하는 가?
같은 $\mathbb{R}^n$에 포함되는 u,v에 대해서 둘 사이 거리는 다음과 같이 구할 수 있다;
$$dist(u,v)= \lVert\mathbf{u}-\mathbf{v}\rVert$$

제대로 계산하기 앞서, 우리는 vector norm에 대해서 집고 넘어가야한다.    
> **Vector norm**이란,   
> 방향을 고려하지 않고 순수 vector의 길이에 대한 값이다.   
> non-nogative scalar $\lVert v \rVert$ 를 말하며 두벡터간의 내적값 $v\cdot v$에
> square root를 씌운 값이다.
> $$ \lVert v \rVert= \sqrt{ \mathbf{v}\cdot\mathbf{v}}=({v_1}^2+{v_2}^2+\cdots+{v_n}^2)^0.5, {\lVert v \rVert}^2=\mathbf{v}\cdot\mathbf{v}$$   
> 
> 여기서 vector norm이 1인 vector를 **Unit vector**라 한다.
> 따라서 이후 계속 언급될 **Normalizing**은 nonzero vector가 주어졌을때, 해당 vector를 unit vector로 만드는 과정이다; $\mathbf{u}={1\over{\lVert v \rVert}}\mathbf{v}$
> - 즉, u는 v와 같은 direction을 가진 길이가 1인 벡터다.

예제를 통해 vector dist를 구해보자면,

$u=\begin{bmatrix} 6 \\ 1 \end{bmatrix}, v=\begin{bmatrix} 3 \\ 2 \end{bmatrix}$

$u-v=\begin{bmatrix} 6 \\ 1 \end{bmatrix}-\begin{bmatrix} 3 \\ 2 \end{bmatrix}=\begin{bmatrix} 3 \\ -1 \end{bmatrix}$

$\lVert u-v \rVert=\sqrt{3^2+(-1)^2}=\sqrt{10}$

이를 그래프상에서 보자면,
![figure2](/assets/img/posts/LinearAlgebra3/figure2.png)    

다시 b와 Span{a_1,a_2,a_3}에 대해서 생각해보자.    
vector b가 space에 가장 근접하려면 어떻게 해야할까? 아까 배운 내용에서 추론하면, space에서의 vector와 거리가 가장 짧으면 된다. 여기서 거리가 가장 짧은 vector는 vector b가 space에 수직으로 선분을 내렸을 때 수선과의 교점을 지나는 vector일 것이다. 머리속으로 그림을 그려봐라.    
그렇다면 $\hat{b}$를 구하기 위해, 벡터간의 각도를 구할 수 있어야한다.
### point2) 벡터간의 각도는 어떻게 구하는가?
벡터 u,v에 대해서 내적을 하면, norm과 angle을 통해서 표현할 수 있다;
$u\cdot v = \lVert u\rVert\lVert v\rVert\cos{\theta}$   
![figure3](/assets/img/posts/LinearAlgebra3/figure3.png)    

이때 두벡터의 각이 90도일 경우, **orthogonal**하다고 하며 가장 큰 특징은 두벡터의 내적값이 0이라는 것이다;    
$u\cdot v=\lVert u \rVert \lVert v \rVert \cos{\theta}=0$   
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
$$A \in \mathbb{R}^{m\times n}, \mathbf{b} \in \mathbb{R}^n$$이며 m>>n인, over-determinde system $$Ax\simeq b$$가 주어졌을 때, least squares solution $\hat{x}$는 아래와 같이 정의된다;   

$$ \hat{x} =\arg \underset{x}{min} \lVert{\mathbf{b}-Ax} \rVert$$

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
$\hat{x}$는 **normal equation**을 통해 구할 수 있다;    

$$A^TA\hat{x}=A^Tb$$

$$\hat{x}=(A^TA)^{-1}A^Tb$$
  
- 이때 $C=A^TA \in \mathbb{R}^{n\times n}$ 여야하고, $d=A^Tb \in \mathbb{R}^n$이여야 한다.
  - n은 여기서 feature 개수와 같다.
- 나아가 $C=A^TA$가 invertible해야지만, 항을 넘길 수 있다.

갑자기 normal equation 식이 튀어나와 혼란스러울 것이다. normal equation이 least squares 문제에서 어떻게 도출되었는지 다시 차례차례 전개해보자.

$$ \hat{x} =\arg \underset{x}{\min} \lVert{\mathbf{b}-Ax} \rVert = \arg \underset{x}{\min} {\lVert \mathbf{b}-Ax \rVert}^2 $$

> b-Ax vector의 길이가 최소가 되는 문제는 해당 vector의 제곱을 취해도 최소가 되어야 하므로 두 문제는 동일하다.

$$ = \arg \underset{x}{\min} (\mathbf{b}-Ax)^T(\mathbf{b}-Ax) = b^Tb-x^TA^Tb-b^TAx+x^TA^TAx $$

> [NOTE]    
> $${\lVert x \rVert}^2=x^Tx$$ 식을 활용할 수 있다.
  - 위 식의 증명은 간단한 예제로 가능하다. x=[3 4]일 때, x^2=9+16이다. 이는 x^T와 x를 내적한 값과 동일하다

해당 식은 여러 벡터로 이뤄져있지만, 결과적으로 목적함수 값은 벡터의 길이인 스칼라 값이다. 따라서 스칼라값을 미분하면 0을 얻을 수 있기 때문에 해당 식을 미분하여 다음과 같은 결론을 얻을 수 있다.

$$ 0-A^Tb-A^Tb+2A^TAx=0 $$

$$ \rightarrow A^TAx=A^Tb $$

미분이 어떻게 전개됐는지 상세 설명을 해보자면,   
> [NOTE1]   
> $$f(x)=a^Tx=x^Ta \rightarrow {df\over{dx}}=a$$   
> [NOTE2]   
> $$f\dot g = f' \dot g + f \dot g'$$   
> ex)$$x^3=x^2 \dot x = 2x\dot x + x^2 \dot 1$$

1. $$b^Tb$$는 하나의 스칼라 값이기에 미분하면 0이다.
2. $$x^TA^Tb$$에서 $A^Tb$의 값을 계산하면 하나의 column vector가 된다. 따라서 $A^Tb$는 NOTE1에서의 a 역할을 한다. 따라서 해당 항을 미분하면 $A^Tb$가 나온다. 3번째 항도 마찬가지다.
3. $$x^TA^TAx$$에 NOTE2를 적용시켜, $$x^T \dot A^TAx$$로 계산할 수 있다. 그럼 마찬가지로 NOTE1에 따라 $$A^TAx+ (x^TA^TA)^T$$ 미분값을 얻는다. 그래서 $$2A^TAx$$가 나오게 된것이다.

결과적으로 $A^TA$를 우변으로 넘겨 최종 식을 정리할 수 있다.   

$$x=(A^TA)^{-1}A^Tb$$

이 풀이로 수명을 예측하는 문제를 풀어보자
![figure6](/assets/img/posts/LinearAlgebra3/figure6.png)

### point4) 만약 $C=A^TA$가 non-invertible하면 어떻게 되는가?
위의 normal equation 내용은 $A^TA$가 invertible한 경우에 대해서만 다뤘다.
- A의 vector가 서로 linear independent한 경우, $A^TA$는 invertible하다.
그러나 non-invertible한 경우는 어떻게 해야할까?     

우선 b가 Ax 평면에 내리는 수선의 발은 언제나 1개로 unique하며 vector가 평행하지 않은 이상 수선의 발을 내리지 못하는 경우는 없다.   
A가 선형독립일 때 수선의 발인 $\hat{b}$를 나타낼 수 있는 선형 계수는 하나의 조합 밖에 없다. 그러나, A가 선형독립이 아닌 경우 무수히 많은 조합으로 hat b를 나타낼 수 있게 되며, 이는 **normal equation에서의 식에서 정립되지 않는다.**    

그럼 실제 데이터셋에서 이런 경우를 만났을 때 어떻게 해야하는가 혼란스러울 것이다. 
그러나, 실제 학습 데이터셋에서 A^TA가 non-invertible한 경우는 거의 없다.    
- feature vector의 측면에서 dependent를 따져보면, 많은 수의 데이터셋에서 각 feature끼리 정확히 선형 의존이 되는 경우는 정말정말저엉말 어렵다. 
- 따라서 딥러닝 학습셋에서 A^TA는 대부분 invertible한 케이스다.


## 3. Orthogonal Projection
normal equation을 orthogonal 특징에 초점을 둔 과점에서도 해석할 수 있다.
우리는 앞서 $\hat{b}$를 구할 때, b가 Ax 평면에 orthogonal하다는 성질을 활용해 이해했었다. 따라서 noraml equation을 $\hat{b}$에 대해 써보면;

$$\hat{b}=A\hat{x}$$

$$\hat{x}=(A^TA)^{-1}A^Tb$$

$$\rightarrow \hat{b}=f(b)=A\hat{x}=A(A^TA)^{-1}A^Tb$$

이 식을 통해 $\hat{b}$를 b vector에 행렬을 곱해 얻어내는 linear transformation으로 표현할 수 있다는 것을 알 수 있고 즉, 구하고자 하는 projection을 다른 projection의 선형 결합으로 쪼개어 표현할 수 있다고 해석할 수 있다. 
 
projection을 구하기 이전에 orthogonal에 대해서 좀 더 알아보자.   
> **orthogonal set - Defn**    
> {u_1, ..., u_p} vector set이 있을 때, 해당 set의 모든 vector가   
> $u_i \cdot u_j = 0 whenever j \ne j$를 만족하면 **orthogonal set**이라 부른다. 
> **orthonormal set - Defn**   
> {u_1, ..., u_p} vector set이 있을 때, orthogonal set이면서 동시에 unit vector로 이뤄진 경우 **orthonormal set**이라 부른다.
> - 일련의 vector set이 있을 때, orthogonal(or orthonormal) basis set으로 변환하는 시스템을 Gram-Schmidt process라 한다. 

orthogonal basis vector ($\into W$) set이 주어졌을 때, vector y를 W 평면에 orthogonal porjection을 취해보자 (orthogonal projection of 𝐲 ∈ R𝑛 **onto** 𝑊.)
1.  Orthogonal Projection $\hat{y}$ of y onto Line
![figure7](/assets/img/posts/LinearAlgebra3/figure7.png)   
위 figure에서 $$\hat{y}$$를 "방향 x 길이"로 표현할 수 있다.
1. $$\hat{y}$$의 길이   
   벡터 y와 u 사이 각이 $\theta$라고 주어졌을 때,   
    $$\lVert \hat{y} \rVert =  \lVert y \rVert \cos{\theta} ={y\cdot u \over {\lVert u \rVert}}$$
2. $$\hat{y}$$의 방향
   u 벡터의 방향과 동일하다(=u방향의 unit-vector); $${1 \over {\lVert u \rVert}}\times u $$

$$\hat{y}={1 \over {\lVert u \rVert}} \times u \times {y\cdot u \over {\lVert u \rVert}} = {y\cdot u \over {u \cdot u}}u$$

만약 u가 애초에 unit vector라면, $u \cdot u = 1$이기에 다음과 같은 결론을 얻는다.

$$\hat{y}= (y \cdot u)u$$

3.  Orthogonal Projection $\hat{y}$ of y onto Plane
사영할 subspace의 dimension이 늘어났다고 크게 바뀌지 않는다.
앞서 말했듯, 구하고자 하는 $\hat{y}$는 x_1과 x_2 라인에 y를 사영시킨 $\hat{y_1}, \hat{y_2}$의 선형결합으로 표현가능하기 때문이다.   
![figure8](/assets/img/posts/LinearAlgebra3/figure8.png)     

따라서 아래와 같이 선형결합을 통해 hat y를 구할 수 있다;

$$\hat{y}={y \cdot u_1 \over {u_1 \cdot u_1}}u_1+{y \cdot u_2 \over {u_2 \cdot u_2}}u_2$$

마찬가지로 u_1과 u_2가 unit vector이면 다음과 같다;

$$\hat{y}= (y \cdot u_1)u_1+(y \cdot u_2)u_2$$


### point5) Orthogonal projection을 linear transformation으로 표현해보자

Consider a transformation of orthogonal projection 𝐛 of 𝐛, given orthonormal basis {𝐮1, 𝐮2} of a subspace 𝑊
- 이때 우리는 NOTE1를 다시 한번 써서 전개할 것이다.

$$\hat{b}=f(b)=(b\cdot u_1)u_1+(b\cdot u_2)u_2 = ({u_1}^Tb)u_1+({u_2}^Tb)u_2=(u_1{u_1}^T)b+(u_2{u_2}^T)b=(u_1{u_1}^T+u_2{u_2}^T)b$$

$$ = \begin{bmatrix} u_1 & u_2  \end{bmatrix}\begin{bmatrix} {u_1}^T \\ {u_2}^T  \end{bmatrix}b= UU^Tb$$

다시 normal equation으로 돌아와서 $$A=U=\begin{bmatrix} u_1 & u_2 \end{bmatrix}$$라면,   

$$C=A^TA=\begin{bmatrix} {u_1}^T \\ {u_2}^T \end{bmatrix} \begin{bmatrix} u_1 & u_2 \end{bmatrix}=I$$

이기 때문에, 

$$\hat{b}=A(A^TA)^{-1}A^Tb= A(I^-1)A^Tb=AA^Tb=UU^Tb$$

로 쓸 수 있다!

## 4. Gram-Schmidt Orthogonalization
이점이

## QR Factorization