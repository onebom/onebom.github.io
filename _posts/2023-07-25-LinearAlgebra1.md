---
title: 인공지능을 위한 선형대수:Linear System & Transformation(1)
author: onebom
date: 2021-07-25 23:15:00 +0800
categories: [DL, LinearAlgebra]
tags: [DL, LinearAlgebra]
toc: true
math: true
img_path: '_site/posts/linearAlgebra1'
---

해당 포스트는 주재걸교수님의 "인공지능을 위한 선형대수" 강의를 듣고 작성하였습니다.
오늘의 글을 통해 꼭 알아야 하는 내용은 다음과 같습니다.
> [summary]
> 1. linear equation을 matrix간

---
## 1. Linear System
Linear System이란 하나 이상의 (같은 vatiables `x_1,...,x_n`을 공유하는) linear equation의 모음이다.    
linear equation은 우리가 흔히 아는 $$ 𝑎_1𝑥_1+𝑎_2𝑥_2+⋯+𝑎_{𝑛}𝑥_{𝑛} = 𝑏 $$ 꼴의 등식을 말하며, 행렬을 사용하여 $$ 𝐚𝑇𝐱 = 𝑏 $$와 같이 표현할 수 있다.

[이제부터 아래의 예시를 들어 뒷내용을 설명하고자 한다]

|Person ID|Weight|Height|is_smoking|Life-span|
|---|---|---|---|---|
|1|60kg|5.5ft|1|66|
|2|65kg|5.0ft|0|74|
|3|55kg|6.0ft|1|78|

사람의 몸무게,키,흡연유무에 관한 데이터를 통해 수명을 예측한다고 가정해보자.   

$$ 60x_1+5.5x_2+1x_3=66 $$   

$$ 65x_1+5.0x_2+0x_3=74 $$   

$$ 55x_1+6.0x_2+1x_3=78 $$   

우리는 해당 정보를 위와같이 linear system으로 표현할 수 있다.    

반면, matrix를 활용하여 표현할 수도 있다.   

$$ A = \begin{bmatrix} 60 & 5.5 & 1 \\ 65 & 5.0 & 0 \\ 55 & 6.0 & 1 \end{bmatrix}, x= \begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix}, b= \begin{bmatrix} 66 \\ 74 \\ 78 \end{bmatrix}$$   

결과적으로, 여러개의 Liear equation은 single matrix equation으로 변환할 수 있다.   


### Point1) x의 해는 어떻게 구하는가?
여러방법이 있겠지만 첫번째 방법으로는 Inverse Matrix를 사용하는 것이다.x를 풀어내기에 앞서, Identity&Inverse Matrix에 대해 알아보고 넘어가자   
> **[Identity Matrix]**   
> 항등행렬은 대각성분이 모두 1이고, 나머지는 0인 squatre matrix이다.   
>  ex) $$ I_3= \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$$   
> - $$I_n \in \mathbb{R}^{n \times{n}}$$ 으로 표기한다
> - 항등행렬은 어떠한 x벡터든($$x \in \mathbb{R}^{n}$$) 해당 벡터를 곱하면 그 벡터를 반환한다(즉, 보존가능하다);   
> $$I_{n}\mathbf{x}=\mathbf{x}$$
> 
> **[Inverse Matrix]**   
> square matirx인 $A \in \mathbb{R}^{n \times n}$에 대해서,   
> $A^{-1}A=AA^{-1}=I_n$   
> 을 만족하는 $A^{-1}$을 의미한다.역행렬은 다음과 같이 구할 수 있다;   
> ex) $$A=\begin{bmatrix} a & b \\ c & d \end{bmatrix}$$ 에 대해서, $$A^{-1}= {1 \over {ad-bc}} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix} $$   
> *역행렬을 구하는 방법을 간단히 표기해뒀지만 더 구체적으로 알고싶다면, Ch1.2의 Gaussian elimination, row reduction, echelon form를 읽어보면된다.*

역행렬을 통해 $$Ax=b$$를 다음의 전개로 풀어낼 수 있다.   

$$ Ax=b $$  

$$ A^{-1}Ax=A^{-1}b $$   

$$ I_nx=A^{-1}b $$   

$$ x=A^{-1}b $$   

예제를 통해서 풀어보면, 다음과 같다; Ax=b

$$ \begin{bmatrix} 60 & 5.5 & 1 \\ 65 & 5.0 & 0 \\ 55 & 6.0 & 1 \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix} = \begin{bmatrix} 66 \\ 74 \\ 78 \end{bmatrix} $$   

$$ A^{-1} = \begin{bmatrix} 0.087 & 0.0087 & -0.087 \\ -1.1304 & 0.087 & 1.1314 \\ 2 & -1 & -1 \end{bmatrix} $$   

$$x=A^{-1}b= \begin{bmatrix} 0.087 & 0.0087 & -0.087 \\ -1.1304 & 0.087 & 1.1314 \\ 2 & -1 & -1 \end{bmatrix}\begin{bmatrix} 66 \\ 74 \\ 78 \end{bmatrix} = \begin{bmatrix} -0.4 \\ 20 \\ -20 \end{bmatrix} $$

위의 식에 따라, life-span=-0.4x(weight)+20x(height)-20x(is_smoking) 으로 표현할 수 있다.   
그러나 여기는 전제조건이 붙는다. => *"A가 inverse Matrix를 가질 수 있는 행렬이여야 한다"*


> 그럼 우리는 다음과 같은 질문을 생각할 수 있다.   
> 1. 어떤 조건을 충족해야만 행렬이 Inverse Matrix를 갖는가
> 2. Inverse Matrix를 가지지 못하는 행렬이 주어졌을 때, 어떻게 x의 해를 구하는가


### Point2) 어떤 조건이 충족되어야 A는 Inverse Matrix를 갖는가?
이는 행렬 A의 **determinant** 값을 알면 알 수 있다.   

determinant(행렬식)은 행렬을 대표하는 값으로 nxn의 square matix A에 대해 다음과 같이 정의된다;   

$$\begin{align*}
detA&={ a }_{ 11 }det{ A }_{ 11 }-{ a }_{ 12 }det{ A }_{ 12 }+...+{ (-1) }^{ 1+n }det{ A }_{ 1n }\\ &=\sum _{ j=1 }^{ n }{ { (-1) }^{ 1+j }{ a }_{ 1j }det{ A }_{ 1j } } 
\end{align*}$$

- $$det{ A }_{ ij }$$이란 A에서 1행과 1열을 제외한 행렬의 행렬식을 의미한다. 
- 2x2 행렬의 요소값이 a,b,c,d라고 할때($$\begin{bmatrix} a & b \\ c & d \end{bmatrix}$$) detA은 $$ad-bc$$이다!   

결과적으로, $$det A \ne 0$$이면 invertible하다(Inverse Matirx를 갖는다).   
반대로, $$det A = 0$$인 경우 Inverse Matrix를 가지지 못한다.   

이유는 간단하다; Inverse Matrix를 구하는 식을 다시한번 생각해보자.   

$$A^{-1}= {1 \over {ad-bc}} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix} $$   

- 역행렬을 구할때, determinant값이 분모값으로 곱해진다. 당연히 0이되는 경우, 해당식은 성립되지 않으면서 역행렬을 구할 수 없게된다.   

### Point3) non-Inverrtible Matrix인 A가 주어졌을 때, x의 해는 어떻게 구하는가?
결론부터 말하자면, A가 non-invertible한 경우 Ax=b는 해가 없거나(*no solution*) 무수히 많다(*infinitely many solution*).   
정방행렬이 아닌 모든 rectangular matrix A($$\in \mathbb{R}^{m \times n}$$)는 non-invertible하다.  

Ax=b --> $$ \begin{bmatrix} 60 & 5.5 & 1 \\ 65 & 5.0 & 0 \\ 55 & 6.0 & 1 \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix} = \begin{bmatrix} 66 \\ 74 \\ 78 \end{bmatrix} $$   

이때 m = equation 수, n= 변수 수 다.
1. m>n 인 경우, equation의 수가 변수 보다 많은 경우다;   
   equation이 중복되는 경우가 아닌 이상, 해를 가지지 못한다(over-determined system)    
2. m<n인 경우, equation의 수가 변수보다 적은 경우다;
    대부분의 경우 해가 무수히 많다(under-determined system)


## 2. Linear Combination
Matrix equation을 **vector**로도 표현할 수 있다.

$$ c_1v_1+ \cdots + c_pv_p$$   

vector들이 주어졌을때, scalars(**weights or coefficienet**라 부름)와의 결합(linear combination)으로 표현한다.   
linear combination을 matrix equation에 적용하면 다음과 같이 나타낼 수 있다;

$$ \begin{bmatrix} 60 & 5.5 & 1 \\ 65 & 5.0 & 0 \\ 55 & 6.0 & 1 \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix} = \begin{bmatrix} 66 \\ 74 \\ 78 \end{bmatrix} $$  

$$\begin{bmatrix} 60 \\ 65 \\ 55 \end{bmatrix}x_1+\begin{bmatrix} 5.5 \\ 5.0 \\ 6.0 \end{bmatrix}x_2 + \begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix}x_3 = \begin{bmatrix} 66 \\ 74 \\ 78 \end{bmatrix}$$

즉, 지금껏 봐와던 Ax=b는 $$\mathbf{a_1}x_1 + \mathbf{a_2}x_2 + \mathbf{a_3}x_3 = \mathbf{b}$$ 와 같이 표현할 수 있다.

### point4) **vector equation에서는 해를 어떻게 구할 수 있을까?**
이를 알아보기전에 span개념을 먼저 알아야한다.
> **[Span]**   
> $$v_1, \cdots , v_p$$ ($$ \in \mathbb{R}^n $$)이 주어졌을 때, $$Span\{ v_1, \cdots , v_p \}$$은 해당 벡터들의 모든 조합을 통해 정의되는 공간이다.
> - 따라서, $$Span\{v_1, \cdots , v_p\}$$은   
>    $$ c_1v_1+ \cdots + c_pv_p$$으로 표현 할 수 있고, 이때 c_1...c_p는 임의의 모든 실수가 들어갈 수 있다.
> - $$Span\{v_1, \cdots , v_p\}$$은 *subset of $$\mathbb{R}^n$$ spanned by $$v_1,\cdots, v_p$$*라고 말하기도 한다.
> 
> 기하학적 관점에서 span에 대해 이해해보자.  
> ![figure1](figure1.png)   
> 3차원공간의 0벡터가 아닌 두 v_1과 v_2이 주어졌을때, v_1에 어떤 scalar를 곱하더라도 v_2이 표현되지 않는다면 $$Span\{v_1, v_2\}$$는 3차원 상의 평면(plane)을 의미하게된다.(v_1, v_2, 0 벡터를 포함하는)

다시 vector equation으로 돌아와서, 

$$\begin{bmatrix} 60 \\ 65 \\ 55 \end{bmatrix}x_1+\begin{bmatrix} 5.5 \\ 5.0 \\ 6.0 \end{bmatrix}x_2 + \begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix}x_3 = \begin{bmatrix} 66 \\ 74 \\ 78 \end{bmatrix}$$

이러한 형태의 vector equation이 해를 갖기 위해선, $$\mathbf{b} \in Span{\mathbf{a_1},\mathbf{a_2},\mathbf{a_3}}$$여야 한다.    
=> 즉, b가 a_1, a_2, a_3들로 표현할 수 있는 공간 상에 있어야 된다는 것이다.   
- 예를 들자면, xy평면을 표현하는 v1, v2 벡터가 있을때, 당연하게도 두벡터를 가지고 z축에 있는 값을 표현하지는 못하지 않는가

vector equation을 알았다면 우리는 4가지 방법을 통해 행렬을 쪼개 표현할 수 있다.
1. inner product
   
   $$ \begin{bmatrix} 60 & 5.5 & 1 \\ 65 & 5.0 & 0 \\ 55 & 6.0 & 1 \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix} = A\mathbb{x} = \begin{bmatrix} \mathbb{a_1} & \mathbb{a_2} & \mathbb{a_3} \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix} = \mathbb{a_1}x_1+\mathbb{a_2}x_2+\mathbb{a_3}x_3 $$
   
2. column combination
   수식 쓰기 귀찮아
3. row combination
4. **sum of rank-1 outer product**
   - Rank가 1인 matrix의 outer product 합으로도 행렬을 표현할 수 있다.

## 3. Linear Independence

$$\begin{bmatrix} 60 \\ 65 \\ 55 \end{bmatrix}x_1+\begin{bmatrix} 5.5 \\ 5.0 \\ 6.0 \end{bmatrix}x_2 + \begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix}x_3 = \begin{bmatrix} 66 \\ 74 \\ 78 \end{bmatrix}$$

$$\mathbb{a_1}x_1+\mathbb{a_2}x_2+\mathbb{a_3}x_3 = \mathbb{b}$$

우리는 앞선 vector equation의 해를 구하는 예제에서 b가 span 영역에 있어야 해가 존재한다는 것을 알았다. 그렇다면,

### point5) 해가 unique한지 알아보려면 어떻게 해야할까?
결론부터 말하자면, 주어진 행렬 A의 vector set($\mathbb{a_1},\mathbb{a_2},\mathbb{a_3}$)이 **linearly independent**한지 알아보면 된다.   
반면, $\mathbb{a_1},\mathbb{a_2},\mathbb{a_3}$가 서로 **linearly dependent**하다면 무수히 많은 해가 존재할 것이다.   

**Linear Dependent**란,
$$v_1, \cdots, v_p \in \mathbb{R}^{n}$$ 이 주어졌을 때, 이 중 임의로 뽑은 v_j가 $$\{v_1, v_2, \cdots , v_{j-1}\}$$들의 combination으로 표현 가능한 벡터인 경우 linearly dependent하다고 한다. 
- 그러한 v_j가 $$\{v_1, \cdots, v_p \}$$에서 하나도 없다면, linearly independent하다고 한다.

### point6) vector set이 linear independent한지 어떻게 알수 있는가?
위의 정의대로 하나하나 모든 벡터들이 조건을 성립하는지 보는 일은 여간 성가시다. 우리는 간단한 방법으로도 이를 알 수 있다.   
바로, $$x_1\mathbf{v1}+x_2\mathbf{v2}+\cdots+x_3\mathbf{v3}=\mathbf{0}$$ 식(homogeneous equation)을 세워, x의 해가 0 vector만 존재하는지 알아보면 된다.   

homogenous equation에서 해를 하나만 가지고, 아래와 같은 해를 가질때;   

$$\mathbf{x}= \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_p \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \\ \vdots \\ 0 \end{bmatrix}$$

우리는 **trivial solution**을 가진다고 한다.   
- $$v_1, \cdots, v_p$$가 linearly independent하다면, trivial solution을 가진다.
- 반면, $$v_1, \cdots, v_p$$가 linearly dependent하다면, non trivial solution을 가진다. 즉, 적어도 한개의 x_j가 0이 아닐것이다.

이를 증명해보면, 임의의 x_j가 0이 아닌경우   

$$x_jv_j= -x_1v_1-\cdots-x_{j-1}v_{j-1}$$

$$v_j= -{x_1 \over x_j}v_1-\cdots-{x_{j-1} \over x_j}v_{j-1} \in Span\{v_1,v_2,\cdots, v_{j-1}\}$$


이전 span을 설명하던 figure1을 상기시켜보자.   
v1과 v2로 선형결합과 일치하는 v3이 있을 때, v3은 두 벡터성분으로 표현 가능하기 때문에 Span{v1,v2}에 포함될 것이다.    
즉, **linearly dependent vector는 Span을 확장시키지 않는다!**    
- v3=av1+bv2로 표현되어진다면, c1v1+c2v2+c3v3= (c1+a)v1+(c2+b)v2 이다.   
- 따라서, v_3가 Span{v_1,v_2}에 속한다면, Span{v1,v2,v3}=Span{v1,v2} 이 성립된다.
  

그럼 이제, 벡터들이 linear dependent한 경우 vector equation이 어떻게 전개되는지 몸소 알아보자!

$$x_1v_1+x_2v_2+x_3v_3=b$$일 때, $$x=\begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix} = \begin{bmatrix} 3 \\ 2 \\ 1 \end{bmatrix}$$에 대해서   
$$3v_1+2v_2+1v_3=b$$로 표현 할 수 있다.   
이때 $$v_3=2v_1+3v_2$$가 주어지면,   
$$3v_1+2v_2+1v_3=3v_1+2v_2+(2v_1+3v_2)=5v_1+5v_2$$가 된다.   
따라서, $$x=\begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix} = \begin{bmatrix} 5 \\ 5 \\ 0 \end{bmatrix} $$이 또다른 해가 될 수 있다.




