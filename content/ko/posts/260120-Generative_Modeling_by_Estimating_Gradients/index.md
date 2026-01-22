---
title: Score-based generative modeling
date: 2026-01-20
author: onebom
description: Yang Song - Generative Modeling by Estimating Gradients of the Data Distribution
isStarred: true
draft: false
math: true
---
[Yang Song - Generative Modeling by Estimating Gradients of the Data Distribution](https://yang-song.net/blog/2021/score/)을 번역한 글입니다.
번역 과정에서 일부 의역을 포함했습니다.

## 1.Introduction

generative modeling 기술은 "어떻게 probability distribution"을 나타내는 지로 두 그룹으로 나눌 수 있습니다.
1. **likelihood-based models**   
  distribution의 P.D.F(probability density function)또는 P.M.F(probability mass function)을 maximum likelihood를 근사하는 방식으로 학습하는 방식.
    - autoregressive models, normalizing flow models, energy-based model(EBMs), variational auto-enocder(VAEs) 모델들이 보통 likelihood-based model 방식입니다.

2. **implicit generative models**  
  model의 sampling process를 사용해 probability distribution을 implicit하게 표현하는 방식.
    - 가장 대표적인 예는 generative adversarial network(GANs)입니다. random gaussian vector를 Neural network로 변환하여 데이터 분포에서 새로운 샘플을 합성합니다. 

<br>

<img src="1.png" width="1000"/>
<figcaption>[Bayesian networks, Markov random fields (MRF), autoregressive models, and normalizing flow models are all examples of likelihood-based models. All these models represent the probability density or mass function of a distribution]</figcaption>

<br>

<img src="2.png" width="1000"/>
<figcaption>[GAN is an example of implicit models. It implicitly represents a distribution over all objects that can be produced by the generator networks]</figcaption>

<br>

근데, 두 모델 전부 확실한 limitation을 가집니다.   
likelihood-based model은 likelihood를 계산하는 tractable normalizing constant를 보장하기 위해서 모델 아키텍처에 대한 강한 restriction를 필요로하거나, maximum likelihood를 근사하기 위한 간접 objective에 의존해야합니다.
반면, implicit generative model은 종종 adverarial training을 필요로 하며, 이 방식은 학습 안정성이 너무 낮고 mode collapse를 발생시킬 수 있습니다.

따라서 이 블로그 포스트에서는, 위 한계들을 우회할 수 있는 probability distribution 표현 방식을 소개하려고 합니다.
key idea는 *Log P.D.F의 gradient*인 **score function**를 modeling하는 것입니다.
**score-based model**은 tractable normalizing constance를 필요로 하지 않고, **score matching**을 통해서 직접적으로 학습이 가능합니다.

![3](3.jpg)*Score function (the vector field) and density function (contours) of a mixture of two Gaussians*

score-based model은 많은 downstream task(image generation, audio synthesis, shape generation, muision generation 등)에서 state-of-the-art를 달성했습니다.
또한, [normalizing flow model]()와 개념적으로 연결되어 있어 정확한 likelihood computation과 representation learning이 가능합니다.
score modeling&estimating은 inverse problem 해결을 용이하게 합니다(with applications such as image inpainting[18, 21] , image colorization [21] , compressive sensing, and medical image reconstruction (e.g., CT, MRI)).

![4](4.jpg)*Score function (the vector field) and density function (contours) of a mixture of two Gaussians.*

이 포스트에서 score-based generative modeling에 대한 motivation과 intuition, 뿐만아니라 basic concepts, properties을 보이고자 합니다.

## The score function, score-based models, and score matching
data distribution $p(x)$를 따르면서 i.i.d((independent and identically distributed)) data point로 구성된 dataset $ \lbrace x_1,x_2, \cdots, x_N \rbrace $이 주어진다고 가정하겠습니다.
generative modeling의 목표는 data distribution에 model을 fitting함으로써, 학습된 분포로부터 sampling하여 새로운 data point를 합성하는 것입니다.

그러한 generative model을 만들기 위해서는, 우선 probability distribution을 represent하는 방식이 필요합니다.
likelihood-based model에서 사용가능한 한가지는 방법으로는, P.D.F 또는 P.M.F를 직접적으로 modeling하는 것입니다.
학습 가능한 parameter $\theta$로 parameterized된 실수 함수 $f_{\theta}(x) \in \mathbb{R}$를 가정해 보겠습니다  
그럼 p.d.f를 다음과 같이 정의할 수 있습니다;

$$
p_{\theta}(x) = \frac{e^{-f_{\theta}(x)}}{Z_\theta}
$$

$Z_{\theta}>0$는 $\theta$에 의존하는 normalizing constant이므로 $\int p_{\theta}(x) \ d\mathbf{x}=1$가 됩니다. 여기서 f_{\theta}(x)는 unnormalized probabilistic model, 또는 energy-based model이라고 합니다.

우리는 $p_{\theta}(x)$를 data의 log-likelihood를 최대화함으로써 학습할 수 있습니다;

$$
\max_{\theta} \sum^N_{i=1} \log{p_{\theta}(\mathbf{x}_i)}
$$

그러나, 위 수식은 $p_{\theta}(x)$가 normalized probability density function여야 합니다.
이렇게되면, $p_{\theta}(x)$를 연산하기 위해서 normalizing constant인 $Z_{\theta}$를 알아야만 하며, 이는 일반적인 $f_{\theta}(x)$에 대해서 다루기 어려운 연산량을 요구하게 됩니다.
따라서, maximum likelihood 학습이 가능하도록 하기 위해서는 모델 아키텍처를 제한하여(e.g. causal convolutions in autoregressive mdoels, inverible networks in normalizing flow models) $Z_{\theta}$를 구할 수 있도록 하거나, 연산량이 엄청 비싸더라도 normalizing constant를 근사해야 했습니다(e.g. variational inference in VAEs, or MCMC sampling used in contrastive divergence). 

근데!, densitiy function 대신 score function을 모델링함으로써, 우리는 다루기 어려운 normalizing constants $Z_{\theta}$의 어려움을 피할 수 있습니다.  
distribution $p(x)$의 **score function**은 다음처럼 정의할 수 있습니다;

$$
\begin{equation} \nabla_\mathbf{x} \log p(\mathbf{x}), \notag \end{equation}
$$

그리고, socre function을 학습하는 model을 **score-based model**이라고 부르며, $s_{\theta}(x)$로 씁니다.
score-based model은 $s_\theta(x) \approx \nabla_{\mathbf{x}} \log{p(\mathbf{x})}$를 학습하고, normalizing constant에 대한 고려 없이도 parameterzied 가능합니다.
예를 들어, 아래 수식처럼 우리는 score-based model을 energy-based model(수식1)로 파라미터화 할 수 있습니다;  

<br>
$$
\begin{equation} \mathbf{s}_\theta (\mathbf{x}) = \nabla_{\mathbf{x}} \log p_\theta (\mathbf{x} ) = -\nabla_{\mathbf{x}} f_\theta (\mathbf{x}) - \underbrace{\nabla_\mathbf{x} \log Z_\theta}_{=0} = -\nabla_\mathbf{x} f_\theta(\mathbf{x}). \end{equation}
$$ 
<br> 

여기서 중요한건, score-based model $s_\theta (x)$는 normalizing constance $Z_\theta$와 독립이라는 점입니다! 


![5](5.gif)*Parameterizing probability density functions. No matter how you change the model family and parameters, it has to be normalized (area under the curve must integrate to one).*

![6](6.gif)*Parameterizing score functions. No need to worry about normalization.*


likelihood-based model들과 유사하게도, 우리는 모델과 실제 data distribution 사이 Fisher divergence를 최소화함으로써 score-based model을 학습할 수 있습니다.

<br>
$$
\begin{equation} \mathbb{E}_{p(\mathbf{x})}[\| \nabla_\mathbf{x} \log p(\mathbf{x}) - \mathbf{s}_\theta(\mathbf{x}) \|_2^2] \end{equation}
$$
<br>

직관적으로 설명하자면, Fisher divergence는 ground-truth data score와 score-based model 사이 $\ell_2$ distance의 제곱값을 비교합니다.
하지만, divergence를 직접 연산하는 것은 unknown data score $\nabla_\mathbf{x} \log p(\mathbf{x})$를 필요로 하기 때문에 불가능합니다.
다행히도, score matching이라는 방법들이 존재하며, 이 방법들은 ground-trurh data score에 대해서 모르더라도 Fisher divergence를 최소화합니다.
Score matching의 objectives는 dataset에서 직접 추정될 수 있으며, stochastic gradient descent를 통해 최적화될 수 있습니다.
이는 normalizing constant를 알고서 likelihood-based model을 학습시키기 위한 log-likelihood objective와 유사합니다.
이는 알려진 정규화 상수를 가진 확률 기반 모델을 훈련시키기 위한 로그 우도 목표와 유사합니다. 
우리는 **adversarial optimization 없이** score matching objective를 최소화하여 score-based model을 학습할 수 있습니다.

게다가, score matching objective를 사용하는 것은 우리에게 상당한 modeling flexibility를 줍니다.
Fisher divergence는 $s_\theta (x)$가 normalized distribution의 실제 score function일 필요가 없습니다. 단순히 ground-truth data score와 score-based modeldml $\ell_2$ distance를 비교할 뿐입니다.
실제로는, score-based model이 유일하게 필요로 하는것은 입력 및 출력의 차원이 동일한 vector-value function입니다.

간단히 요약하자면, score function을 모델링하여 분포를 나타낼 수 있으며, score matching을 통해 score-based model의 free-from 아키텍처를 학습시켜 score function을 추정할 수 있습니다. 

## Langevin dynamics
score-based model $s_\theta(x) \approx \nabla_x \log p(x)$를 학습시키고 나면, 우리는 **Langevin dynamics**라 불리는 iterative procedure를 사용해서 모델로부터 sample을 얻을 수 있습니다.


Langevin dynamics는 score function $\nabla_\mathbf{x} \log p(\mathbf{x})$만을 사용해서 distribution $p(x)$로부터 sampling하기 위한 MCMC procedure를 제공합니다. 
구체적으로, 임의의 prior distribution $\mathbf{x}_0 \sim \pi(\mathbf{x})$로부터 chain을 초기화 한 다음 아래 과정을 반복합니다;  

<br>
$$
\begin{align} \mathbf{x}_{i+1} \gets \mathbf{x}_i + \epsilon \nabla_\mathbf{x} \log p(\mathbf{x}) + \sqrt{2\epsilon}~ \mathbf{z}_i, \quad i=0,1,\cdots, K, \end{align}
$$
<br>

$z_i$는 가우시안 분포를 따릅니다; $\mathbf{z}_i \sim \mathcal{N}(0, I)$.
$\epsilon \to 0$이고 $K \to \infty$ 일 때, $x_K$는 몇 regularity condition을 따르며 위 절차를 통해 $p(x)$로 수렴합니다. 실제로, $\epsilon$이 충분히 작고 $K$가 충분히 큰 경우 오차는 무시될정도로 작아집니다.

![7](7.gif)*Using Langevin dynamics to sample from a mixture of two Gaussians.*

Langevin dynamics는 $\nabla_\mathbf{x} \log p(\mathbf{x})$를 통해서 $p(x)$에 접근한다는 것을 기억하십시오.
$s_\theta(x) \approx \nabla_x \log p(x)$이기에, 우리는 score-based model $s_\theta(x)$를 MCMC 수식에 대입하여 sample을 생성할 수 있습니다.

## Naive score-based generative modeling and its pitfalls
이제서야 우리는 어떻게 "score-based model이 score matching으로 학습되는 지"와 "Lagenvin dynamics를 통해 sample을 생성할 수 있는지"에 대해서 알게 되었습니다.
그러나, 실제로 이 navie한 방식을 적용하기에는 아직 한계가 있습니다.
지금까지는 주목하지 못했던 score matching방식의 함정에 대해서 이야기 해보겠습니다.

![8](8.jpg)*Score-based generative modeling with score matching + Langevin dynamics.*

주요 문제점은 score matching objective를 연산하기 위한 data point가 충분하지 않은 low density인 영역에서 score function을 추정하는 것은 부정확할 수 밖에 없다는 사실 입니다. 

score matching은 fisher divergence를 최소화하는데 하는 function입니다.
따라서, true data score function과 score-based model 사이 $\ell_2$ 차이값은 $p(x)$에 의해 가중되기 떄문에 $p(x)$가 작은 low density regions에서는 값이 대부분 무시될 것이기 떄문에 위와 같은 문제가 발생합니다.

<br>
$$
\mathbb{E}_{p(\mathbf{x})}[\| \nabla_\mathbf{x} \log p(\mathbf{x}) - \mathbf{s}_\theta(\mathbf{x}) \|_2^2] = \int p(\mathbf{x}) \| \nabla_\mathbf{x} \log p(\mathbf{x}) - \mathbf{s}_\theta(\mathbf{x}) \|_2^2 \mathrm{d}\mathbf{x}.
$$
<br>

이러한 문제는 아래 그림에서 볼 수 있듯이 저조한 결과를 보이게 됩니다.

![9](9.jpg)*Estimated scores are only accurate in high density regions.*

Langevie dynamics로 sampling할 때, high dimenstional space에 놓여있는 data에 대해서 initial sample은 low density region에서 시작할 가능성이 매우 높습니다.
따라서 부정확한 score-based model을 사용하면 MCMC sampling 과정의 시작부터 Langevin dynamics가 탈선하여 데이터를 대표하는 high quality를 생성하지 못하게 됩니다.

## Score-based generative modeling with multiple noise perturbations
## Score-based generative modeling with stochastic differential equations (SDEs) ⭐️
## Connection to diffusion models and others
## Concluding remarks