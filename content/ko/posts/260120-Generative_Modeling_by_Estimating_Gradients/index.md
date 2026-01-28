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

본 post는 generative modeling의 새로운 방향을 제시합니다. 포스트를 통해 noise-pertubated data distribution에 대한 score function(gradients of log probability density function)을 배우고, Langevin-type sampling으로 새로운 sample들을 생성해보고자 합니다. score-based generative model이라 불리는 생성모델은 기존 생성모델에 비해 몇가지 중요한 장점을 가집니다; 1)adversarial training이 없는 GAN 보다 좋은 수준의 sample quality, 2) 유연한 모델 아키텍처, 3) 정확한 log-likelihood computation, 마지막으로 4) re-training 필요없는 inverse problem solving. 이 포스트에서는 score-based generative model의 intuition, basic concepts, potential application에 대해 구체적으로 다루고자 합니다.

## 1.Introduction

generative modeling 기술은 "어떻게 probability distribution"을 나타내는 지로 두 그룹으로 나눌 수 있습니다.
1. **likelihood-based models**   
  distribution의 P.D.F(probability density function)또는 P.M.F(probability mass function)을 maximum likelihood를 근사하는 방식으로 학습하는 방식.
    - autoregressive models, normalizing flow models, energy-based model(EBMs), variational auto-enocder(VAEs) 모델들이 보통 likelihood-based model 방식입니다.

2. **implicit generative models**  
  model의 sampling process를 사용해 probability distribution을 implicit하게 표현하는 방식.
    - 가장 대표적인 예는 generative adversarial network(GANs)입니다. random gaussian vector를 Neural network로 변환하여 데이터 분포에서 새로운 샘플을 합성합니다. 

<br>

<center>
<img src="1.png" width="800"/>
<figcaption>[Bayesian networks, Markov random fields (MRF), autoregressive models, and normalizing flow models are all examples of likelihood-based models. All these models represent the probability density or mass function of a distribution]</figcaption>
</center>

<br>

<center>
<img src="2.png" width="800"/>
<figcaption>[GAN is an example of implicit models. It implicitly represents a distribution over all objects that can be produced by the generator networks]</figcaption>
</center>

<br>

근데, 두 모델 전부 확실한 limitation을 가집니다.   
likelihood-based model은 likelihood를 계산하는 tractable normalizing constant를 보장하기 위해서 모델 아키텍처에 대한 강한 restriction를 필요로하거나, maximum likelihood를 근사하기 위한 간접 objective에 의존해야합니다.
반면, implicit generative model은 종종 adverarial training을 필요로 하며, 이 방식은 학습 안정성이 너무 낮고 mode collapse를 발생시킬 수 있습니다.

따라서 이 블로그 포스트에서는, 위 한계들을 우회할 수 있는 probability distribution 표현 방식을 소개하려고 합니다.
key idea는 *Log P.D.F의 gradient*인 **score function**를 modeling하는 것입니다.
**score-based model**은 tractable normalizing constance를 필요로 하지 않고, **score matching**을 통해서 직접적으로 학습이 가능합니다.

<center>
<img src="3.jpg" width="300"/>
<figcaption>[Score function (the vector field) and density function (contours) of a mixture of two Gaussians]</figcaption>
</center>

score-based model은 많은 downstream task(image generation, audio synthesis, shape generation, muision generation 등)에서 state-of-the-art를 달성했습니다.
또한, [normalizing flow model]()와 개념적으로 연결되어 있어 정확한 likelihood computation과 representation learning이 가능합니다.
score modeling&estimating은 inverse problem 해결을 용이하게 합니다(with applications such as image inpainting, image colorization , compressive sensing, and medical image reconstruction (e.g., CT, MRI)).

<center>
<img src="4.jpg" width="800"/>
<figcaption>[1024 x 1024 samples generated from score-based models]</figcaption>
</center>


## 2. The score function, score-based models, and score matching
data distribution $p(x)$를 따르면서 i.i.d((independent and identically distributed)) data point로 구성된 dataset $ \lbrace x_1,x_2, \cdots, x_N \rbrace $이 주어진다고 가정하겠습니다.
generative modeling의 목표는 data distribution에 model을 fitting함으로써, 학습된 분포로부터 sampling하여 새로운 data point를 합성하는 것입니다.

그러한 generative model을 만들기 위해서는, 우선 probability distribution을 represent하는 방식이 필요합니다.
likelihood-based model에서 사용가능한 한가지는 방법으로는, P.D.F 또는 P.M.F를 직접적으로 modeling하는 것입니다.
학습 가능한 parameter $\theta$로 parameterized된 실수 함수 $f_{\theta}(x) \in \mathbb{R}$를 가정해 보겠습니다  
그럼 p.d.f를 다음과 같이 정의할 수 있습니다;

$$
\begin{align} p_\theta(\mathbf{x}) = \frac{e^{-f_\theta(\mathbf{x})}}{Z_\theta}, \end{align}
$$

$Z_{\theta}>0$는 $\theta$에 의존하는 normalizing constant이므로 $\int p_{\theta}(x) \ d\mathbf{x}=1$가 됩니다. 여기서 f_{\theta}(x)는 unnormalized probabilistic model, 또는 energy-based model이라고 합니다.

우리는 $p_{\theta}(x)$를 data의 log-likelihood를 최대화함으로써 학습할 수 있습니다;

$$
\begin{align} \max_\theta \sum_{i=1}^N \log p_\theta(\mathbf{x}_i). \end{align}
$$

그러나, 위 수식은 $p_{\theta}(x)$가 normalized probability density function여야 합니다.
이렇게되면, $p_{\theta}(x)$를 연산하기 위해서 normalizing constant인 $Z_{\theta}$를 알아야만 하며, 이는 일반적인 $f_{\theta}(x)$에 대해서 다루기 어려운 연산량을 요구하게 됩니다.
따라서, maximum likelihood 학습이 가능하도록 하기 위해서는 모델 아키텍처를 제한하여(e.g. causal convolutions in autoregressive mdoels, inverible networks in normalizing flow models) $Z_{\theta}$를 구할 수 있도록 하거나, 연산량이 엄청 비싸더라도 normalizing constant를 근사해야 했습니다(e.g. variational inference in VAEs, or MCMC sampling used in contrastive divergence). 

근데!, densitiy function 대신 score function을 모델링함으로써, 우리는 다루기 어려운 normalizing constants $Z_{\theta}$의 어려움을 피할 수 있습니다. distribution $p(x)$의 **score function**은 다음처럼 정의할 수 있습니다;

$$
\begin{equation} \nabla_\mathbf{x} \log p(\mathbf{x}), \notag \end{equation}
$$

그리고, socre function을 학습하는 model을 **score-based model**이라고 부르며, $s_{\theta}(x)$로 씁니다.
score-based model은 $s_\theta(x) \approx \nabla_{\mathbf{x}} \log{p(\mathbf{x})}$를 학습하고, normalizing constant에 대한 고려 없이도 parameterzied 가능합니다.
예를 들어, 아래 수식처럼 우리는 score-based model을 energy-based model(eq 1)로 파라미터화 할 수 있습니다;  

<br>
$$
\begin{equation} \mathbf{s}_\theta (\mathbf{x}) = \nabla_{\mathbf{x}} \log p_\theta (\mathbf{x} ) = -\nabla_{\mathbf{x}} f_\theta (\mathbf{x}) - \underbrace{\nabla_\mathbf{x} \log Z_\theta}_{=0} = -\nabla_\mathbf{x} f_\theta(\mathbf{x}). \end{equation}
$$ 
<br> 

여기서 중요한건, score-based model $s_\theta (x)$는 normalizing constance $Z_\theta$와 독립이라는 점입니다! 

<br>

<center>
<img src="5.gif" width="500"/>
<figcaption>[Probability density functions 파라미터화. model famility(아키텍처)와 pramters들이 어떻게 변하든 정규화되어야 합니다.(곡선 아래 면적이 1이 되어야만 합니다)]</figcaption>
</center>

<center>
<img src="6.gif" width="500"/>
<figcaption>[Score functions 파라미터화. normalization에 대한 고려는 안해도 됩니다.]</figcaption>
</center>

<br>

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

## 3. Langevin dynamics
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

<center>
<img src="7.gif" width="300"/>
<figcaption>[Langevin dynamics를 사용한 두 gaussian mixture에서의 sampling 과정]</figcaption>
</center>

Langevin dynamics는 $\nabla_\mathbf{x} \log p(\mathbf{x})$를 통해서 $p(x)$에 접근한다는 것을 기억하십시오.
$s_\theta(x) \approx \nabla_x \log p(x)$이기에, 우리는 score-based model $s_\theta(x)$를 MCMC 수식에 대입하여 sample을 생성할 수 있습니다.

## 4.Naive score-based generative modeling and its pitfalls
이제서야 우리는 어떻게 "score-based model이 score matching으로 학습되는 지"와 "Lagenvin dynamics를 통해 sample을 생성할 수 있는지"에 대해서 알게 되었습니다.
그러나, 실제로 이 navie한 방식을 적용하기에는 아직 한계가 있습니다.
지금까지는 주목하지 못했던 score matching방식의 함정에 대해서 이야기 해보겠습니다.

<center>
<img src="8.jpg" width="800"/>
<figcaption>[Score matching + Langevin dynamics를 사용한 score-based generative modeling]</figcaption>
</center>

주요 문제점은 score matching objective를 연산하기 위한 data point가 충분하지 않은 low density인 영역에서 score function을 추정하는 것은 부정확할 수 밖에 없다는 사실 입니다. 

score matching은 fisher divergence를 최소화하는데 하는 function입니다.
따라서, true data score function과 score-based model 사이 $\ell_2$ 차이값은 $p(x)$에 의해 가중되기 때문에 $p(x)$가 작은 low density regions에서는 값이 대부분 무시되어 위와 같은 문제가 발생합니다.

<br>
$$
\mathbb{E}_{p(\mathbf{x})}[\| \nabla_\mathbf{x} \log p(\mathbf{x}) - \mathbf{s}_\theta(\mathbf{x}) \|_2^2] = \int p(\mathbf{x}) \| \nabla_\mathbf{x} \log p(\mathbf{x}) - \mathbf{s}_\theta(\mathbf{x}) \|_2^2 \mathrm{d}\mathbf{x}.
$$
<br>

이러한 문제는 아래 그림에서 볼 수 있듯이 저조한 결과를 보이게 됩니다.

<center>
<img src="9.jpg" width="800"/>
<figcaption>[score를 추정하는 것은 high denstiy regison에서만 정확하다]</figcaption>
</center>

Langevie dynamics로 sampling할 때, high dimenstional space에 놓여있는 data를 생성하기까지의 과정에서 initial sample은 low density region에서 시작할 가능성이 매우 높습니다.
따라서 부정확한 score-based model을 사용하면 MCMC sampling 과정의 시작부터 Langevin dynamics가 탈선하게되어 데이터를 대표하는 high quality를 생성하지 못하게 됩니다.

## 5.Score-based generative modeling with multiple noise perturbations
그렇다면, low data densitiy 영역에서 정확한 score matching을 구하기 어려운 문제를 어떻게 우회해서 해결할 수 있을까요?  
해결책은 data point들에 noise를 섞고(perturb), pure한 data point 대신 noisy data points에 대해서 score-based models를 학습시키는 것입니다.
noise magnitude가 충분히 커지게 되면, low data density region을 다룰 수 있게되어 score 추정의 정확도를 높일 수 있습니다.
예로, 두 Gaussians mixture에 Gaussian noise를 섞어봅시다.

<center>
<img src="10.jpg" width="800"/>
<figcaption>[low data density regions는 감소하여 노이즈에 의해 perturb(교란된) 데이터 분포에 대한 score 추정값은 정확해집니다. in everywhere! ]</figcaption>
</center>

그럼에도 여전히 한가지 의문점이 듭니다; perturbation process에 쓰이는 noise scale을 어떻게 정해야하는가?  
noise가 너무 크다면, low density 영역을 많이 커버할 수 있지만 데이터를 너무 망가뜨려(over-corrupt) 실제 데이터 분포와 크게 달라지게 됩니다.
noise가 너무 작다면, 기존 데이터 분포와는 많이 달라지지 않지만, low density 영역까지는 모델이 커버할 수 없게 됩니다.

두 문제 다 일어나지 않게 충족하기 위해서는, 다양한 scale의 noise를 사용해서 모델이 전부 사용하도록 해야합니다.
항상 isotriopic Gaussian noise만 사용해서 data를 perturb하고, noise의 표준편차가 총 $L$ step에 거쳐 증가한다고 가정해봅시다 ($\sigma_1 < \sigma_2 < \cdots < \sigma_L$). 
그리고 data distribution $p(x)$에 gaussian noise $N(0, \sigma^2_i I), i=1,2,\cdots,L$를 더해서 noise-perturbed distribution을 얻습니다;

$$
p_{\sigma_i}(\mathbf{x}) = \int p(\mathbf{y}) \mathcal{N}(\mathbf{x}; \mathbf{y}, \sigma_i^2 I) \mathrm{d} \mathbf{y}.
$$

주목해야하는 부분은, guassian noise $z$로 $x+\sigma_i z$를 얻는 방식과 $x ~ p(x)$을 sampling하는 방식을 사용해서 $p_{\sigma_{i}}(x)$로부터 sample을 쉽게 얻을 수 있다는 것입니다.   


다음으로, 모든 $i=1,2,\cdots,L$에 대해 $s_\theta(\mathbf{x}, i) \approx \nabla_\mathbf{x} \log p_{\sigma_i}(\mathbf{x})$)를 구하는 score matching을 사용하여 **Noise Conditional Score-based model** $s_\theta(x, i)$을 학습시킴으로써 noise-perturbed distribution $\nabla_\mathbf{x} \log p_{\sigma_i}(\mathbf{x})$의 score function을 추정할 수 있습니다.

<center>
<img src="11.jpg" width="800"/>
<figcaption>[첫번째 열은 data distribution에 multiple-scale의 gaussian noise를 더해 perturb된 모습을 보여줍니다. 두번째 열은 perturb된 distribution에서 score function을 jointly 추정한 모습입니다.]</figcaption>
</center>

<center>
<img src="12.jpg" width="800"/>
<figcaption>[image에 multiple-scale의 guassian noise를 더해 perturbing한 예시입니다.]</figcaption>
</center>

$s_\theta(x,i)$의 학습 목적은 모든 noise scales의 Fisher divergences를 wieghted sum하는 것입니다.
특히나, 우리는 아래의 objective를 사용합니다;

<br>
$$
\begin{equation} \sum_{i=1}^L \lambda(i) \mathbb{E}_{p_{\sigma_i}(\mathbf{x})}[\| \nabla_\mathbf{x} \log p_{\sigma_i}(\mathbf{x}) - \mathbf{s}_\theta(\mathbf{x}, i) \|_2^2],  \end{equation}
$$
<br>

$\lambda(i) \in \mathbb{R}_{> 0}$는 positive weighting function이고, 주로 $\lambda(i)=\sigma^2_i$로 선택합니다. 위에 적힌 목적함수는 naive(unconditional) score-based model 경우 때와 마찬가지로 score matching으로 최적화됩니다.

noise-conditional score based model $s_\theta(x,i)$을 학습하고 난 뒤, 우리는 Langevin dynamics를 $i=L,L-1,\cdots,1$에 대해 차례로 실행함으로써 모델로부터 sample을 생성할 수 있다.
이 방법은 noise scale $\sigma_i$를 점진적으로 감소(anneals)하기에 **annealed Langevin dynamics**라고 부릅니다.

<center>
<img src="13.gif" width="800"/>
<figcaption>[Annealed Langevin dynamics는 점진적으로 감소하는 noise scale을 가진 Langevin chain squence를 결합합니다.]</figcaption>
</center>

<br>

<center>
<img src="14.gif" width="500"/>
<figcaption>[Annealed Langevin dynamics for the Noise Conditional Score Network (NCSN) model trained on CelebA (left) and CIFAR-10 (right). We can start from unstructured noise, modify images according to the scores, and generate nice samples. The method achieved state-of-the-art Inception score on CIFAR-10 at its time.]</figcaption>
</center>

<br>

다음으로, multiple noise scale을 가진 score-based generative models을 튜닝하기 위한 권장 사항입니다;
1. [Geometric Progression](https://en.wikipedia.org/wiki/Geometric_progression#:~:text=In%20mathematics%2C%20a%20geometric%20progression,number%20called%20the%20common%20ratio.)으로 $\sigma_1 < \sigma_2 < \cdots < \sigma_l$을 선택하고, $\sigma_L$은 충분히 작은 값으로 모든 training data points와의 pairwise distance 최댓값과 유사하도록 합니다.
2. U-net의 skip connection을 써서 score-based model $s_\theta(x,i)$를 파라미터화 합니다.
3. test time에서는 score-based model의 weight에 exponential moving average(EMA)를 적용합니다.

이러한 모범 사례를 통해 아래와 같은 다양한 데이터셋에서 GAN과 비슷한 품질의 고품질 이미지 샘플을 생성할 수 있습니다:

<center>
<img src="15.jpg" width="800"/>
<figcaption>[Samples from the NCSNv2 model. From left to right: FFHQ 256x256, LSUN bedroom 128x128, LSUN tower 128x128, LSUN church_outdoor 96x96, and CelebA 64x64]</figcaption>
</center>

## 6.Score-based generative modeling with stochastic differential equations (SDEs) ⭐️

앞서 논의했듯이, multiple noise scale을 도입하는 것은 score-based generative model의 성공에 있어 핵심적인 요소이다. 
noise scale의 개수를 무한대로 일반화하면, **샘플의 품질이 향상**될 뿐만 아니라 **정확한 로그 우도(log-likelihood) 계산**과 **역문제(inverse problem) 해결을 위한 제어 가능한 생성(controllable generation)** 등 여러 이점을 얻을 수 있다.

### Perturbing  data with a SDE
noise scale이 무한으로 근사할 때, 우리는 연속적으로 증가하는 noise level을 사용해서 필수적으로 데이터 분포를 perturb합니다.
이 경우, noise perturbation 절차는 아래 묘사되는 것처럼 [continous-time stochastic process](https://en.wikipedia.org/wiki/Stochastic_process#:~:text=A%20stochastic%20process%20is%20defined,measurable%20with%20respect%20to%20some)입니다.

<center>
<img src="16.gif" width="800"/>
<figcaption>[연속적인 stochastic 프로세스를 사용해 data를 noise로 perturbing합니다.]</figcaption>
</center>

stochastic process를 어떻게 명확하게 설명할 수 있을까요? 
많은 stochastic process들(특히, [diffusion process]())는 stochastic differential equation(SDEs)의 해입니다. 
일반적으로 SDE는 다음과 같은 형태를 갖습니다;
$$
\begin{align} \mathrm{d}\mathbf{x} = \mathbf{f}(\mathbf{x}, t) \mathrm{d}t + g(t) \mathrm{d} \mathbf{w}, \end{align}
$$

$f(\dot,t)$는 dirft coefficient라는 vector-valued function이고, $g(t)$는 diffusion coefficient라 불리는 real-value function입니다.
$\mathbf{w}$는 [standard Brownian motion]()을 나타냅니다. 
그리고, $d\mathbf{w}$는 infinitesimal white noise로 볼 수 있습니다.
SDE의 해는 random variables $\{\mathbf{x}(t)\}_{t \in [0,T]}$의 continuous collection입니다.
이 random variable들은 time index t가 0에서 T로 증가하는 동안의 stochastic trajectory를 따릅니다.  

$p_t(x)$를 $\mathbf{x}(t)$에 대한 p.d.f라고 해봅시다.
유한한 수의 noise scale을 가진다면, $t\in[0,T]$는  $i=1,\cdots,L$와 유사해지고 $p_t(x)$는 $p_{\sigma_i}(x)$와 유사해집니다.
명확하게 말하면, $p_0(x)=p(x)$는 perturbation이 없었던 t=0에서의 pure한 데이터 분포를 말합니다.
충분히 긴 T에 대해서 stochastic process를 거쳐 perturbing된 $p_T(x)$는 추적가능한 noise 분포이며 *prior distribution*이라고 부르는 $\phi(x)$와 유사해집니다.  
이제부터, $p_T(x)$와 $p_{\sigma_L} (x)$ 둘은 $\sigma_L$이 충분히 크고 유한한 noise scale의 경우에선 유사하다는 것을 주목해야합니다. 

noise perturbation을 더하는 방법은 여러 가지가 있습니다.
예를 들어, 다음과 같은 SDE는; 
$$
\begin{align} \mathrm{d}\mathbf{x} = e^{t} \mathrm{d} \mathbf{w} \end{align}
$$
$\sigma_1 < \sigma_2 < \cdots < \sigma_L$에 따라 $N(0,\sigma^2_1 I), N(0,\sigma^2_2 I), \cdots, N(0,\sigma^2_L I)$와 같이 평균이 0이면서 variance가 exponentially 증가하는 가우시안 노이즈를 더합니다.
일반적으로 SDE에 잘 작동하는 noise perturbation 방법으로는 3가지가 있습니다;   
1)Varaince Exploding SDE(VE SDE, 위 예시), 2)Varaince Preserving(VP SDE), 3)sub-VP SDE

### Reversing the SDE for sample generation
유한한 noise scale을 사용하면, **annealed Langevin dynamics**으로 perturbation에 대한 역과정으로 sample을 얻을 수 있다는 사실을 다시 상기해봅시다.
i.e. Langevin dynamics를 사용하면 각 noise perturbation 분포에서 순차적 sampling이 가능했습니다.
반면, 무한한 noise scale의 경우는 reverse perturbation process와 유사한 reverse SDE로 sample 생성이 가능합니다.

<center>
<img src="17.gif" width="800"/>
<figcaption>[perturbation process를 반대로 수행하여 noise에서 data를 생성합니다.]</figcaption>
</center>

중요한건, SDE는 모두 그에 대응되는 reverse SDE를 갖는다는 것입니다. 그에 대한 closed form은 다음과 같습니다;

$$
\begin{equation} \mathrm{d}\mathbf{x} = [\mathbf{f}(\mathbf{x}, t) - g^2(t) \nabla_\mathbf{x} \log p_t(\mathbf{x})]\mathrm{d}t + g(t) \mathrm{d} \mathbf{w}.\end{equation}
$$

여기서 SDE가 time 역순($t=T$ to $t=0$)으로 backward에서도 solved되어야 하기 때문에 $dt$는 negative 무한 time step을 의미합니다.
reverse SDE를 연산하기 위해서, 우리는 $p_t(x)$의 score function인 $\nabla_\mathbf{x} \log p_t(\mathbf{x})$를 정확히 추정해야만 합니다.

<center>
<img src="18.jpg" width="800"/>
<figcaption>[역 SDE를 풀면 score-based generative model이 생성됩니다. 
데이터를 간단한 노이즈 분포로 변환하는 데 SDE를 사용할 수 있습니다. 
각 중간 time step에서도 분포의 score를 알면 노이즈에서 샘플을 생성하도록 역변환할 수 있습니다.]</figcaption>
</center>

### Estimating the reverse SDE with score-based models and score matching

### How to solve the reverse SDE

### Probability flow ODE

### Controllable generation for inverse problem solving

## 7.Connection to diffusion models and others

## 8.Concluding remarks