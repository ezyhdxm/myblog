---
layout: post
title:  "统计决策"
comments: true
date:   2020-06-07 22:40:00 +0800
tags: random
lang: zh
---

<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

---

### 1. 先验分布

我们首先回顾一些概率分布的密度函数。

|                     分布                     |                           密度函数                           |             期望              |                           方差                            |
| :------------------------------------------: | :----------------------------------------------------------: | :---------------------------: | :-------------------------------------------------------: |
|            $$Ga(\alpha, \lambda)$$             | $$p(x\mid\alpha, \lambda)=\frac{\lambda^{\alpha}}{\Gamma(\alpha)} x^{\alpha-1} e^{-\lambda x}, x>0$$ |   $$\frac{\alpha}{\lambda}$$    |                $$\frac{\alpha}{\lambda^2}$$                 |
|            $$IGa(\alpha, \lambda)$$            | $$p(x \mid \alpha, \lambda)=\frac{\lambda^{\alpha}}{\Gamma(\alpha)} (\frac{1}{x})^{\alpha+1} e^{-\frac{\lambda}{x}} , x>0$$ |  $$\frac{\lambda}{\alpha-1}$$   |        $$\frac{\lambda^2}{(\alpha-1)^2(\alpha-2)}$$         |
|             $$Be(\alpha, \beta)$$              | $$p(x\mid \alpha,\beta) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{\mathrm{B}(\alpha, \beta)},\mathrm{B}(\alpha, \beta)=\frac{\Gamma(\alpha) \Gamma(\beta)}{\Gamma(\alpha+\beta)}$$ | $$\frac{\alpha}{\alpha+\beta}$$ | $$\frac{\alpha \beta}{(\alpha+\beta)^{2}(\alpha+\beta+1)}$$ |
| $$\chi_{k}^{2} = Ga(\frac{k}{2},\frac{1}{2})$$ | $$\frac{1}{2^{\frac{k}{2}} \Gamma\left(\frac{k}{2}\right)} x^{\frac{k}{2}-1} e^{-\frac{z}{2}}$$ |              $$k$$              |                           $$2k$$                            |
|                  $$t_{\nu}$$                   | $$\frac{\Gamma((\nu+1) / 2)}{\sqrt{\nu \pi} \Gamma(\nu / 2)\left(1+x^{2} / \nu\right)^{(\nu+1) / 2}}$$ |              $$0$$              |                    $$\frac{\nu}{\nu-2}$$                    |

* **例 1.1** 设$$x_1, \dots, x_n$$是来自正态分布$$N\left(\theta, \sigma^{2}\right)$$ 的一个样本观测值。在$$\sigma^2$$已知的情形下，此样本的似然函数为
  $$
  p(\boldsymbol{x} \mid \theta)=\left(-\frac{1}{\sqrt{2 \pi} \sigma}\right)^{n} \exp \left\{-\frac{1}{2 \sigma^{2}} \sum_{i=1}^{n}\left(x_{i}-\theta\right)^{2}\right\}
  $$
  选取$$\pi(\theta)=\frac{1}{\sqrt{2 \pi} \tau} \exp \left\{-\frac{(\theta-\mu)^{2}}{2 \tau^{2}}\right\}$$作为$$\theta$$的先验分布。设$$a = \frac{1}{\tau^2}$$， $$b=\frac{n}{\sigma^{2}}$$，则关于$$\theta$$的后验分布是一个正态分布，均值$$\mu_1 = \frac{a \mu+b \bar{x}}{a+b}$$，方差$$\tau_1^2=\frac{1}{a+b}$$。

  如果$$\theta$$已知，那么我们可以取$$IGA(\alpha, \lambda)$$作为$$\sigma^2$$的先验分布，它的后验为$$I G a\left(\alpha+\frac{n}{2}, \lambda+\frac{1}{2} \sum_{i=1}^{n}\left(x_{i}-\theta\right)^{2}\right)$$。

---

在选好先验分布以后，如果有一些对于超参数的历史数据，那么先验分布中超参数的确定可以利用先验矩、先验分位数等等来进行。此外，我们也可以利用边缘分布来确定先验密度。$$ML-II$$方法中的超参数由$$m(x \mid \hat{\lambda})=\sup _{\lambda \in \mathbb{\Lambda}} \prod_{i=1}^n m\left(x_{i} \mid \lambda\right)$$给出。另一种利用边缘分布的方式是矩方法。

矩方法首先计算出分布$$p(x\mid\theta)$$的期望$$\mu(\theta)$$与方差$$\sigma^2(\theta)$$。利用它们可以方便地得到边缘分布$$m(x\mid\lambda)$$的期望$$\mu_m(\lambda)$$与$$\sigma_m^2(\lambda)$$，因为有关系$$\mu_m(\lambda) = E\left[\mu(\theta)\mid\lambda\right]$$与$$\sigma_{m}^{2}(\lambda)=E\left[\sigma^{2}(\theta)\mid\lambda\right]+E\left[\left(\mu(\theta)-\mu_{m}(\lambda)\right)^{2}\mid\lambda\right]$$。从样本$$\boldsymbol{x}$$中可以计算得到样本均值$$\hat{\mu}_{m}$$与方差$$\hat\sigma_m^2$$，列出方程就可求解$$\lambda$$。

现在的问题是，对于感兴趣的参数，我们要选取什么样的先验分布族？我们来看看在没有先验信息时应该怎么办。

* **贝叶斯假设**

  对没有先验信息的一种理解是$$\theta$$在取值范围上服从均匀分布，即
  $$
  \pi(\theta)=\left\{\begin{array}{ll}
  c, & \theta \in \Theta \\
  0, & \theta \notin \Theta
  \end{array}\right.
  $$
  这个理解称为贝叶斯假设。有时这不能给出一个正常的密度函数，因此我们引入广义先验分布的概念，即

  * 对于总体$$X \sim f(x \mid \theta), \theta \in \Theta$$，若参数的先验分布$$\pi(\theta)$$满足

    1. $$\pi(\theta) \geqslant 0$$，且$$\int_{\mathbf{\theta}} \pi(\theta) d \theta=\infty$$；
    2. 由此决定的后验密度是正常密度函数；

    则称$$\pi(\theta)$$为广义先验分布。

  贝叶斯假设还有一个问题，那就是它不满足变换下的不变性。比如设一个参数$$\theta$$的先验是无信息先验，那么通过一个可逆变换得到的$$\eta = f(\theta)$$理应也有无信息先验。但是当$$\theta$$的先验取为常值时，$$\eta$$的先验为$$\mid\frac{d\theta}{d\eta}\mid\pi(f^{-1}(\eta))$$，这不一定是常值。这就意味着贝叶斯假设并不可以随意使用。在感兴趣参数是位置参数时，贝叶斯假设是成立的。

* **尺度参数的无信息先验**

  假设总体$$X$$的密度函数具有形式$$\frac{1}{\sigma}p\left(\frac{x}{\sigma}\right)$$，那么就称$$\sigma$$为尺度参数。它的无信息先验可取为$$\pi(\sigma)=\sigma^{-1}, \sigma>0$$。

* **Fisher信息阵确定无信息先验**

  Jeffreys提出了一个更加一般的方法。密度函数$$p(x \mid \boldsymbol\theta)$$中的$$p$$维参数$$\boldsymbol{\theta}$$的无信息先验可按如下步骤求解

  1. 写出样本对数似然函数
     $$
     l(\boldsymbol{\theta} \mid \boldsymbol{x})=\ln \left[\prod_{i=1}^{n} p\left(x_{i} \mid \boldsymbol{\theta}\right)\right]=\sum_{i=1}^{n} \ln p\left(x_{i} \mid \boldsymbol{\theta}\right)
     $$
     
2. 求样本信息阵
     $$
     I(\boldsymbol\theta)=E\left(-\frac{\partial^2 l}{\partial\theta_{i} \partial \theta_{j}}\mid\boldsymbol\theta\right)
     $$
     
  3. $$\boldsymbol{\theta}$$的无信息先验密度为
$$
     \pi(\boldsymbol\theta)=[\operatorname{det} I(\boldsymbol\theta)]^{1 / 2}
$$

  有时不能很确定超参数的选取时，可以对超参数再加一层先验，这个先验往往由无信息先验给出。我们有时要对多个有关联的参数$$\theta_1,\dots,\theta_j$$进行估计，这个时候可以将它们看作来自某个共同分布$$\pi(\theta \mid \phi)$$。

  

### 2. 贝叶斯推断

* **参数估计**

使得后验密度$$\pi(\theta\mid\boldsymbol{x})$$达到最大值的$$\hat\theta_{MD}$$称为最大后验估计，后验分布的期望$$\hat\theta_E$$称为$$\theta$$的后验期望估计。当估计取为$$\hat\theta_E$$时，可以使后验均方差达到最小。因为有分解式
$$
MSE(\hat\theta\mid\boldsymbol{x}) = Var(\hat\theta\mid\boldsymbol{x}) + (\hat\theta - \hat\theta_E)^2
$$

* **可信区间**

当我们获得了参数$$\theta$$的后验分布$$\pi(\theta\mid\boldsymbol{x})$$后，可立即得到$$\theta$$落在某个区间$$[a,b]$$中的后验概率，从而我们可以得到对应概率$$1-\alpha$$的可信区间。

* **假设检验**

获得后验分布后，可以计算两个假设的后验概率$$\alpha_0$$与$$\alpha_1$$，通过比较后验概率比$$\alpha_0/\alpha_1$$，可拒绝或接受原假设。

* **预测**

在得到后验分布$$\pi(\theta\mid\mathbf{x})$$后，若要对总体$$p(z\mid\theta)$$未来观测值做预测，则有
$$
m(z\mid\boldsymbol{x}) = \int_{\theta} p(z \mid \theta) \pi(\theta \mid \boldsymbol{x}) d \theta
$$
在没有数据的情况下，则可直接使用先验分布计算得到$$m(z)$$来进行预测。



### 3. 收益与损失

决策问题有三个要素，即状态集$$\Theta=\{\theta\}$$、行动集$$\mathscr{A}=\{a\}$$以及收益函数$$Q(\theta, a)$$。

* **决策准则**

  我们称一个行动$$a_1$$是容许的，若不存在满足下面两个条件的行动$$a_2$$

  1. 对所有$$\theta \in \Theta$$，有$$Q\left(\theta, a_{2}\right) \geqslant Q\left(\theta, a_{1}\right)$$；
  2. 至少有一个$$\theta$$，可使上面不等号严格成立。

  若存在这样的$$a_2$$，则称$$a_1$$是非容许的。若两个行动$$a_1$$与$$a_2$$的收益函数在$$\Theta$$上处处相等，则称$$a_1$$与$$a_2$$是等价的。

  

损失函数定义为$$L(\theta, a)=\max _{a \in \mathscr{A}} Q(\theta, a)-Q(\theta, a)$$，或者对于支付函数$$W(\theta, a)$$而言，$$L(\theta, a)=W(\theta, a)-\min _{a \in \mathscr{A}} W(\theta, a)$$。当$$\theta$$与$$a$$都是实数时，我们可以认为当$$\theta$$与$$a$$相差越大时损失越大，比如将$$a$$理解为对$$\theta$$的估计。那么损失函数就有形式$$L(\theta, a)=\lambda(\theta) g(\mid a-\theta\mid)$$。常用的损失函数可参考下表。

| 损失函数               | 表达式                                                       |
| ---------------------- | ------------------------------------------------------------ |
| 平方损失               | $$(a-\theta)^{2}$$                                             |
| 加权平方损失           | $$\lambda(\theta)(a-\theta)^{2}$$                              |
| 线性损失               | $$\left\{\begin{array}{ll}k_{0}(\theta-a), & a \leqslant \theta \\ k_{1}(a-\theta), & a>\theta\end{array}\right.$$ |
| 加权线性损失           | $$\left\{\begin{array}{ll}k_{0}(\theta)(\theta-a), & a \leqslant \theta \\ k_{1}(\theta)(a-\theta), & a>\theta\end{array}\right.$$ |
| 0-1损失                | $$\left\{\begin{array}{l}0,\mid a-\theta\mid \leqslant \varepsilon \\ 1,\mid a-\theta\mid>\varepsilon\end{array}\right.$$ |
| 多元二次损失           | $$(a-\theta)^{\prime} A(a-\theta)$$                            |
| 二行动线性决策问题损失 | $$\left\{\begin{array}{ll}\mathbf{b}_{1}+m_{1} \theta, & a=a_{1} \\ \mathbf{b}_{2}+m_{2} \theta, & a=a_{2}\end{array}\right.$$ |

### 4. 决策

在给定的贝叶斯决策问题中，称从样本空间到行动集上的一个映射$$\delta(\boldsymbol{x})$$为该问题的一个决策函数。所有决策函数组成决策函数类，用$$\mathscr{D} = \{\delta(\boldsymbol{x})\}$$表示。我们将损失函数$$L(\theta, \delta(\boldsymbol{x}))$$对样本分布$$p_{\theta}(\boldsymbol{x})$$的期望$$R(\theta, \delta)=E_{\theta}[L(\theta, \delta(\boldsymbol{x})]$$称为$$\delta(\boldsymbol{x})$$的风险函数。对决策函数以及风险函数同样也有容许性的概念。

挑选决策函数的一个标准是MiniMax准则。称$$M^{\star}=\underset{\delta \in \mathscr{D}}{\operatorname{Min}} \underset{\theta \in \boldsymbol{\theta}}{\operatorname{Max}} R(\theta, \delta)=\operatorname{Min}_{\delta\in \mathscr{D}} \operatorname{Max}_{\theta \in \boldsymbol{\theta}} E_\theta[L(\theta, \delta(\boldsymbol{X}))]$$为最小最大风险。若有$$\delta^{*}$$使得$$\underset{\theta}{\operatorname{Max}} R\left(\theta, \delta^{*}\right)=\mathrm{M}^{\star}$$，则称它为最小最大准则下的最优决策函数，或最小最大决策函数。我们有一些结论。

* **定理 4.1** 若$$\delta_{0}(\boldsymbol{x})$$是参数$$\theta$$的唯一MiniMax估计，则$$\delta_{0}(\boldsymbol{x})$$也是$$\theta$$的容许估计。
* **定理 4.2** 若$$\delta_{0}(\boldsymbol{x})$$是参数$$\theta$$的容许估计，且在参数空间$$\Theta$$上有常数风险，则$$\delta_{0}(\boldsymbol{x})$$也是$$\theta$$的MiniMax估计。

当然，决策函数不一定要是确定的，它也可能是一个随机变量，给出了采用某个行动的一个分布。

风险函数$$R(\theta,\delta)$$对先验分布$$\pi(\theta)$$的期望称为$$\delta(\boldsymbol{x})$$的贝叶斯风险，记作$$R_{\pi}(\delta)$$。若有$$\delta^{*}$$使得$$R_{\pi}\left(\delta^{*}\right)=\operatorname{Min}_{\delta \in \mathscr{D}} R_{\pi}(\delta)$$，则称它为贝叶斯风险准则下的最优决策函数。

损失函数对后验分布$$\pi(\theta\mid\boldsymbol{x})$$的期望称为后验风险，记为$$R_{\pi}(\delta\mid\boldsymbol{x})=E_{\theta}[L(\theta, \delta(\boldsymbol{x}))\mid\boldsymbol{x}]$$。假如在$$\mathscr{D}$$中存在最小化后验风险的决策函数，则称它是后验风险准则下的最优决策函数。它在一定条件下与贝叶斯风险有关系$$R_{\pi}(\delta) = \int R_{\pi}(\delta \mid \boldsymbol{x}) m(\boldsymbol{x}) d \boldsymbol{x}$$。

* **定理 4.3** 若贝叶斯风险满足$$\operatorname{Min}_{\delta \in \mathscr{D}} \mathbf{R}_{\pi}(\delta)<\infty$$，则贝叶斯决策函数与后验型决策函数是等价的。若损失函数$$L(\theta, a)$$是$$a$$的严格凸函数，则贝叶斯解几乎处处唯一。

我们接下来给出一些常用损失函数下的贝叶斯估计，以下都假设$$x\sim p(x\mid\theta),\;\theta\sim\pi(\theta)$$。

| 损失函数                                                     | 贝叶斯估计                                                   |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| $$(\delta-\theta)^{2}$$                                        | $$E[\theta\mid \boldsymbol{x}]$$                               |
| $$\lambda(\theta)(\delta-\theta)^{2}$$                         | $$\frac{E[\lambda(\theta) \theta \mid x]}{E[\lambda(\theta) \mid x]}$$ |
| $$\lambda(\theta)(\delta-g(\theta))^{2}$$                      | $$\frac{E[\lambda(\theta) g(\theta) \mid x]}{E[\lambda(\theta) \mid x]}$$ |
| $$\left\{\begin{array}{l}k_{0}(\theta-a), a \leq \theta \\ k_{1}(a-\theta), a>\theta\end{array}, k_{0}, k_{1}>0\right.$$ | 后验$$\frac{k_{0}}{k_{0}+k_{1}}$$分位数                        |
| $$(\vec{\delta}-\vec{\theta})^{T} Q(\vec{\delta}-\vec{\theta})$$，$$Q$$正定 | $$E[\vec{\theta} \mid x]$$                                     |
| $$\mid a-\theta\mid$$                                          | 后验中位数                                                   |
| $$\left\{\begin{array}{l}0,\mid a-\theta\mid \leq \varepsilon \\ 1,\mid a-\theta\mid>\varepsilon\end{array}\right.$$ | 后验众数                                                     |

最后是一些贝叶斯解的性质。

* **定理 4.4** 若$$\delta_{\pi}(x)$$为统计问题关于损失函数$$L(\theta, \delta)$$和先验分布$$\pi(\theta)$$在决策函数类$$\mathscr{D}$$上的唯一贝叶斯解，则它是容许的。
* **定理 4.5** 若$$\delta_{0}(x)$$为统计问题关于损失函数$$L(\theta, \delta)$$和先验分布$$\pi(\theta)$$在决策函数类$$\mathscr{D}$$上的贝叶斯解，$$\pi(\theta)$$在$$\Theta$$上处处为正，风险函数$$R(\theta, \delta)$$对任意$$\delta$$都是$$\theta$$的连续函数，且$$\delta_0(x)$$的贝叶斯风险有限，则$$\delta_{0}(x)$$是容许的。

* **定理 4.6** 对$$\theta$$的任意正常先验分布$$\pi(\theta)$$，记$$\delta_{\pi}(x)$$为相应的贝叶斯解，则有$$R_{\pi}\left(\delta_{\pi}\right) \leq M^{*}=\min _{\delta \in \mathcal{D}} \max _{\theta \in \Theta} R(\theta, \delta)$$；若存在正常先验分布$$\pi(\theta)$$，使得$$R_{\pi}\left(\delta_{\pi}\right)=\max _{\theta \in \Theta} R\left(\theta, \delta_{\pi}\right)$$，或者$$\delta_{\pi}(x)$$的风险函数是常数，则$$\delta_{\pi}(x)$$也是MiniMax解。
* **定理 4.7** 若有一正常先验分布列$$\left\{\pi_{k}(\theta), k=1,2, \cdots\right\}$$，其相应的贝叶斯解及贝叶斯风险为$$\left\{\delta_{\pi_{k}}(x), k=1,2, \cdots\right\}$$与$$\left\{r_{k}=R_{\pi_{k}}\left(\delta_{\pi_{k}}\right), k=1,2,\dots\right\}$$，且$$\lim _{k \rightarrow \infty} r_{k}=r$$。则若存在决策函数$$\delta^{*}(x)$$，使得$$\max _{\theta \in \Theta} R\left(\theta, \delta^{*}\right) \leq r$$，则$$\delta^{*}(x)$$为MiniMax解；若$$\delta^{*}(x)$$的风险函数为常数，且等于$$r$$，则$$\delta^{*}(x)$$为MiniMax解。

