---
title: "The NTK Limit, Kernel Ridge Regression, Benign Overfitting, and Mean Field"
subtitle: "ICTP Summer School on Machine Learning — Lecture 6 (Day 8)"
short_title: NTK, KRR, Benign Overfitting, Mean Field
authors:
  - name: Andrea Montanari, transcribed by Max Hirsch with the Claude LLM
subject: Lecture Notes
bibliography: references.bib
---

These notes cover the Day 8 lecture by Andrea Montanari. We begin by making the NTK theorem concrete: an explicit initialization scheme, the kernel trick giving a finite-width KRR predictor, and the Zhong 2023 theorem quantifying how large $m$ must be for the finite-width predictor to match the infinite-width one. We then state the GMMM theorem (Ghorbani–Mei–Misiakiewicz–Montanari 2021) characterizing the exact risk of the infinite-width KRR estimator — revealing a **staircase** structure — and explain the **benign overfitting** mechanism. The second half of the lecture introduces the **mean field limit** of two-layer networks as an alternative to lazy training that enables feature learning, described in three equivalent ways: McKean–Vlasov ODE, Fokker–Planck PDE, and JKO Wasserstein gradient flow.

## Explicit NTK Construction and the Finite-Width KRR Predictor

### A Concrete Initialization Scheme

The abstract NTK theorem (Lecture 5) applies to any network satisfying the Lipschitz condition. Here we specialize to a concrete symmetric initialization that gives a well-conditioned NTK. Consider

$$
f(x;\theta) = \frac{\alpha}{\sqrt{m}}\sum_{k=1}^m b_k\,\sigma(w_k^\top x), \qquad b_1 = \cdots = b_{m/2} = +1,\quad b_{m/2+1} = \cdots = b_m = -1 \;\text{(fixed)},
$$

with $w_1,\ldots,w_{m/2} \sim \mathrm{Unif}(S^{d-1})$ and $w_{m/2+k} = w_k$. The outer weights $b_k$ are fixed; only the $w_k$ are trained. This ensures $f(\cdot;\theta_0) = 0$ at initialization and $\sigma_\ell \neq 0$ for all $\ell$.

The feature map is $\phi(x) = \nabla_\theta f(x;\theta_0) \in \mathbb{R}^p$ ($p = md$), and the **empirical NTK** is

$$
K_m(x_1,x_2) = \frac{1}{m}\sum_{k=1}^m \frac{\langle x_1,x_2\rangle}{d}\,\sigma'(w_k^\top x_1)\,\sigma'(w_k^\top x_2).
$$

### The Linearized Model and Kernel Trick

The linearized model is $f_\mathrm{lin}(x;\theta) = \langle\phi(x), \theta - \theta_0\rangle$, with feature matrix $\Phi \in \mathbb{R}^{n\times p}$. The ridge-regularized solution is

$$
\hat b_\lambda = (\Phi^\top\Phi + \lambda I_p)^{-1}\Phi^\top y.
$$

By the **kernel trick** ($p \gg n$ in the overparameterized regime):

$$
\hat b_\lambda = \Phi^\top(\underbrace{\Phi\Phi^\top}_{K_{m,n}} + \lambda I_n)^{-1}y, \qquad \hat f_\lambda(x) = K_m(x,\mathbb{X})(\lambda I_n + K_{m,n})^{-1}y.
$$

This is the **finite-width KRR predictor** $\hat f_{\mathrm{lin},m,\lambda}$.

### Infinite-Width Limit

As $m\to\infty$, $K_m(x_1,x_2) \to K(x_1,x_2) = h_d(\langle x_1,x_2\rangle/d)$ where

$$
h_d(q) = q\cdot\mathbb{E}[\sigma'(G_1)\sigma'(G_2)], \qquad (G_1,G_2) \sim \mathcal{N}\!\left(0,\binom{1\;q}{q\;1}\right),
$$

and $h_d \xrightarrow{d\to\infty} h$. The **infinite-width NTK predictor** is KRR with kernel $K$:

$$
\hat f_{\infty,\lambda}(x) = K(x,\mathbb{X})(\lambda I_n + K(\mathbb{X},\mathbb{X}))^{-1}y.
$$

### Parenthesis: RKHS

For a kernel $K(x_1,x_2) = \sum_\ell \lambda_\ell^2\psi_\ell(x_1)\psi_\ell(x_2)$, the RKHS norm is $\|f\|_K^2 = \sum_\ell \lambda_\ell^{-2}\langle\psi_\ell,f\rangle^2$, and KRR solves

$$
\hat f_\lambda = \operatorname*{arg\,min}_f\left\{\frac{1}{n}\sum_i(y_i-f(x_i))^2 + \lambda\|f\|_K^2\right\}.
$$

### Approximation Theorem: How Large Must $m$ Be?

:::{prf:theorem} Finite-Width NTK Approximation [@oyamle2021soltanolkotabi; @zhong2023memory]
:label: thm-finite-width

Assume $n \geq d$, $md \geq C_n\log n$, $\alpha \geq C\sqrt{n^2/(md)}$. Then with high probability:

(i) **Training convergence:** $\hat R_n(\theta_t) \to 0$ exponentially fast.

(ii) **Finite-to-linearized approximation** (Oyamle–Soltanokotabi 2021): gradient flow tracks the linearized KRR model,

$$
\|f(\theta_t) - f_\mathrm{lin}(\theta_t)\|_{L^2} \leq \frac{C}{\alpha^2}\sqrt{\frac{n^5}{md^4}} = \frac{\|a\|_1}{m}\sqrt{\frac{d}{n}}.
$$

(iii) **Finite-to-infinite-width approximation** (Zhong 2023): for $d^{\ell+\delta} \leq n \leq d^{\ell+1-\delta}$ and all $\lambda \in [0,\lambda_*]$,

$$
R(\hat f_{m,n,\lambda}) = R(\hat f_{\infty,n,\lambda}) + O\!\left(\sqrt{\frac{n(\log n)^c}{md}}\right).
$$
:::

As long as $md \gg n(\log n)^c$, the finite-width predictor is statistically equivalent to the infinite-width one. All the GMMM theory below therefore applies to the actual finite-width network.

### Proof Sketch: Matrix Bernstein

The key step is $K_{m,n} \approx K_n$. Write $K_{m,n} = \sum_{\ell=1}^m Z_\ell$ (rank-$d$ i.i.d. matrices). Direct concentration fails since $Z_\ell$ are not centered. Instead normalize:

$$
K_n^{-1/2}K_{m,n}K_n^{-1/2} - I = \sum_{\ell=1}^m \hat Z_\ell, \qquad \hat Z_\ell = K_n^{-1/2}Z_\ell K_n^{-1/2} - \tfrac{1}{m}I_n.
$$

The $\hat Z_\ell$ are mean-zero with bounded operator norm. Apply the **matrix Bernstein inequality** to get the desired concentration.

---

## The GMMM Theorem: Staircase Risk of KRR

### Setting

Data $x_i = \sqrt{d}\,z_i$, $z_i \overset{\text{iid}}{\sim} \mathrm{Unif}(S^{d-1})$, labels $y_i = f_*(x_i)+\varepsilon_i$, $\mathrm{Var}(\varepsilon_i) = \sigma^2$. Kernel $K(x_1,x_2) = h(\langle x_1,x_2\rangle/d)$ with Gegenbauer expansion $h(u) = \sum_\ell b_\ell Q_\ell(u)$.

:::{prf:theorem} Ghorbani–Mei–Misiakiewicz–Montanari [@ghorbani2021linearized]
:label: thm-gmmm

Assume $\sigma$ is $\sigma$-generic ($b_\ell > 0\;\forall\ell$) and $\mathbb{E}\sigma^2(G) < \infty$. For $d^\ell \ll n \ll d^{\ell+1}$ and $\lambda \in [0,\lambda_*]$:

$$
R(\hat f_{\infty,\lambda}) - R_\mathrm{Bayes} = \|P_{>\ell}f_*\|_{L^2}^2 + o_\mathbb{P}(1)\cdot\|f_*\|_{L^2}.
$$
:::

### The Staircase

The KRR estimator learns all polynomial components up to degree $\ell$ when $n \asymp d^\ell$, and nothing beyond. Test risk drops at each threshold $n \approx d^j$ by $\|P_j f_*\|^2$, giving a staircase in $\log n$.

:::{admonition} Remark: Curse of Dimensionality for NTK
:class: warning
To learn a degree-$\ell$ target, the NTK requires $n \gg d^\ell$ samples. Feature learning (mean field) requires only $n \gg d$.
:::

### Mechanism: Kernel Matrix Eigenstructure

$$
K_n[i,j] = \sum_\ell \frac{b_\ell}{d^\ell}Y_\ell(z_i)^\top Y_\ell(z_j) \approx \underbrace{\sum_{\ell=0}^L \frac{b_\ell}{d^\ell}\hat Y_{\ell,n}\hat Y_{\ell,n}^\top}_{K_{\leq L}} + \lambda_0 I.
$$

When $D(L) \ll n \ll D(L+1)$, the low-rank term $K_{\leq L}$ dominates; $K_{>L} \approx \lambda_0 I$ acts as self-induced ridge. The KRR predictor reduces to regression on degree-$\leq L$ features.

---

## Benign Overfitting

Interpolation $\hat f(x_i) = y_i$ decomposes as $\hat f = \underbrace{P_{\leq\ell}f_*}_\text{smooth} + \hat f_\text{spiky}$. Benign overfitting requires:

1. $\|\hat f_\text{spiky}\|_{L^2} = o(1)$ — negligible test-error contribution,
2. $\hat f_\text{spiky}(x_i) \approx \varepsilon_i$ — interpolates the noise.

Both hold in high $d$: high-degree features have pointwise values $\sim\sqrt{D(\ell)} \gg 1$ but $L^2$ norm $\sim 1$, and test points are nearly orthogonal to training points (concentration of measure). At fixed $d$ this fails: nearby test points "see" the spiky component.

---

## Lazy Training: Single-Index Model

Data: $x_i \sim \mathrm{Unif}(\sqrt{d}\,S^{d-1})$, $f_*(x) = \varphi(w_*^\top x)$, $w_* \in S^{d-1}$.

The $\ell$-th Hermite coefficient $\varphi_\ell = \int \mathrm{He}_\ell(u)\varphi(u)\,\gamma(du)$ (independent of $d$ as $d\to\infty$) controls the staircase: $\|P_\ell f_*\|_{L^2}^2 = \varphi_\ell^2$. By the GMMM theorem, the NTK requires $n \gg d^\ell$ if $\varphi$ is not a degree-$\ell$ polynomial.

---

## Feature Learning Beats Lazy Training

### Proposition (Single Neuron, Feature Learning)

Assume $\varphi$ is strictly increasing with bounded derivative. If $n \geq Cd\log d$, then with high probability gradient descent on the single-neuron risk

$$
\hat R_n(w) = \frac{1}{2n}\sum_i(y_i - \sigma(w^\top x_i))^2, \qquad \varphi = \sigma,
$$

converges to $\hat w$ with $\|\hat w - w_*\| \leq C\sqrt{d\log d/n}$ and $R(\hat w) = R_\mathrm{Bayes} + O(d\log d/n)$.

### Two-Step Estimator ($n \gg d$ Suffices)

**Step 1.** Form $\tilde w = \frac{1}{n}\sum_i y_i x_i$. By Stein's lemma ($\mathbb{E}[g\,h(g)] = \mathbb{E}[h'(g)]$):

$$
\mathbb{E}[\tilde w] = \mathbb{E}[\varphi(w_*^\top x)\,x] = \mathbb{E}[\varphi'(w_*^\top x)]\,w_* = c\,w_*.
$$

Variance $\mathbb{E}\|\tilde w - \mathbb{E}\tilde w\|^2 = O(d/n)$, so $n \gg d \Rightarrow \|\hat w - w_*\| = O(\sqrt{d/n}) \to 0$.

**Step 2.** Project: $t_i = \langle\hat w, x_i\rangle$, fit $y_i \sim \varphi(t_i)$ by 1D regression. Achieves Bayes risk.

The two-step estimator requires only $n \gg d$ vs. $n \gg d^\ell$ for NTK — exponentially fewer samples.

---

## Mean Field Limit of Two-Layer Networks

### Lazy vs. Mean Field

Plotting test error $R(\hat f) - R_\mathrm{Bayes}$ vs. number of neurons $m$:

- **Lazy training** ($1/m$ init): error stays near $\|P_{>1}\varphi\|^2$ for all $m$ — NTK is stuck.
- **Mean field** ($1/m$ init, parameters allowed to move): error $\to 0$ as $m\to\infty$ at rate $O(1/m)$.

### Setup

$$
f(x) = \frac{1}{m}\sum_{k=1}^m a_k\,\bar\sigma(\langle w_k,x\rangle) = \frac{1}{m}\sum_{k=1}^m \bar\sigma(x;\theta_k), \qquad \theta_k = (a_k,w_k) \in \mathbb{R}^{d+1}.
$$

SGD with square loss $\ell(y,f(x)) = \frac{1}{2}(y-f(x))^2$, parameters $(\theta_k)_{k\leq m}$ initialized i.i.d. from $\rho_0$:

$$
\theta^{k+1} = \theta^k - \gamma m\,\nabla_\theta\ell(y_{I(k)}, f(x_{I(k)};\theta^k)).
$$

Two settings: *finite sample* ($I(k)\sim\mathrm{Unif}(\{1,\ldots,n\})$) or *online SGD* (fresh samples).

### Mean Field Limit: $m\to\infty$, $\gamma\to 0$

Population gradient flow: $\dot\Theta_t = -m\nabla_\theta R(\theta_t)$. The dynamics are permutation-invariant in neuron indices, motivating the **empirical measure** $\hat\rho_t^{(m)} = \frac{1}{m}\sum_k\delta_{\theta_k(t)}$ with $f_t(x) = \int\sigma(x;\theta)\hat\rho_t^{(m)}(d\theta)$.

Since $(\theta_k)$ are i.i.d. $\sim\rho_0$ initially, $\hat\rho_0^{(m)}\Rightarrow\rho_0$. The **propagation of chaos** hypothesis ("hope"):

$$
\hat\rho_t^{(m)} \Rightarrow \rho_t \quad \text{as } m\to\infty.
$$

### Three Equivalent Descriptions

Define the one- and two-body potentials:

$$
V(\theta) = -\mathbb{E}\{\sigma(x;\theta)y\}, \qquad U(\theta_1,\theta_2) = \mathbb{E}[\sigma(x;\theta_1)\,\sigma(x;\theta_2)].
$$

**(1) McKean–Vlasov ODE on $\mathbb{R}^{d+1}$:**

$$
\dot\Theta_t = -\nabla_\theta\!\left[V(\Theta_t) + \int U(\Theta_t,\theta)\,\rho_t(d\theta)\right], \qquad \rho_t = \mathrm{Law}(\Theta_t), \quad \rho_0 \text{ given.}
$$

$\rho_t$ is the distribution of neuron parameters at time $t$; $\Theta_t$ is the trajectory of a typical neuron.

The gradient takes the explicit form

$$
\dot\Theta_i = -\nabla_{\theta_i}\!\left[V(\theta_i) + \frac{1}{m}\sum_j U(\theta_i,\theta_j)\right] \xrightarrow{m\to\infty} -\nabla_\theta\!\left[V(\Theta_t) + \int U(\Theta_t,\theta)\rho_t(d\theta)\right].
$$

**(2) Fokker–Planck PDE (Continuity Equation):**

For test function $h$, $\frac{d}{dt}\mathbb{E}[h(\Theta_t)] = -\mathbb{E}\langle\nabla h(\Theta_t), \nabla_\theta\Psi(\theta;\rho_t)\rangle$ where $\Psi(\theta;\rho_t) = V(\theta) + \int U(\theta,\theta')\rho_t(d\theta')$. Writing in density form:

$$
\boxed{\partial_t\rho_t = \nabla\cdot\bigl[\rho_t\,\nabla\Psi(\theta;\rho_t)\bigr].}
$$

This is **gradient flow of $R(\rho)$ in $W_2$**, where

$$
R(\rho) = \tfrac{1}{2}\mathbb{E}\!\left[\!\left(y - \textstyle\int\sigma(x;\theta)\rho(d\theta)\right)^2\right] = \mathrm{const} + \int V(\theta)\rho(d\theta) + \tfrac{1}{2}\iint U(\theta_1,\theta_2)\rho(d\theta_1)\rho(d\theta_2).
$$

**(3) JKO Scheme / Minimizing Movements (DeGiorgi) on $(\mathcal{P}(\mathbb{R}^{d+1}), W_2)$:**

Gradient flow on Riemannian manifold $(M,d)$ is discretized by the **proximal algorithm**:

$$
x_0^\varepsilon = x_0, \qquad x_{k+1}^\varepsilon = \operatorname*{arg\,min}_{x\in M}\left\{\frac{1}{2\varepsilon}\,d(x,x_k^\varepsilon)^2 + F(x)\right\}.
$$

In the Euclidean case $d(x,y)=\|x-y\|$ this recovers gradient descent: $x_{k+1}\approx x_k - \varepsilon\nabla F(x_k)$. Interpolating by geodesics and taking $\varepsilon\to 0$: if $\limsup_{\varepsilon\to 0, t\in[0,T]} d(x(t),x^\varepsilon(t)) = 0$, then $x$ is the gradient flow for $F$ on $(M,d)$.

Applied to $M = \mathcal{P}(\mathbb{R}^{d+1})$ with $W_2$ and $F = R(\rho)$, this is the **JKO (Jordan–Kinderlehrer–Otto 1996) scheme**, recovers the continuity equation, and connects to the **minimizing movements** framework of DeGiorgi.

:::{admonition} Mean Field vs. NTK
:class: tip

In the **NTK/lazy training** limit, parameters move $O(1/\sqrt{m})$ from initialization, features are **frozen**, and the network is equivalent to a kernel method. Feature learning is impossible.

In the **mean field limit**, each $\theta_k$ evolves by the McKean–Vlasov ODE, moving a finite distance. The distribution $\rho_t$ can concentrate on a submanifold adapted to the data, enabling features to be **learned**. This is what achieves $R(\hat w) = R_\mathrm{Bayes} + O(d\log d/n)$ for the single-index model with only $n \gg d$ samples.
:::

---

## Wasserstein Gradient Flow: Metric Space Foundations

### Recap and the $W_2$ Distance

We recap the three descriptions of the mean field limit, now fleshing out the Wasserstein gradient flow structure that underlies description (3). The Wasserstein-2 distance between $\rho_1, \rho_2 \in \mathcal{P}(\mathbb{R}^{d+1})$ is

$$
W_2(\rho_1,\rho_2) = \sqrt{\inf_{\gamma \in \mathcal{C}(\rho_1,\rho_2)} \int \|\theta_1 - \theta_2\|^2\,\gamma(d\theta_1,d\theta_2)},
$$

where $\mathcal{C}(\rho_1,\rho_2)$ is the set of couplings (joint distributions with marginals $\rho_1$ and $\rho_2$). The mean field dynamics are gradient flow of $R(\rho)$ in $(\mathcal{P}(\mathbb{R}^{d+1}), W_2)$.

### Gradient Flow on a Metric Space: Metric Speed and Slope

To define gradient flow on a metric space $(M, d)$ without a linear structure, one works with the **metric speed** and **metric slope** of $F : M \to \mathbb{R}$ [@ambrosio2005gradient]:

$$
|\dot x|(t) = \lim_{\varepsilon \to 0} \frac{d(x(t+\varepsilon), x(t))}{\varepsilon}, \qquad |\nabla F|(x) = \limsup_{d(z,x)\to 0} \frac{[F(z) - F(x)]^+}{d(z,x)}.
$$

A curve $t \mapsto x(t)$ is a **gradient flow** for $F$ on $(M,d)$ if

$$
-\frac{d}{dt}F(x(t)) \geq \frac{1}{2}\bigl\{|\dot x|(t)^2 + |\nabla F|^2(x(t))\bigr\},
$$

with equality if and only if $x$ is the gradient flow. The definition, due to Ambrosio, Gigli, and Savaré [@ambrosio2005gradient], generalizes the Euclidean identity $-\frac{d}{dt}F(x(t)) = \|\dot x(t)\|^2 = \|\nabla F(x(t))\|^2$ and makes sense in any metric space.

### Applying to $(\mathcal{P}, W_2)$

Setting $x = \rho$, $d = W_2$, $F = R$, the metric speed of the mean field trajectory satisfies

$$
|\dot\rho|^2(t) \leq \int \|\nabla\Psi(\theta;\rho_t)\|^2\,\rho_t(d\theta),
$$

since the upper bound on $W_2(\rho_{t+\varepsilon}, \rho_t)$ comes from pushing $\rho_t$ along the flow $\theta(t+\varepsilon) = \theta(t) - \varepsilon\nabla\Psi(\theta(t);\rho_t) + o(\varepsilon)$. The metric slope satisfies

$$
|\nabla R|(\rho_*) = \lim_{\substack{W_2(\rho,\rho_*)\to 0}} \frac{[R(\rho) - R(\rho_*)]^+}{W_2(\rho,\rho_*)} = \left(\int \|\nabla\Psi(\theta;\rho_*)\|^2\,\rho_*(d\theta)\right)^{1/2}.
$$

The verification uses the Taylor expansion of $R$: for $\rho = (I + \varepsilon \xi)_\#\rho_*$ with displacement field $\xi$,

$$
R(\rho) - R(\rho_*) = -\int\langle\nabla\Psi(\theta_0;\rho_*),\,\theta_1-\theta_0\rangle\,\gamma(d\theta_0,d\theta_1) + O(W_2^2) \leq \left(\int\|\nabla\Psi\|^2\rho_*\right)^{1/2} W_2(\rho,\rho_*) + O(W_2^2).
$$

Hence $-\frac{d}{dt}R(\rho_t) = \int\|\nabla\Psi(\theta;\rho_t)\|^2\,\rho_t(d\theta) = \frac{1}{2}(|\dot\rho|^2 + |\nabla R|^2)$, confirming the gradient flow identity.

### Convergence Theorem for SGD to Mean Field

:::{prf:theorem} Propagation of Chaos [@mei2018mean]
:label: thm-poc

Assume $\theta_i(0) \overset{\text{iid}}{\sim} \rho_0$ and online SGD with step size $\varepsilon$. Under:
1. $|a|, |y| \leq C$, $\nabla_\theta\sigma(x,\theta)$ is $O(1)$-subgaussian,
2. $\theta \mapsto V(\theta)$, $(\theta_1,\theta_2)\mapsto U(\theta_1,\theta_2)$ differentiable with Lipschitz gradient,

for all $T \geq 1$, with probability $\geq 1 - 2e^{-z/2}$:

$$
\sup_{t \in [0,T]} |R^{(m)}(\theta(\lfloor t/\varepsilon \rfloor)) - R(\rho_t)| \leq \frac{e^{CT}}{\sqrt{m}}\bigl[\sqrt{\log m} + z\bigr] + e^{CT}\bigl[\sqrt{d\log m} + z\bigr]\sqrt{\varepsilon}.
$$

At time $t = \Theta(1)$, using $n = t/\varepsilon$ samples with $\varepsilon d \ll 1$ and $n \gg d$, the empirical risk of the $m$-neuron network tracks the mean field risk.
:::

The proof uses the **Dobrushin propagation of chaos** technique @dobrushin1979vlasov, coupling three dynamics initialized from the same $\rho_0$:

1. **SGD:** $\theta_i(t+\varepsilon) = \theta_i(t) - \varepsilon m\,\nabla_{\theta_i}\ell(y_{I(t)}, f(x_{I(t)};\theta(t)))$
2. **GD on empirical risk:** $\hat\theta_i(t+\varepsilon) = \hat\theta_i(t) - m\int_t^{t+\varepsilon}\nabla_{\theta_i}R^{(m)}(\hat\theta(s))\,ds$, with $\nabla_{\theta_i}R^{(m)} = \nabla_{\theta_i}\Psi(\hat\theta_i;\hat\rho_t^{(m)})$
3. **Nonlinear McKean–Vlasov:** $\tilde\theta_i(t+\varepsilon) = \tilde\theta_i(t) - m\int_t^{t+\varepsilon}\nabla_{\theta_i}\Psi(\tilde\theta_i(s);\rho_s)\,ds$, with $(\tilde\theta_i(t))_{i\leq m} \overset{\text{iid}}{\sim} \rho_t$

The total error decomposes as $\|f_\mathrm{SGD} - f_\mathrm{MF}\| \leq \|f_\mathrm{SGD} - f_\mathrm{GD}\| + \|f_\mathrm{GD} - f_\mathrm{MF}\|$. Term 1 is bounded by standard stochastic approximation; term 2 is the propagation of chaos error, controlled by the Dobrushin–McKean coupling.

**Symmetry.** For the single-index model $x_i \sim \mathcal{N}(0, I_d)$, $y_i = \varphi(w_*^\top x_i) + \varepsilon_i$: if $\rho_0$ is invariant under rotations around $w_*$ then $\rho_t$ is for all $t$. This reduces the effective dynamics to a 1D ODE in the projection $\langle w, w_*\rangle$, a key simplification.

### Fixed Points and Global Minimizers

:::{prf:lemma} Fixed Point Characterization
:label: lem-fixedpt

$\rho_*$ is a fixed point of the mean field gradient flow if and only if

$$
\mathrm{supp}(\rho_*) \subseteq \{\theta : \nabla\Psi(\theta;\rho_*) = 0\}.
$$

Moreover, $-\frac{d}{dt}R(\rho_t) = \frac{1}{2}(|\dot\rho|^2 + |\nabla R|^2) \geq 0$ with equality iff $\rho_t$ is a fixed point.

**Theorem.** The Fokker–Planck equation is gradient flow in $(\mathcal{P}, W_2)$ for the **free energy** $\mathcal{D}(\rho\|\rho_\mathrm{Boltz})$ (KL divergence from the Boltzmann distribution) $[@jordan1998variational]$.
:::

The lemma is proved as follows. Since $R(\rho) = \frac{1}{2}\mathbb{E}[(y-\int\sigma(x;\theta)\rho(d\theta))^2]$, its convexity follows from:

$$
R(t\rho_1 + (1-t)\rho_2) \leq tR(\rho_1) + (1-t)R(\rho_2),
$$

which holds since $R$ is a quadratic in the linear functional $\rho \mapsto \int\sigma(x;\theta)\rho(d\theta)$. Expanding $R(\rho_s) - R(\rho_*)$ for $\rho_s = s\rho + (1-s)\rho_*$ and differentiating at $s=0$:

$$
R(\rho) - R(\rho_*) = \int\Psi(\theta;\rho_*)(\rho - \rho_*)(d\theta) + O(W_2^2) \geq 0 \;\;\forall\rho \iff \Psi(\theta;\rho_*) \geq \min_\theta \Psi(\theta;\rho_*) \;\;\rho_*\text{-a.e.}
$$

:::{prf:lemma} Existence and Structure of Minimizers
:label: lem-exist

Assume $\theta \mapsto V(\theta)$, $(\theta_1,\theta_2)\mapsto U(\theta_1,\theta_2)$ are continuous, bounded below, and there exist $\varepsilon, C > 0$ such that $R(\rho) \leq \inf_\rho R(\rho) + \varepsilon \Rightarrow \int\|\theta\|^a\rho(d\theta) < C$. Then:

1. $\min_\rho R(\rho)$ is achieved at some $\rho_*$.
2. $\rho_*$ is a minimizer iff $\mathrm{supp}(\rho_*) \subseteq \operatorname*{arg\,min}_\theta\Psi(\theta;\rho_*)$.
:::

Part (1) follows from lower semicontinuity of $R$ under weak convergence (which holds because $\sigma$ is continuous and bounded) and tightness of sublevel sets. Part (2) follows from the first-order optimality condition above.

:::{prf:corollary}
If $\mathrm{supp}(\rho_*) = \{\theta^*\}$ (a single atom) and $\rho_*$ is a fixed point of the gradient flow, then $\rho_*$ is a **global minimizer** of $R$.
:::

This corollary captures the mean field intuition: if the dynamics collapse to a single neuron trajectory $\theta^*$ (which happens, for example, in the single-index model under the symmetry reduction above), then that trajectory achieves the global minimum of $R$.
