---
title: "Feature Learning in Two-Layer Networks: Weak Recovery and the Generative Exponent"
subtitle: "ICTP Summer School on Machine Learning — Lecture 7 (Day 9, Morning)"
short_title: Weak Recovery and Generative Exponent
authors:
  - name: Andrea Montanari, transcribed by Max Hirsch with the Claude LLM
subject: Lecture Notes
bibliography: references.bib
---

These notes cover the second lecture of Day 9 by Andrea Montanari. We ask when a two-layer network trained by (stochastic) gradient descent can learn the single-index model $y = \varphi(w_*^\top x)$ in polynomial time with $n = \tilde O(d)$ samples. The mean field theory (MFT) provides a sharp separation from the NTK regime: while Rademacher complexity remains small along MFT dynamics, it blows up in the NTK regime. The lecture introduces the **generative exponent** $k_*$ of a link function $\varphi$ and proves that: (a) weak recovery requires $n \geq \Omega(d^{k_*/2})$ for any efficient algorithm (lower bound from SQ/LDP hardness), and (b) a computationally efficient algorithm achieving weak recovery with $n = \tilde O(d^{k_*/2})$ is given by a **degree-$k_*$ preprocessing** estimator, as proved in @damian2024computational.

## MFT vs. NTK: Rademacher Complexity

We begin with a structural observation that sharpens the NTK/feature-learning distinction.

**In mean field theory** (MFT), the Rademacher complexity $\mathrm{Rd}_n(\mathcal{F}_t)$ — where $\mathcal{F}_t$ is the function class at time $t$ — stays small ($= O(1)$) on times $t = O(1)$. This is because in MFT the mean field potential $\Psi(\theta;\rho_t)$ has a small number of effective degrees of freedom (essentially 1, by the symmetry reduction for the single-index model), and the Barron norm $\|f_t\|_\sigma$ remains bounded.

**In the NTK regime**, however, as we showed via the GMMM theorem (Lecture 6), $\mathrm{Rd}_n(\mathcal{F}) \geq c\sigma > 0$ when the network class can overfit. The Rademacher bound does not vanish.

This distinction is the precise sense in which MFT enables generalization that NTK cannot: the effective complexity of the hypothesis class stays controlled throughout training.

### Case Study: Single-Index Model

We focus on the single-index model $y_i = \varphi(\langle w_*,x_i\rangle) + \varepsilon_i$ with $x_i \sim \mathcal{N}(0,I_d)$, $\varepsilon_i \sim \mathcal{N}(0,\tau^2)$, and ask: can a two-layer network $f(x;\theta) = \frac{1}{m}\sum_{k=1}^m a_k\sigma(w_k^\top x)$ learn this model efficiently?

Three approaches and their sample complexities:

- **RKHS/NTK methods** (e.g. KRR with the NTK): require $n \geq d^c$ for every constant $c > 0$ by the GMMM theorem (Lecture 6). Computationally efficient but statistically far from optimal.
- **ERM over functions of bounded Barron norm** with $\|f_*\|_\sigma \asymp 1$: requires $n = \tilde O(d)$ samples (from the Rademacher bound, Lecture 4), but is **computationally inefficient** — minimizing the empirical risk over the Barron ball is NP-hard in general.
- **Gradient-based methods on a two-layer network** (the question): can they be efficient?

The answer depends on the link function $\varphi$ via the **generative exponent** defined below. By the results of @damian2024computational, there is a polynomial-time algorithm achieving weak recovery when $n \geq \tilde O(d^{k_*/2})$, and no efficient algorithm can do better.

---

## Weak Recovery and the Generative Exponent

### Definition

:::{prf:definition} Generative Exponent
:label: def-genexp

Let $G \sim \mathcal{N}(0,1)$ and $Y = \varphi(G) + \varepsilon$ with $\varepsilon \sim \mathcal{N}(0,\tau^2)$. The **generative exponent** of $(G,Y)$ is

$$
k_* = \inf\{k \geq 1 : \mathbb{E}[\mathrm{He}_k(G) \mid Y] \not\equiv 0\} = \inf\{k \geq 1 : \mathbb{E}[\mathrm{He}_k(G) \mid Y]^2 > 0\},
$$

where $\mathrm{He}_k$ is the $k$-th probabilist's Hermite polynomial (so $\mathrm{He}_0 = 1$, $\mathrm{He}_1(u)=u$, $\mathrm{He}_2(u)=u^2-1$, etc.).
:::

Equivalently, $k_*$ is the index of the first nonzero Hermite coefficient $\lambda_k = \mathbb{E}[\varphi(G)\mathrm{He}_k(G)]$ of $\varphi$: $k_* = \inf\{k : \lambda_k \neq 0\}$.

Note: $\lambda_1 = \mathbb{E}[\varphi(G)G]$ (the Stein coefficient), so $k_* = 1$ iff $\mathbb{E}[\varphi(G)G] \neq 0$.

**Examples:**
- $\varphi(u) = u$ (linear): $k_* = 1$ (trivial).
- $\varphi(u) = u^2$ (quadratic, **phase retrieval**): $\lambda_1 = \mathbb{E}[G^3] = 0$, $\lambda_2 = \mathbb{E}[G^2(G^2-1)] = 2 \neq 0$, so $k_* = 2$.
- $\varphi(u) = u^3$: $k_* = 1$ since $\lambda_1 = \mathbb{E}[G^4] = 3 \neq 0$. Wait — $\lambda_1 = \mathbb{E}[\varphi(G)\cdot G] = \mathbb{E}[G^4] = 3$, so $k_* = 1$.
- $\varphi(u) = \mathrm{He}_k(u)$: $k_* = k$ (the identity by orthogonality).

### The Weak Recovery Problem

**Weak recovery** means finding $\hat w$ such that $|\langle\hat w, w_*\rangle| \geq \varepsilon > 0$ (non-trivial correlation with $w_*$) with high probability. It is a necessary first step toward full estimation.

**Goal:** construct an efficient estimator achieving weak recovery using $n = \tilde O(d^{k_*/2})$ samples.

**Why $k_*/2$?** As we will see, the natural preprocessing statistics live in $(\mathbb{R}^d)^{\otimes k_*}$ (a tensor of order $k_*$), which has $d^{k_*}$ entries. Estimating a rank-1 tensor from $n$ samples by a spectral method requires $n \gg d^{k_*/2}$ (the threshold for a Wishart-type matrix with $d^{k_*/2} \times d^{k_*/2}$ structure).

---

## Weak Recovery Idea \#1: Score Function (Preprocessing)

### Linear Preprocessing ($k_* = 1$)

When $k_* = 1$, i.e. $\lambda_1 = \mathbb{E}[Y G] \neq 0$, the estimator

$$
\tilde w = \frac{1}{n}\sum_{i=1}^n y_i x_i, \quad \hat w = \frac{\tilde w}{\|\tilde w\|},
$$

achieves weak recovery. Its expectation is $\mathbb{E}\tilde w = \mathbb{E}[yx] = \mathbb{E}[\varphi(w_*^\top x)\,x] = \lambda_1 w_*$ (by Stein's lemma $\mathbb{E}[xh(w^\top x)] = \mathbb{E}[h'(w^\top x)]w$). The variance is $\mathbb{E}\|\tilde w - \lambda_1 w_*\|^2 = \frac{1}{n}\mathbb{E}[\|yx\|^2] \leq \frac{Cd}{n}$, so $\|\hat w - w_*\| = O(\sqrt{d/n})$ and weak recovery holds for $n/d$ bounded away from 0 (@proposition-feature-learning in Lecture 6).

**Failure case:** $k_* \geq 2$, which happens when $\mathbb{E}[YG] = 0$, i.e. when $\varphi$ is an even function or more generally has no odd-degree component in its Hermite expansion. The prototype is **phase retrieval** where $\varphi(u) = u^2$.

### General Preprocessing: Optimizing the Linear Functional

To handle $k_* \geq 2$, one replaces $y$ by a preprocessed label $T(y)$. The estimator becomes

$$
\tilde w = \frac{1}{n}\sum_{i=1}^n T(y_i)\,x_i, \quad \hat w = \frac{\tilde w}{\|\tilde w\|}.
$$

The expectation is $\mathbb{E}[\tilde w] = \mathbb{E}[T(Y)X] = \mathbb{E}[T(Y)\mathbb{E}[G|Y]]\,w_* = \lambda_1(T)\,w_*$ where $\lambda_1(T) = \mathbb{E}[T(Y)G]$. This is non-zero iff $\mathbb{E}[T(Y)G] \neq 0$, i.e. iff $T$ is not orthogonal to $\mathbb{E}[G|Y]$.

**Optimal choice** (the Preprocessing Trick): maximize $\mathbb{E}[T(Y)G]$ subject to $\mathbb{E}[T(Y)^2] = 1$. By Cauchy–Schwarz:

$$
\mathbb{E}[T(Y)G] = \mathbb{E}[T(Y)\mathbb{E}[G|Y]] \leq (\mathbb{E}[T(Y)^2])^{1/2}\big(\mathbb{E}[\mathbb{E}[G|Y]^2]\big)^{1/2} = \lambda_1(T)_\mathrm{opt}^{1/2},
$$

with equality at $T(y) = \mathbb{E}[G|Y=y]$ (normalized).

**Works iff $k_* = 1$**: The optimal preprocessing works iff $\lambda_1 = \mathbb{E}[\mathbb{E}[G|Y]^2] > 0$, i.e. iff $\mathbb{P}(\mathbb{E}[G|Y] = 0) < 1$. This fails precisely when $\mathbb{E}[G|Y] = 0$ a.s., which happens when $\varphi$ is even (e.g. phase retrieval: $\varphi(u) = u^2$ means $Y = G^2 + \varepsilon$, so $\mathbb{E}[G|Y=y] = 0$ since $G$ and $-G$ are exchangeable given $Y$).

---

## Weak Recovery Idea \#2: Second-Moment Matrix

### When $k_* = 2$: Second-Moment Estimator

For $k_* = 2$ (i.e. $\lambda_1 = 0$ but $\lambda_2 = \mathbb{E}[\varphi(G)\mathrm{He}_2(G)] = \mathbb{E}[\varphi(G)(G^2-1)] \neq 0$), consider the **empirical second-moment matrix**:

$$
\hat M = \frac{1}{n}\sum_{i=1}^n y_i\,x_i x_i^\top \in \mathbb{R}^{d\times d}, \quad \hat w = v_1(\hat M) \text{ (leading eigenvector)}.
$$

**Population mean:** Using $y = \varphi(w_*^\top x)$ and $x = g w_* + x^\perp$ with $g = w_*^\top x \sim \mathcal{N}(0,1)$ and $x^\perp \perp w_*$:

$$
M = \mathbb{E}[\hat M] = \mathbb{E}[\varphi(w_*^\top x)\,xx^\top] = \mathbb{E}[\varphi(g)\,xx^\top].
$$

Computing: $\mathbb{E}[\varphi(g) x_j x_k] = \mathbb{E}[\varphi(g)(gw_{*j} + x_j^\perp)(gw_{*k}+x_k^\perp)]$. Cross terms vanish by independence, and $\mathbb{E}[\varphi(g)x_j^\perp x_k^\perp] = \mathbb{E}[\varphi(g)]\delta_{jk}$ (since $x^\perp \perp g$). Therefore:

$$
M = \mathbb{E}[\varphi(G)]\,I + \mathbb{E}[\varphi(G)G^2]\,w_*w_*^\top = \underbrace{\mathbb{E}[\varphi(G)]}_{=c}\,I + \underbrace{\mathbb{E}[\varphi''(G)]}_{=\lambda_2}\,w_*w_*^\top,
$$

where the second step uses $\mathbb{E}[\varphi(G)G^2] = \mathbb{E}[\varphi(G)\mathrm{He}_2(G)] + \mathbb{E}[\varphi(G)] = \lambda_2 + c$, giving $\mathbb{E}\hat M = cI + \lambda_2 w_*w_*^\top$. (Here $\lambda_2 = \mathbb{E}[\varphi''(G)] = \mathbb{E}[\varphi(G)\mathrm{He}_2(G)]$ by integration by parts.)

So $M = cI + \lambda_2 w_*w_*^\top$ is a rank-1 perturbation of a multiple of the identity. The leading eigenvector is $w_*$ (assuming $\lambda_2 \neq 0$).

**Concentration:** $\hat M - M = \frac{1}{n}\sum_{i=1}^n Z_i$ where $Z_i = y_ix_ix_i^\top - M$ are i.i.d. mean-zero. By an $\varepsilon$-net argument (high-dimensional probability, HDP):

$$
\|\hat M - M\|_\mathrm{op} \leq C\sqrt{\frac{d}{n}} \quad \text{w.h.p.}
$$

By Davis–Kahan (eigenvalue perturbation), $\|v_1(\hat M) - v_1(M)\| \leq C\sqrt{d/n}/|\lambda_2|$, giving weak recovery whenever $n/d \geq c_0$ for $c_0$ sufficiently large. Note: if $\lambda_1 = 0$ (i.e. $k_* \geq 2$) and $n/d < c_1$, then weak recovery via the second-moment method is **impossible** — a sharp threshold.

### WLOG Reduction

WLOG assume $w_* = e_1$ and define $g = \langle x, e_1\rangle = X e_1$ (first coordinate). Then $X = [g \mid \tilde X]^n$ where $\tilde X \in \mathbb{R}^{n\times(d-1)}$ is independent of $g$, and

$$
\hat M = \frac{1}{n}X^\top D X, \qquad D = \mathrm{diag}(y_1,\ldots,y_n).
$$

This is a **rank-2 + Wishart** structure: $\hat M \approx \frac{\lambda_2}{n}(Xe_1)(Xe_1)^\top + W$ where $W = \frac{1}{n}\tilde X^\top D\tilde X$ is approximately a Wishart matrix (spectrum of a low-rank perturbation of a random matrix) — a classical problem in random matrix theory.

---

## The Generative Exponent and the Damian et al. Theorem

### General $k_*$: Degree-$k_*$ Tensor Preprocessing

For general $k_*$, neither linear nor second-moment preprocessing works. The key idea is to preprocess with a **degree-$k_*$ statistic**.

Define the empirical degree-$k_*$ tensor:

$$
\hat T = \frac{1}{n}\sum_{i=1}^n y_i\,\mathrm{He}_{k_*}(x_i) \in (\mathbb{R}^d)^{\otimes k_*},
$$

where $\mathrm{He}_{k_*}(x)$ denotes the $k_*$-th Hermite tensor: $Q_{k_*}(x)_{i_1,\ldots,i_{k_*}} = x_{i_1}\cdots x_{i_{k_*}}$ (symmetrized monomial), with $\mathrm{He}_{k_*}$ obtained by orthogonalizing $Q_0, Q_1, \ldots, Q_{k_*-1}$ (Gram–Schmidt), giving $\mathrm{He}_0 = Q_0$, $\mathrm{He}_1 = Q_1$, $\mathrm{He}_k = Q_k - \text{lower order terms}$.

The expectation is $\mathbb{E}\hat T = \lambda_{k_*} w_*^{\otimes k_*}$, a rank-1 tensor proportional to $w_*^{\otimes k_*}$, since all lower Hermite coefficients $\lambda_1 = \cdots = \lambda_{k_*-1} = 0$ by definition of $k_*$.

**Partial trace:** For $k_*$ even, extract $w_*$ by taking the partial trace. Define the contracted matrix

$$
\hat M^{(k_*)} = \hat T[I^{\otimes(k_*/2-1)}], \quad \hat M^{(k_*)}_{ij} = \sum_{i_1,\ldots,i_{k_*-1}}\hat T_{j,i_1,i_1,i_2,i_2,\ldots,i_{k_*/2-1},i_{k_*/2-1}}.
$$

Then $\mathbb{E}\hat M^{(k_*)} = \lambda_{k_*} w_*w_*^\top$ (a rank-1 matrix), and $\hat w = v_1(\hat M^{(k_*)})$ achieves weak recovery.

:::{prf:theorem} Weak Recovery via Degree-$k_*$ Preprocessing (Damian et al. 2024)
:label: thm-damian

Consider the single-index model $y_i = \varphi(\langle w_*, x_i\rangle) + \varepsilon_i$ with $x_i \sim \mathcal{N}(0,I_d)$, $\varepsilon_i \sim \mathcal{N}(0,\tau^2)$. Let $k_*$ be the generative exponent. If $n \geq Cd^{k_*/2+1}\log n$ for a constant $C$ depending on $\varphi$ and $\tau$, there exists a **polynomial-time algorithm** achieving weak recovery: an estimator $\hat w$ satisfying

$$
\mathbb{P}(|\langle\hat w, w_*\rangle| \geq \varepsilon) \to 1 \quad \text{as } n,d\to\infty.
$$

The algorithm uses the degree-$k_*$ tensor statistic $\hat T = \frac{1}{n}\sum_i y_i\,\mathrm{He}_{k_*}(x_i)$ and the partial-trace-based spectral estimator.

Moreover, no efficient algorithm (in the SQ or LDP framework) can achieve weak recovery with $n \leq Cd^{k_*/2} - 1$ samples: there is a **computational-statistical gap** whenever $k_* \geq 3$.
:::

The key quantity is $\mathbb{E}\hat T = \lambda_{k_*} w_*^{\otimes k_*}$. Note: the ideal estimator would be $\hat w = \operatorname*{arg\,max}\{\langle\hat T, w^{\otimes k_*}\rangle : \|w\| = 1\}$, the best rank-1 approximation to $\hat T$, but this is NP-hard in general (tensor PCA is hard for $k \geq 3$). The partial-trace method sidesteps this by reducing to a matrix spectral problem at the cost of a $\sqrt{d}$ factor in the sample complexity.

### Two-Layer Network Implementation

**Can a two-layer net implement this efficiently?** Here is a gradient-based algorithm:

1. **One step of GD** on the network risk $\hat R_n(a,w)$ with respect to $w$, starting from a random initialization where $w_i = c_i w_* + w_i^\perp$ (unknown):

$$
\nabla_w\hat R_n(w) = \frac{1}{n}\sum_{i=1}^n \ell'(y_i;\langle w,x_i\rangle)\,x_i.
$$

After one GD step: the weights move in the direction $w_i \leftarrow c_i w_* + w_i^\perp + \text{signal} \cdot w_*$, where the signal comes from $\frac{1}{m}\sum_i a_i\nabla_{w_i}\hat R_n$.

2. **Freeze $w_i$** and minimize $\hat R_n$ over $a_i$ (a linear regression): $\phi(x) = (\sigma(w_1^\top x),\ldots,\sigma(w_m^\top x))^\top$, giving the "one-step" kernel $K_m^{(1\mathrm{step})}(x_1,x_2) = \frac{1}{m}\sum_i\sigma(w_i^\top x_1)\sigma(w_i^\top x_2)$.

The key observation: $\|f_*\|_{K_m^{(1\mathrm{step})}} \ll \|f_*\|_{K_m^{(0\mathrm{step})}}$ (the RKHS norm of $f_*$ in the one-step kernel is much smaller than in the zero-step/NTK kernel), meaning $f_*$ is easier to approximate in the learned kernel than the random one.

**First gradient step gradient:**

$$
\nabla\hat R_n(w_0) = \frac{1}{n}\sum_{i=1}^n \ell'(y_i;\,w_0^\top x_i)\,x_i, \quad \ell(y;z) = (y-\sigma(z))^2,
$$

so $\ell'(y_i;w_0^\top x_i) = T_i(y_i)$ evaluated at the current prediction, which is exactly the preprocessing function! As $w_0^\top x_i \approx 0$ at initialization, $T_i(y_i) = \ell'(y_i;0) \propto (y_i - \sigma(0))$, giving the degree-1 preprocessing. For higher $k_*$, one applies $k_*-1$ preprocessing steps to extract the generative exponent.

---

## Gradient and Hessian of the Single-Neuron Risk

The analysis above connects to the gradient and Hessian of the single-neuron empirical risk

$$
\hat R_n(w) = \frac{1}{n}\sum_{i=1}^n \ell(y_i;\,w^\top x_i), \quad e.g.\; \ell(y;z) = (y-\sigma(z))^2.
$$

At $w = w_0$, the gradient and Hessian are

$$
\nabla\hat R_n(w_0) = \frac{1}{n}\sum_{i=1}^n \underbrace{\ell'(y_i;\,w_0^\top x_i)}_{=:T_i(y_i)}\,x_i,
$$

$$
\nabla^2\hat R_n(w_0) = \frac{1}{n}\sum_{i=1}^n \underbrace{\ell''(y_i;\,w_0^\top x_i)}_{=:T_i(y_i)}\,x_ix_i^\top.
$$

Both have the form of a weighted sum of outer products, where the weights $T_i(y_i) = \ell'(y_i; w_0^\top x_i)$ are label-dependent preprocessings. This connects the gradient step to the preprocessing trick: one step of GD at initialization $w_0$ with $w_0^\top x_i \approx 0$ produces the estimator $\tilde w = \nabla\hat R_n(w_0)$, which is exactly the linear preprocessing estimator $\frac{1}{n}\sum_i T_i(y_i) x_i$ for $T_i(y) = \ell'(y;0)$.

For the square loss with $\sigma = \mathrm{ReLU}$: $T(y) = -(y - 0^+) = -y$ (up to a constant), recovering $\tilde w \propto \frac{1}{n}\sum_i y_i x_i$, which is the Stein estimator when $\lambda_1 \neq 0$.
