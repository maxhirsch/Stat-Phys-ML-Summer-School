---
title: "The Barron Norm, Double Descent, and the Neural Tangent Kernel"
subtitle: "ICTP Summer School on Machine Learning — Lecture 5 (Day 7)"
short_title: Barron Norm and NTK
authors:
  - name: Andrea Montanari, transcribed by Max Hirsch with the Claude LLM
subject: Lecture Notes
bibliography: references.bib
---

These notes cover the Day 7 lecture by Andrea Montanari. The first half completes the story of Barron spaces begun in Lecture 4: we give a concrete spectral characterization of the Barron norm via spherical harmonics, show when it is finite, and note its invariance under low-dimensional projections. The second half turns to the empirical phenomenon of **double descent**, introduces the gradient flow ODE for neural network training, and proves the key theorem on convergence to a linearized (NTK) model when the network is sufficiently wide.

## Completing the Barron Norm: Spherical Harmonics Representation

Recall from Lecture 4 the two-layer setting. We work on $X = L^2(\mathbb{R}^d; \mathbb{P})$ with $\mathbb{P} = \mathrm{Unif}(\sqrt{d}\,S^{d-1})$. The generator set is

$$
G = \{\pm\sigma(\langle w, \cdot \rangle) : w \in \mathbb{R}^d,\; \|w\|_2 \leq 1\},
$$

and the Barron norm is $\|f\|_\sigma = \inf\{|\mu| : f(\cdot) = \int \sigma(\langle w, \cdot \rangle)\,\mu(dw)\} = \inf\{\lambda > 0 : f/\lambda \in B_G\}$.

A function $f$ with $\|f\|_\sigma < \infty$ can be represented using $\approx \|f\|_\sigma^2$ neurons with output-layer $\ell^1$ norm $r_0 \geq \|f\|_\sigma$.

### Three Open Questions About $\|f\|_\sigma$

1. What is $\|f\|_\sigma$ concretely?
2. How do we compute it?
3. For which $f$ is it finite?

### Spherical Harmonics and the Gegenbauer Expansion

The sphere $S^{d-1}$ admits an orthonormal basis of **spherical harmonics** $Y_\ell(w) = (Y_{\ell,1}(w),\ldots,Y_{\ell,D(\ell)}(w))$ with $D(\ell) \asymp d^\ell/\ell!$, satisfying $\int Y_\ell(w)Y_\ell(w)^\top\nu_0(dw) = I_{D(\ell)}$ and $\langle Y_{\ell,i}, Y_{\ell,j}\rangle = \delta_{ij}$. The only degree-$\ell$ polynomial invariant under rotations around $z_0 \in S^{d-1}$ is

$$
x \mapsto Q_\ell(\langle z_0, x\rangle), \qquad Q_\ell(\langle z_0, x\rangle) = \frac{1}{\sqrt{D(\ell)}}\,Y_\ell(z_0)^\top Y_\ell(x),
$$

where $Q_\ell$ is the Gegenbauer polynomial of degree $\ell$. As $d\to\infty$, $Q_\ell(v) \to \mathrm{He}_\ell(\sqrt{d}\,v)$ (Hermite polynomials).

### Hermite Coefficients and Finiteness

Working on $\sqrt{d}\,S^{d-1}$, any activation $\sigma$ expands as

$$
\sigma(\sqrt{d}\langle w,x\rangle) = \sum_{\ell=0}^\infty \sigma_\ell\,Q_\ell(\langle w,x\rangle), \qquad \sigma_\ell = \int \sigma(\sqrt{d}\,v)\,Q_\ell(v)\,\varsigma_d(dv) \xrightarrow{d\to\infty} \int \sigma(u)\,\mathrm{He}_\ell(u)\,\gamma(du).
$$

The Barron norm satisfies $\|f\|_\sigma \leq \sqrt{\sum_{\ell\geq 0} \frac{D(\ell)}{\sigma_\ell^2}\|P_\ell f\|_{L^2}^2}$.

:::{prf:theorem} Finiteness of the Barron Norm
If $\sigma_\ell \neq 0$ for all $\ell \geq 0$ (which holds generically — e.g., for all non-polynomial activations), then $\|f\|_\sigma < \infty$ on a **dense** subset of $L^2(S^{d-1})$.
:::

**Degenerate case:** If $\sigma(t) = t$ (linear), then $\sigma_\ell = 0$ for $\ell \neq 1$ and $\|f\|_\sigma = \infty$ for any nonlinear $f$.

### Invariance Under Low-Dimensional Projections

:::{admonition} Remark
:class: note

If $f(x) = \varphi(U^\top x)$ for $U \in \mathbb{R}^{d\times k}$ orthonormal, then $\|f\|_\sigma = \|\varphi\|_\sigma$ — the Barron norm is the same in $d$ dimensions as in $k$. This means functions that depend only on $k \ll d$ directions have Barron norms scaling with $k$, not $d$, evading the curse of dimensionality.
:::

---

## The Double Descent Phenomenon

### Three Empirical Facts

In practice:
1. **Training error $\ll$ test error** — networks interpolate training data.
2. **Models are near optimal** — test error approaches Bayes risk.
3. **No explicit regularization** needed.

### The Double Descent Curve

As $p/n$ (parameters/samples) grows, test risk exhibits a **double descent**:
- $p/n < 1$ (underparameterized): classical U-shaped bias-variance tradeoff; **uniform convergence regime**.
- $p/n \approx 1$ (interpolation threshold): test risk peaks sharply.
- $p/n > 1$ (overparameterized): test risk **decreases again** toward Bayes risk.

Two mechanisms explain why overparameterization helps:

**Implicit (algorithmic) regularization — NTK regime.** Gradient descent finds the minimum-norm interpolant, biased toward smooth solutions.

**Benign overfitting (self-induced regularization).** High-dimensional geometry makes certain interpolating solutions harmless: the spiky noise-fitting component is orthogonal to the signal in function space.

---

## Gradient Flow and the NTK Linearization

### Setup

Parameters $\theta \in \mathbb{R}^p$, network output vector $f_n(\theta) = (f(x_1;\theta),\ldots,f(x_n;\theta))^\top$, Jacobian $Df_n(\theta) \in \mathbb{R}^{n\times p}$. Empirical risk $\hat R_n(\theta) = \frac{1}{2n}\|y - f_n(\theta)\|^2$.

The **gradient flow** is

$$
\dot\theta_t = -\nabla\hat R_n(\theta_t) = \frac{1}{n}\,Df_n(\theta_t)^\top(y - f_n(\theta_t)).
$$

Setting $y_t = f_n(\theta_t)$ and $K_t = Df_n(\theta_t)Df_n(\theta_t)^\top \in \mathbb{R}^{n\times n}$ (the **NTK matrix**):

$$
\dot y_t = -\frac{1}{n}K_t(y_t - y).
$$

### The Linearized Model

If gradient flow stays near $\theta_0$, Taylor-expand: $f_n(\theta) \approx f_n(\theta_0) + Df_n(\theta_0)(\theta - \theta_0)$, giving the **linearized empirical risk**

$$
\hat R_n^\mathrm{lin}(\theta) = \frac{1}{2n}\|y - f_n(\theta_0) - Df_n(\theta_0)^\top(\theta-\theta_0)\|^2.
$$

The corresponding gradient flow $\dot{\bar\theta}_t = -\nabla\hat R_n^\mathrm{lin}(\bar\theta_t)$ is linear and exactly solvable; the linearized predictor $\bar f_\mathrm{lin}(\cdot) = f(\cdot;\theta_0) + Df(\cdot;\theta_0)(\bar\theta_t - \theta_0)$ converges to a kernel ridge regression solution.

### Convergence Theorem

:::{prf:theorem} NTK Convergence [@du2019gradient; @chizat2020implicit]
:label: thm-ntk

Let $L = \mathrm{Lip}(Df_n)$. Assume

$$
L\|y - f_n(\theta_0)\|_2 \leq \frac{1}{4}\,\sigma_\mathrm{min}(Df_n(\theta_0))^2.
$$

Then for all $t > 0$:

(i) **Exponential convergence:** $\hat R_n(\theta_t) \leq \hat R_n(\theta_0)\,e^{-\lambda_0 t}$, $\quad\lambda_0 = \frac{\sigma_\mathrm{min}(Df_n(\theta_0))^2}{2n}$.

(ii) **Parameters stay near initialization:** $\|\theta_t - \theta_0\|_2 \leq \frac{2}{\sigma_\mathrm{min}(Df_n(\theta_0))}\|y - f_n(\theta_0)\|_2$.

(iii) **Predictor tracks the linearized model:** $\|f_n(\theta_t) - f_n(\bar\theta_t)\|_{L^2(\mathbb{P})} \leq \ldots$
:::

### Full Proof of Part (i) and Part (ii)

**Setup.** Let $\sigma_\mathrm{min} = \sigma_\mathrm{min}(Df_n(\theta_0))$, $r_* = \sigma_\mathrm{min}^2/(2Ln)$, and $t_* = \inf\{t : \|\theta_t - \theta_0\| \geq r_*\}$. We show $t_* = \infty$.

**For $t < t_*$**, Lipschitz continuity of $Df_n$ gives

$$
\|Df_n(\theta_t) - Df_n(\theta_0)\|_\mathrm{op} \leq L\|\theta_t - \theta_0\| < Lr_* = \frac{\sigma_\mathrm{min}^2}{2n} \cdot \frac{n}{1} = \frac{\sigma_\mathrm{min}}{2},
$$

using the matrix inequality $\sigma_\mathrm{min}(A) \geq \sigma_\mathrm{min}(B) - \|A-B\|_\mathrm{op}$:

$$
\sigma_\mathrm{min}(Df_n(\theta_t)) \geq \frac{\sigma_\mathrm{min}}{2}, \quad \lambda_\mathrm{min}(K_t) \geq \frac{\sigma_\mathrm{min}^2}{4}.
$$

**Decay of $\|y_t - y\|^2$.** Since $\dot y_t = -\frac{1}{n}K_t(y_t - y)$:

$$
\frac{d}{dt}\|y_t - y\|^2 = -\frac{2}{n}\langle y_t - y, K_t(y_t-y)\rangle \leq -\frac{\sigma_\mathrm{min}^2}{2n}\|y_t - y\|^2 = -\lambda_0\|y_t-y\|^2.
$$

This gives $\|y_t - y\|^2 \leq \|y_0 - y\|^2 e^{-\lambda_0 t}$, i.e., $\hat R_n(\theta_t) \leq \hat R_n(\theta_0)e^{-\lambda_0 t}$.

**Bounding $\|\theta_t - \theta_0\|$.** From $\dot\theta_t = \frac{1}{n}Df_n(\theta_t)^\top(y - y_t)$:

$$
\frac{d}{dt}\|y_t - y\| = -\frac{1}{n}\frac{\|Df_n(\theta_t)(y_t-y)\|^2}{\|y_t-y\|} \leq -\frac{\sigma_\mathrm{min}}{2} \cdot \frac{1}{n}\|Df_n(\theta_t)(y_t-y)\| = -\frac{\sigma_\mathrm{min}}{2}\|\dot\theta_t\|.
$$

Hence $\frac{d}{dt}\!\left(\|y_t - y\| + \frac{\sigma_\mathrm{min}}{2}\|\theta_t - \theta_0\|\right) \leq 0$, which integrates to

$$
\boxed{\|y_t - y\| + \frac{\sigma_\mathrm{min}}{2}\|\theta_t - \theta_0\| \leq \|y_0 - y\|.}
$$

Therefore $\|\theta_t - \theta_0\| \leq 2\|y_0 - y\|/\sigma_\mathrm{min} < r_*$, so $t_* = \infty$ and both bounds hold globally. $\square$

### Interpretation

The condition $L\|y - f_n(\theta_0)\|_2 \leq \frac{1}{4}\sigma_\mathrm{min}^2$ is satisfied when the network is wide ($m\to\infty$): at random Gaussian initialization, $\sigma_\mathrm{min}(Df_n(\theta_0)) = \Omega(\sqrt{n})$, $\|y - f_n(\theta_0)\|_2 = O(\sqrt{n})$, and $L = O(1/\sqrt{m})$, so the product is $O(\sqrt{n/m}) \to 0$.

Since $\|\theta_t - \theta_0\|$ is bounded, $f_n(\theta_t) \approx f_n(\bar\theta_t)$ throughout training — the **lazy training** (NTK) regime. The limit of gradient flow is the **kernel ridge regression** (KRR) estimator with the NTK kernel $K(x,x') = \langle Df(\cdot;x_0,\theta_0), Df(\cdot;x',\theta_0)\rangle$.
