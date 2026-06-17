---
title: Statistical Physics Approaches to High-Dimensional Learning
subtitle: "ICTP Summer School on Machine Learning — Lecture 3"
short_title: Dynamics of Learning
authors:
  - name: Francesca Mignacco, transcribed by Max Hirsch with the Claude LLM
subject: Lecture Notes
---

These notes cover the third lecture, returning to Francesca Mignacco, on the **dynamics of learning**. The objective is to derive effective, low-dimensional descriptions of training dynamics and apply them to commonly used algorithms such as stochastic gradient descent (SGD). Two complementary settings are covered: **online learning** in two-layer neural networks (where each data point is seen only once, so the dynamics reduce to a small, closed system of ordinary differential equations), and **batch learning**, which requires the heavier machinery of dynamical mean-field theory, the cavity method, and path-integral formulations (the subject of slides referenced throughout, not transcribed here). The online-learning analysis follows @goldt2019dynamics, with extensions discussed in @goldt2020hidden and @refinetti2021classifying.

# Training Dynamics: Setting the Stage

The basic training algorithm is **gradient descent**: at each step, update the weights using the gradient of the loss over the *entire* dataset. In practice, this full-batch gradient is expensive to compute at every step, motivating **stochastic gradient descent (SGD)**: at a given step, only update the weights based on the gradient computed on a small, randomly chosen subset of the data (a **minibatch**), rather than the whole training set.

## Modeling the Noise of SGD

Writing $w_j$ for a generic weight, one full SGD step can be decomposed as

$$
w_j(t + dt) = \underbrace{w_j(t) - dt\, \nabla\mathcal{L}(\underline{w})}_{\text{full-batch gradient step}} \;+\; \underbrace{dt\Big(\nabla\mathcal{L}(\underline{w}) - \widehat{\nabla}_B\mathcal{L}(\underline{w})\Big)}_{\text{SGD noise}},
$$

where $\nabla\mathcal{L}$ is the true (full-dataset) gradient and $\widehat{\nabla}_B\mathcal{L}$ is the gradient estimated from a minibatch of size $B$. The difference between the two is the **SGD noise** term: it vanishes in expectation (the minibatch gradient is an unbiased estimator of the full gradient) but has nonzero variance that depends on the batch size and the data distribution.

**Physical picture.** This decomposition is structurally identical to a stochastic differential equation with a deterministic drift (the full-batch gradient) plus a noise term — exactly the setup of **Langevin dynamics** in statistical physics, where a particle moves down an energy landscape while being buffeted by thermal noise. Modeling the SGD noise as approximately Gaussian (a common and often justified approximation, since the minibatch gradient is itself an average over many data points and the central limit theorem applies) turns the discrete SGD update into a continuous-time Langevin equation, opening the door to the same tools used to study thermally-driven particles: drift-diffusion analysis, Fokker–Planck equations, and fluctuation-dissipation relations.

The remainder of the lecture focuses on **tracking the whole trajectory** of the dynamics, not just its fixed points — that is, understanding how quantities like the generalization error evolve as a function of training time, rather than only their asymptotic value once training has converged. This is a strictly harder and more informative question than the static analyses of Lectures 1 and 2.

# Online Learning in Two-Layer Narrow Networks

The cleanest setting in which to study training dynamics in closed form is **online learning**: at every step the network sees a *fresh*, never-before-seen data point, so there is no question of overfitting or revisiting old data, and the dynamics can be tracked exactly via a small number of scalar order parameters. This setup, and the extensions discussed below, are due to @goldt2019dynamics, @goldt2020hidden, and @refinetti2021classifying.

## Setup: Teacher and Student Networks

The input data are high-dimensional Gaussian vectors,

$$
x \in \mathbb{R}^N, \qquad x \sim \mathcal{N}(0, I_N).
$$

**Teacher.** A fixed, "frozen" two-layer network generates the labels:

$$
y = \sum_{r=1}^{K^*} v_r^*\, \sigma^*(h_r^*), \qquad h_r^* = \frac{w_r^* \cdot x}{\sqrt{N}}, \qquad w_r^* \in \mathbb{R}^N,\; v_r^* \in \mathbb{R}.
$$

Here $K^*$ is the number of hidden units in the teacher, and the weights $\{w_r^*, v_r^*\}$ are drawn once at the start and never updated — they define the ground-truth function the student must learn.

**Student.** A second two-layer network, with possibly different width $K$, is trained to approximate the teacher:

$$
\hat{y}(x) = \sum_{k=1}^{K} v_k\, \sigma(h_k), \qquad h_k = \frac{w_k \cdot x}{\sqrt{N}}, \qquad w_k \in \mathbb{R}^N,\; v_k \in \mathbb{R} \;\;(\text{learned}).
$$

Typical choices of activation function $\sigma$ include $\tanh(h)$ (smooth, bounded) and $\max\{0, h\}$ (ReLU, the standard choice in modern deep learning). The teacher and student may use the same or different activations $\sigma^*, \sigma$.

**Loss.** The discrepancy between student and teacher on a single example is measured by the squared loss,

$$
\ell = \frac{1}{2}\Delta^2, \qquad \Delta = \hat{y}(x) - y.
$$

## Online SGD Updates

In the online setting, at each time step $\mu$ a fresh sample $x^\mu$ (with corresponding teacher label $y^\mu$) is drawn, and a single SGD step is taken using only this sample. For the first-layer weights,

$$
\delta w_k^\mu = w_k^{\mu+1} - w_k^\mu = -\eta\, \nabla_{w_k}\ell^\mu = -\eta\, \Delta^\mu\, v_k^\mu\, \sigma'(h_k^\mu)\, \frac{x^\mu}{\sqrt{N}},
$$

where $\eta$ is the learning rate, and the chain rule produces the factor $\sigma'(h_k^\mu)$ (derivative of the activation) and $v_k^\mu$ (since $\hat{y}$ depends linearly on $v_k$, scaling the gradient with respect to $w_k$). Similarly, for the second-layer (readout) weights,

$$
\delta v_k^\mu = v_k^{\mu+1} - v_k^\mu = -\frac{\eta}{N}\, \partial_{v_k}\ell^\mu = -\frac{\eta}{N}\, \Delta^\mu\, \sigma(h_k^\mu).
$$

*(Note: the readout-weight update carries a $1/N$ rescaling relative to the first-layer update, reflecting the fact that in this scaling convention the two layers are trained at different effective rates — a choice that ensures both layers contribute comparably to the dynamics as $N\to\infty$; this is sometimes called the "mean-field" or "Maclaurin" scaling for two-layer networks.)*

## Order Parameters: Why the Pre-Activations Suffice

**Remark: the pre-activations are jointly Gaussian.** Since $x \sim \mathcal{N}(0, I_N)$ and each $h_k = w_k \cdot x/\sqrt{N}$, $h_r^* = w_r^*\cdot x/\sqrt{N}$ is a linear function of $x$, the entire collection $\{h_k\}_{k=1}^K \cup \{h_r^*\}_{r=1}^{K^*}$ is jointly Gaussian for any fixed set of weights. Each has mean zero,

$$
\mathbb{E}_{x^\mu}\big[h_k^\mu\big] = \mathbb{E}\big[h_k^{*\mu}\big] = 0,
$$

and their joint second moments are controlled entirely by four **order parameters**, all defined as $N$-dimensional inner products between weight vectors, rescaled by $N$:

$$
\mathbb{E}_{x^\mu}\big[h_k^\mu h_{k'}^\mu\big] = \frac{1}{N}w_k^\mu \cdot w_{k'}^\mu \equiv Q_{kk'}^\mu \quad \text{(student–student overlap)},
$$

$$
\mathbb{E}_{x^\mu}\big[h_k^\mu h_r^{*\mu}\big] = \frac{1}{N}w_k^\mu \cdot w_r^{*} \equiv M_{kr}^\mu \quad \text{(teacher–student alignment)},
$$

$$
\mathbb{E}_{x^\mu}\big[h_r^{*\mu} h_{r'}^{*\mu}\big] = \frac{1}{N}w_r^* \cdot w_{r'}^* \equiv T_{rr'} \quad \text{(teacher self-overlap, fixed since the teacher is frozen)}.
$$

*(The intermediate step uses $\mathbb{E}_{x^\mu}[x^\mu (x^\mu)^\top] = I_N$, so that $w_k^\top \mathbb{E}[xx^\top] w_{k'}/N = w_k\cdot w_{k'}/N$.)* The matrix $Q$ is the analogue of the self-overlap matrix from Lecture 1's replica calculation; $M$ plays the role of the magnetization (alignment with the teacher); $T$ is fixed data about the (frozen) teacher.

**Remark: the order parameters are sufficient statistics for the generalization error.** This is the key structural fact that makes the whole approach work. The generalization error, averaged over a fresh test point $x$, is

$$
\mathcal{E}_\text{gen}(w, v; w^*, v^*) = \frac{1}{2}\mathbb{E}_x\big[\Delta^2\big] = \frac{1}{2}\,\mathbb{E}_{h, h^*}\!\left[\left(\sum_k v_k\,\sigma(h_k) - \sum_r v_r^*\,\sigma^*(h_r^*)\right)^2\right],
$$

and crucially, because $(h, h^*)$ is jointly Gaussian and its covariance structure is entirely determined by $(Q, M, T)$, this expectation can be written as a function of the order parameters alone (together with the readout weights $v, v^*$):

$$
\mathcal{E}_\text{gen} = \mathcal{E}_g(Q, M, v; T, v^*).
$$

In other words, $\mathcal{E}_\text{gen}$ does not depend on the full $N$-dimensional weight vectors $w_k$ individually, only on their $\mathcal{O}(K^2)$ pairwise overlaps. This is the same dimensional reduction seen in the replica calculations of Lecture 1: a problem that started in $\mathbb{R}^N$ collapses onto a handful of macroscopic order parameters, but here the reduction applies to the entire *trajectory* of training, not just to the converged equilibrium state.

# Dynamics of the Order Parameters

Take the joint limit $\rho, N \to \infty$ with $\alpha = \rho/N = \mathcal{O}(1)$ (sample complexity, exactly as in Lecture 1) while $\eta, K, K^*$ remain $\mathcal{O}_N(1)$ (fixed, not growing with $N$). The goal is to derive **closed, deterministic ODEs** for $(Q, M)$ as functions of training time, by taking the $N\to\infty$ limit of their discrete SGD updates.

## Deriving the Update for $M$

From the definition $M_{kr}^\mu = w_k^\mu \cdot w_r^* / N$ and the update rule for $w_k$,

$$
\delta M_{kr}^\mu = M_{kr}^{\mu+1} - M_{kr}^\mu = \frac{w_r^* \cdot \delta w_k^\mu}{N} = -\frac{\eta}{N}\, v_k^\mu\, \Delta^\mu\, \sigma'(h_k^\mu)\, h_r^{*\mu},
$$

where the last equality uses $w_r^* \cdot x^\mu/\sqrt{N} = h_r^{*\mu}$ directly, eliminating the explicit dependence on the high-dimensional vectors $w_r^*$ and $x^\mu$ in favor of the *scalar* pre-activation $h_r^{*\mu}$.

## Deriving the Update for $Q$

Similarly, from $Q_{kk'}^\mu = w_k^\mu \cdot w_{k'}^\mu/N$, using the product rule on $\delta(w_k\cdot w_{k'}) = \delta w_k \cdot w_{k'} + w_k\cdot \delta w_{k'} + \delta w_k \cdot \delta w_{k'}$:

$$
\delta Q_{kk'}^\mu = \frac{\delta w_k^\mu \cdot w_{k'}^\mu}{N} + \frac{w_k^\mu \cdot \delta w_{k'}^\mu}{N} + \frac{\delta w_k^\mu \cdot \delta w_{k'}^\mu}{N}.
$$

The first two (linear-in-$\delta w$) terms give

$$
-\frac{\eta}{N}\,\Delta^\mu\Big[v_k^\mu\,\sigma'(h_k^\mu)\,h_{k'}^\mu + v_{k'}^\mu\,\sigma'(h_{k'}^\mu)\,h_k^\mu\Big],
$$

while the quadratic (second-order in $\delta w$) term gives

$$
\frac{\eta^2}{N}\,(\Delta^\mu)^2\, v_k^\mu v_{k'}^\mu\, \sigma'(h_k^\mu)\,\sigma'(h_{k'}^\mu)\,\frac{\|x^\mu\|_2^2}{N}.
$$

*(Note: the quadratic term arises because $\delta w_k$ and $\delta w_{k'}$ are both proportional to $x^\mu$, so their inner product produces $\|x^\mu\|_2^2/N \to 1$ by the law of large numbers for a Gaussian vector in $\mathbb{R}^N$ — this $\mathcal{O}(\eta^2)$ correction is exactly analogous to the Itô correction term that appears when deriving a continuous-time SDE from a discrete-time random walk, and is essential to keep even though it is formally higher order in $\eta$, because it survives the same parameter-counting that makes the linear terms $\mathcal{O}(1/N)$ as well.)*

## The Deterministic Limit

Each individual increment $\delta M_{kr}^\mu$ or $\delta Q_{kk'}^\mu$ is itself a random variable, since it depends on the random draw of $x^\mu$ at step $\mu$. The key step in passing to a deterministic description is to split each increment into its expectation (the **drift**, conditional on the current order parameters) plus a **zero-mean fluctuation**:

$$
\delta M_{kr}^\mu = \underbrace{\mathbb{E}_{h^\mu, h^{*\mu}}\big[\delta M_{kr}^\mu\big]}_{\text{drift, } \mathcal{O}(1/N)} + \underbrace{\delta M_{kr}^\mu - \mathbb{E}\big[\delta M_{kr}^\mu\big]}_{\text{zero-mean fluctuation, } \mathcal{O}(1/N)}.
$$

The drift term is $\mathcal{O}(1/N)$ per step (visible directly from the explicit $1/N$ or $1/\sqrt{N}$ prefactors derived above). The fluctuation term is *also* $\mathcal{O}(1/N)$ per step in magnitude, but — crucially — being a zero-mean, weakly-correlated random variable, its accumulated effect over many steps grows only like the square root of the number of steps (a random-walk scaling), rather than linearly.

This separation of scales is what allows a **law of large numbers** argument: summing the drift over $\mathcal{O}(N)$ steps (i.e., over one "epoch" in which every coordinate of $w$ has, on average, been updated $\mathcal{O}(1)$ times) gives an $\mathcal{O}(1)$ total drift, while the accumulated fluctuations over the same $\mathcal{O}(N)$ steps grow only as $\mathcal{O}(1/\sqrt{N})$ and vanish as $N\to\infty$. This is made precise via the rescaled, continuous training-time variable

$$
\alpha = \frac{\mu}{N} \quad \text{(number of samples seen, per input dimension)},
$$

so that summing $\mathcal{O}(N)$ discrete steps corresponds to advancing $\alpha$ by an $\mathcal{O}(1)$ amount:

$$
M_{kr}^{\alpha N} - M_{kr}^{0} = \underbrace{\sum_{\mu=0}^{\alpha N - 1}\mathbb{E}\big[\delta M_{kr}^\mu\big]}_{\mathcal{O}(1)} + \underbrace{\sum_{\mu=0}^{\alpha N-1}\frac{1}{N}\delta_{kr}^\mu}_{\mathcal{O}(1/\sqrt{N})},
$$

where the fluctuation sum is written suggestively as $\frac{1}{N}\sum \delta_{kr}^\mu$ to emphasize that it is a sum of $\mathcal{O}(N)$ terms each of size $\mathcal{O}(1/N)$, which by a martingale central limit theorem concentrates with fluctuations of order $\sqrt{\mathcal{O}(N)}\times \mathcal{O}(1/N) = \mathcal{O}(1/\sqrt{N})$.

As $N\to\infty$ with $\alpha$ held fixed, the fluctuation term vanishes and $M_{kr}^{\alpha N} \to M_{kr}(\alpha)$, a deterministic, continuous function of $\alpha$, satisfying the **ordinary differential equation**

$$
\frac{dM_{kr}}{d\alpha} = \mathbb{E}_{h, h^*}\big[\delta M_{kr}\big]\cdot N \quad (\text{the drift, in continuum form}),
$$

and analogously for $Q_{kk'}(\alpha)$. This is the central result of the online-learning approach: an $N$-dimensional, stochastic, discrete-time learning process becomes, in the high-dimensional limit, a small ($\mathcal{O}(K^2)$-dimensional) deterministic system of ODEs in continuous "time" $\alpha$. Rigorous proofs of this concentration (i.e. that the fluctuation terms indeed vanish and the ODE limit is exact) are given by @goldt2019dynamics, and refined or extended in subsequent work, including @veiga2022phase and @benarous2021online (see slides for full statements and proof techniques).

## Small-Learning-Rate Expansion

**Remark.** Collect all the order parameters and readout weights into a single state vector,

$$
\Omega \equiv \big(\{M_{kr}\}, \{Q_{kk'}\}, \{v_k\}\big) \in \mathbb{R}^{KK^* + \frac{K}{2}(K+1) + K}.
$$

*(The dimension count: $KK^*$ entries in $M$, $\binom{K+1}{2} = \frac{K(K+1)}{2}$ independent entries in the symmetric matrix $Q$, and $K$ entries in $v$.)* Expanding the drift in powers of the learning rate $\eta$ (since $\delta v_k^\mu = -\eta\,\partial_{v_k}\ell^\mu/N$ is already $\mathcal{O}(\eta)$, and the $Q$-update's quadratic term contributes at $\mathcal{O}(\eta^2)$), the full ODE for $\Omega$ takes the form

$$
\frac{d\Omega}{d\alpha} = \eta\, F_1(\Omega) + \eta^2\, F_2(\Omega) + \mathcal{O}(\eta^3),
$$

for some (model-dependent) functions $F_1, F_2$. Separately, note that the expected update for the first-layer weights is *exactly* a gradient-descent step on the generalization error itself:

$$
\mathbb{E}_{x^\mu}\big[\delta w_k^\mu\big] = -\eta\,\mathbb{E}_{x^\mu}\big[\nabla_{w_k}\ell^\mu\big] = -\eta\,\nabla_{w_k}\mathcal{E}_\text{gen},
$$

confirming that, in expectation, online SGD performs gradient flow on the population loss $\mathcal{E}_\text{gen}$ rather than the (noisier) per-sample loss $\ell^\mu$ — the fluctuations around this drift are exactly the $\mathcal{O}(1/\sqrt N)$ corrections analyzed above.

In the regime of small learning rate, it is natural to rescale time by $\tilde\alpha = \alpha\eta$ (absorbing the leading factor of $\eta$ into the time variable, much as one rescales time when taking a continuum/diffusive limit of a random walk), so that to leading order in $\eta$,

$$
\frac{d\Omega}{d\tilde\alpha} = F_1(\Omega) + \mathcal{O}(\eta),
$$

i.e. the leading-order dynamics become *independent of the learning rate* once time is measured in the rescaled units $\tilde\alpha$. This is the standard trick for relating the "many small steps" and "few large steps" regimes of SGD, and is used extensively in the slides to produce phase diagrams of the learning dynamics (e.g. plateaus, specialization transitions, and other qualitative phenomena in the trajectory of $\mathcal{E}_\text{gen}(\alpha)$) — see slides for the explicit forms of $F_1, F_2$ for specific teacher–student architectures.

# References

This lecture builds on @goldt2019dynamics for the core online-learning ODE framework, with the hidden manifold model extension of @goldt2020hidden and the Gaussian-mixture classification extension of @refinetti2021classifying mentioned as further reading. The batch-learning case (dynamical mean-field theory, cavity method, path-integral formulation) is deferred to the slides referenced at the start of the lecture.
