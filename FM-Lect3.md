---
title: Statistical Physics Approaches to High-Dimensional Learning
subtitle: "ICTP Summer School on Machine Learning — Lecture 3"
short_title: Dynamics of Learning
authors:
  - name: Francesca Mignacco (Princeton University & CUNY Graduate Center), transcribed by Max Hirsch with the Claude LLM
subject: Lecture Notes
venue: ICTP-INdAM-SLMath Summer Graduate School for Machine Learning, Trieste, 15–19 June 2026
bibliography: references.bib
---

These notes cover the third lecture on the **dynamics of learning**, spanning Days 3 and 4 of the school. Two complementary settings are developed: **online learning** in two-layer narrow networks, where each data point is seen exactly once and the dynamics reduce to a closed deterministic ODE system; and **multi-pass SGD** via the dynamical cavity (mean-field) method, which handles temporal memory effects when data are reused. The online-learning framework follows {cite}`goldt2019dynamics; saad1995online`, with extensions in {cite}`goldt2020hidden; refinetti2021classifying`. The lecture opens with a tour of recent applications of the statics theory.

**A note on perspective for PDE readers.** The central achievement of this lecture is a rigorous passage from an $N$-dimensional stochastic process (SGD on $\mathbf{w} \in \mathbb{R}^N$) to a low-dimensional deterministic ODE system. This is exactly analogous to deriving a homogenised or reduced-order model from a fine-scale PDE: the large-$N$ limit plays the role of the homogenisation limit, and the order parameters $(M, Q, v)$ are the effective macroscopic variables. The multi-pass DMFT section adds a further ingredient — temporal memory / retarded friction — which has no direct analogue in standard ODE theory but closely resembles the Volterra integro-differential equations that arise in viscoelastic or history-dependent PDE models.

# Applications of the Statics Theory

## Hidden Manifold Model

Real data do not fill $\mathbb{R}^N$ uniformly; they live on or near a low-dimensional manifold. The **hidden manifold model** {cite}`goldt2020modeling; gerace2020generalisation` captures this: latent codes $c^\mu \in \mathbb{R}^L$ pass through a random feature map,

$$x^\mu = \sigma(F^\top c^\mu)\in\mathbb{R}^N, \qquad y^\mu = f_*(c^\mu\cdot w_*),$$

where $F \in \mathbb{R}^{N\times L}$ is a random matrix. A **Gaussian Equivalence Theorem** (GET) shows that the generalisation and training properties of $\{x^\mu, y^\mu\}$ are statistically equivalent to those of a simpler Gaussian surrogate

$$\tilde{x}^\mu = \kappa_1 F^\top c^\mu + \kappa_* z^\mu, \quad z^\mu \sim \mathcal{N}(0,I_N),$$

with coefficients $\kappa_1 = \mathbb{E}_\xi[\xi\,\sigma(\xi)]$ and $\kappa_*^2 = \mathbb{E}_\xi[\sigma(\xi)^2] - \kappa_1^2$, $\xi\sim\mathcal{N}(0,1)$. The GET is proved via Lindeberg-type interpolation arguments. It connects the replica/AMP analysis of Lectures 1–2 directly to structured-data regimes.

## Neural Manifold Capacity

In neuroscience and deep learning, object representations are smooth manifolds of neural population activity. The **manifold capacity** $\alpha^*(\kappa, R_M, D_M)$ — maximum number of linearly separable object manifolds per neuron — is computed exactly by the replica method {cite}`chung2018classification; mignacco2025neural`. It depends on the manifold's intrinsic dimension $D_M$ and radius $R_M$, and provides a rigorous link between geometric properties of representations and the number of dichotomies a linear readout can implement.

## Bayesian Learning in Wide Networks

For a two-layer network $y = a^\top\phi(W_1 x)$ in the **proportional width regime** $N_1/N_0 = \mathcal{O}(1)$, Bayesian inference can be analysed exactly in the proportional-sample limit {cite}`cui2023bayes; camilli2023fundamental` (linear scaling $P = \alpha N_0$) and in the quadratic-sample regime {cite}`maillard2024bayes; erba2024bayes; barbier2026bayes` ($P = \alpha N_0^2$). In both regimes the posterior collapses to a low-dimensional order-parameter description.

## Statistical Mechanics of Transformers

A full statistical-mechanics theory of deep multi-head self-attention is developed in {cite}`tiberi2024dissecting`. The main result: under Bayesian inference in the proportional limit, the network performs **kernel ridge regression** with an effective "renormalised" kernel

$$\mathscr{K}(x^\mu, x^\nu) = \frac{1}{H^L}\sum_{\pi, \pi' \in \Pi} U^{\pi\pi'} C_{\pi\pi'}^{\mu\nu},$$

where $\pi$ ranges over **attention paths** (sequences of head choices across $L$ layers), $C_{\pi\pi'}^{\mu\nu}$ couples inputs $\mu,\nu$ along paths $\pi,\pi'$, and the order-parameter matrix $U^{\pi\pi'}$ encodes inter-path correlations. Two mechanisms emerge: *attention path suppression* (down-weighting uninformative paths) and *attention path coupling* (exploiting correlations between paths). Counterintuitively, smaller pruned networks can outperform larger ones: for $N \lesssim 100$ the off-diagonal $U^{\pi\pi'}$ terms are active; for $N \gtrsim 1000$ path coupling vanishes ($U$ becomes diagonal) and the large network uses a less informative kernel.

# Training Dynamics: Setting the Stage

## From the Statics Picture to Dynamics

Lectures 1–2 characterised the *endpoint* of training — the fixed point $\hat{\mathbf{w}}$ reached by gradient descent and its generalisation error. This lecture asks: how does the trajectory $\mathbf{w}^t$ get there? Two complementary regimes:

- **Online learning** (each datum seen once): individual SGD steps are independent, enabling an exact derivation of deterministic ODEs for macroscopic order parameters.
- **Multi-pass SGD** (data reused across epochs): past gradients create temporal correlations (memory), requiring a more elaborate dynamical mean-field theory.

## Gradient Descent and SGD

**Full-batch gradient descent** uses the complete dataset at every step:

$$w(t+dt) = w(t) - dt\,\nabla_w\mathcal{L}(w;\mathcal{D}).$$

**Stochastic gradient descent (SGD)** uses a mini-batch $\mathcal{B}(t)$:

$$w(t+dt) = w(t) - dt\,\hat\nabla_{\mathcal{B}}\mathcal{L}(w).$$

The difference $\nabla\mathcal{L} - \hat\nabla_{\mathcal{B}}\mathcal{L}$ is zero-mean noise; for large batches the CLT makes it approximately Gaussian, yielding a Langevin SDE. The key observation: as $N \to \infty$ an $N$-dimensional weight trajectory collapses onto a deterministic ODE for a handful of scalar order parameters — exactly as a fine-scale PDE system collapses to an effective macroscopic equation.

# Online Learning in Two-Layer Networks

## Setup: Teacher and Student

**Data:** $x^\mu \sim \mathcal{N}(0,I_N)$, i.i.d.

**Teacher (frozen):** $y^\mu = \sum_{r=1}^{K^*}v_r^*\,\sigma^*(h_r^{*\mu})$, $h_r^{*\mu} = \frac{w_r^*\cdot x^\mu}{\sqrt{N}}$, $w_r^* \in \mathbb{R}^N$, $v_r^* \in \mathbb{R}$ fixed.

**Student (trained):** $\hat{y}^\mu = \sum_{k=1}^K v_k\,\sigma(h_k^\mu)$, $h_k^\mu = \frac{w_k\cdot x^\mu}{\sqrt{N}}$, $w_k \in \mathbb{R}^N$, $v_k \in \mathbb{R}$ learned. Typical activations: $\sigma(h) = \tanh(h)$, $\operatorname{erf}(h/\sqrt{2})$, or $\max\{0,h\}$.

**Squared loss:** $\ell = \frac{1}{2}\Delta^2$, $\Delta = \hat{y} - y$.

The normalisation $1/\sqrt{N}$ in $h_k^\mu = w_k \cdot x^\mu / \sqrt{N}$ ensures the pre-activations are $\mathcal{O}(1)$ regardless of $N$ — the same role as a mesh-width normalisation in finite-difference schemes.

## Online SGD Updates

At step $\mu$ a fresh i.i.d.\ sample $x^\mu$ is drawn and the weights are updated:

$$w_k^{\mu+1} - w_k^\mu = -\eta\,v_k^\mu\,\Delta^\mu\,\sigma'(h_k^\mu)\,\frac{x^\mu}{\sqrt{N}},$$

$$v_k^{\mu+1} - v_k^\mu = -\frac{\eta}{N}\,\Delta^\mu\,\sigma(h_k^\mu).$$

The factor $1/N$ in the second update ensures that all order parameters evolve on the same timescale.

## Order Parameters and Their Sufficiency

Since $x^\mu \sim \mathcal{N}(0,I_N)$, the pre-activations $(h_k^\mu, h_r^{*\mu})$ are jointly Gaussian with zero mean and second moments:

$$Q_{kk'} = \frac{w_k\cdot w_{k'}}{N} \quad \text{(student–student overlap)}, \qquad M_{kr} = \frac{w_k\cdot w_r^*}{N} \quad \text{(teacher–student alignment)},$$

$$T_{rr'} = \frac{w_r^*\cdot w_{r'}^*}{N} \quad \text{(teacher self-overlap, fixed from the start)}.$$

These $\mathcal{O}(K^2)$ scalar quantities are **sufficient statistics** for the generalisation error: since $(h, h^*)$ is jointly Gaussian with covariance determined entirely by $(Q, M, T)$,

$$\mathcal{E}_\text{gen} = \frac{1}{2}\mathbb{E}_{h,h^*}\!\left[\left(\textstyle\sum_k v_k\sigma(h_k) - \sum_r v_r^*\sigma^*(h_r^*)\right)^2\right],$$

where $(h,h^*) \sim \mathcal{N}(0, \Sigma)$ with $\Sigma$ built from $(Q,M,T)$. The entire $N$-dimensional weight configuration is compressed into a matrix of size $\mathcal{O}(K \times K^*)$.

## Closed ODE System for the Order Parameters

In the joint limit $N, p \to \infty$ with $\alpha = p/N = \mathcal{O}(1)$, the stochastic increments of the order parameters split as

$$\delta M_{kr}^\mu = \underbrace{\mathbb{E}[\delta M_{kr}^\mu]}_{\text{drift, } \mathcal{O}(1/N)} + \underbrace{\text{zero-mean fluctuation}}_{\mathcal{O}(1/\sqrt{N})}.$$

Summing $\mathcal{O}(N)$ steps (one epoch, $\Delta\alpha = 1$), the fluctuation sum is $\mathcal{O}(1/\sqrt{N}) \to 0$. Setting $\tilde\alpha = \alpha\eta$ (rescaled time), the order parameters satisfy deterministic ODEs {cite}`goldt2019dynamics; saad1995online`:

$$\frac{dM_{ir}}{d\alpha} = -\eta\,v_i\,\mathbb{E}_{h,h^*}\!\big[\Delta\,\sigma'(h_i)\,h_r^*\big],$$

$$\frac{dQ_{ij}}{d\alpha} = -\eta\!\left(v_i\,\mathbb{E}[\Delta\,\sigma'(h_i)\,h_j] + v_j\,\mathbb{E}[\Delta\,\sigma'(h_j)\,h_i]\right) + \underbrace{\eta^2 v_i v_j\,\mathbb{E}[\Delta^2\,\sigma'(h_i)\,\sigma'(h_j)]}_{\text{Itô correction (from discrete steps)}},$$

$$\frac{dv_k}{d\alpha} = -\frac{\eta}{v}\,\mathbb{E}_{h,h^*}\!\big[\Delta\,\sigma(h_k)\big].$$

All expectations are over $(h, h^*) \sim \mathcal{N}(0,\Sigma)$ with $\Sigma$ from the current $(Q,M,T)$; they can be computed numerically, or analytically when $\sigma = \operatorname{erf}$. The Itô correction in $dQ_{ij}/d\alpha$ arises from the discrete-to-continuous passage: $\|x^\mu\|_2^2/N \to 1$ by the law of large numbers for Gaussian vectors, giving an $\mathcal{O}(\eta^2)$ term with no analogue in continuous-time gradient flow. This is the stochastic analogue of the stability correction in an explicit time-stepping scheme.

Rigorous proofs of convergence to this ODE system are in {cite}`goldt2019dynamics; veiga2022phase; benarous2021online; benarous2022online`.

# Plateau Dynamics and the Specialisation Transition

## The Soft Committee Machine

The ODEs become explicit for the **soft committee machine** {cite}`saad1995online`: fixed readout weights $v_k = 1/K$, equal-width teacher $K = K^*$, erf activation, and isotropic teacher $T = I$. Small initialisation $w_k^{(0)} \sim \mathcal{N}(0,\rho^2)$ with $\rho \ll 1$.

**Goal:** each student unit should specialise to one teacher direction, $w_k \to w_r^*$ for some permutation.

By the permutation symmetry of the initialisation, the dynamics remain on an **invariant manifold** of the ODE:

$$M_{kr} = R\,\delta_{kr} + S\,(1-\delta_{kr}), \qquad Q_{kk'} = Q\,\delta_{kk'} + C\,(1-\delta_{kk'}),$$

with $Q = (MM^\top)_{kk}$. Only two scalar unknowns remain: $R$ (same-index student–teacher alignment) and $S$ (cross-index alignment).

**Symmetric fixed point.** The reduced ODE system has a fixed point at

$$R^* = S^* = \frac{1}{\sqrt{K(2K-1)}}, \qquad Q^* = C^* = \frac{1}{2K-1},$$

where all student units overlap equally with all teacher directions — no specialisation. The generalisation error at this *plateau* is

$$\mathcal{E}_g^\text{plateau} = \frac{K}{6} - \frac{K^2}{\pi}\arcsin\!\left(\frac{1}{2K}\right),$$

which *increases* with $K$: wider networks have worse plateaus because they have more symmetry to break. The student is stuck in a saddle point of the loss landscape.

## Escape from the Plateau: Linear Stability Analysis

Linearise the ODE around $(R^*, S^*)$:

$$\frac{d}{d\tilde\alpha}\begin{pmatrix}\delta R\\\delta S\end{pmatrix} = J\begin{pmatrix}\delta R\\\delta S\end{pmatrix},$$

where $J$ is the Jacobian evaluated at the fixed point. The permutation symmetry of the problem block-diagonalises $J$ into two one-dimensional modes:

- **Symmetric mode** $(\delta R, \delta S) \propto (1,1)$: eigenvalue $\lambda_\text{sym} < 0$ — stable (perturbations along the manifold decay).
- **Specialisation mode** $(\delta R, \delta S) \propto (K-1,-1)$: eigenvalue $\lambda_\text{spec} > 0$ — **unstable** (the plateau is a saddle point). For $K \gg 1$: $\lambda_\text{spec} \approx \frac{1}{2\pi K}$.

This is exactly a standard linear stability analysis for a fixed point of an ODE — familiar from bifurcation theory. The specialisation mode is the unstable manifold direction.

**Escape time.** Since $w_k^{(0)} \sim \mathcal{N}(0,\rho^2)$ with $\rho \ll 1$, the specialisation amplitude at initialisation is $a(0) \sim \rho/\sqrt{N} \sim 1/\sqrt{N}$. It grows as $a(\tilde\alpha) = a(0)\,e^{\lambda_\text{spec}\tilde\alpha}$. Escaping when $a \sim \mathcal{O}(1)$:

$$\tilde\alpha_\text{esc} \approx \frac{\ln N}{2\lambda_\text{spec}} \underset{K\gg 1}{\approx} \pi K\ln N.$$

The plateau duration is **logarithmic in $N$** (fine) and **linear in $K$** (a genuine bottleneck for wide networks). This gives a precise quantitative prediction for when training suddenly accelerates — the hallmark of the specialisation transition.

## Graded Teacher and Cascade of Specialisations

For a teacher with non-isotropic self-overlap $T = \operatorname{diag}(t_1, \ldots, t_K)$, $t_1 > \cdots > t_K$, the specialisation modes become unstable at different rates. The result is a **cascade** of specialisations $\tilde\alpha_\text{esc}^{(1)} < \cdots < \tilde\alpha_\text{esc}^{(K)}$: the student learns teacher directions one at a time, strongest first, each appearing as a step-down in $\mathcal{E}_\text{gen}(\tilde\alpha)$. This is the ODE-level mechanism behind the empirical observation of "grokking" and staged learning in modern networks.

# Optimal Learning Rate Schedules

*(Reference: {cite}`saad1997online`)*

Since the ODE for $\Omega = (M, Q, v)$ is exact as $N \to \infty$, one can optimise the schedule $\eta(\alpha)$ to minimise $\mathcal{E}_g(\Omega(\alpha_F))$ at a final time $\alpha_F$. This is a standard **optimal control** problem — exactly as one minimises a cost functional in PDE-constrained optimisation.

## Locally Optimal Rate

At each time, choose $\eta$ to maximise the instantaneous rate of decrease of $\mathcal{E}_g$:

$$\frac{d\mathcal{E}_g}{d\alpha} = \eta\underbrace{(\nabla_\Omega\mathcal{E}_g\cdot F_1)}_{\leq 0\text{ (descent)}} + \eta^2\underbrace{(\nabla_\Omega\mathcal{E}_g\cdot F_2)}_{> 0\text{ (noise inflation)}},$$

where $F_1$ is the drift and $F_2$ is the noise covariance contribution. Setting $d(d\mathcal{E}_g/d\alpha)/d\eta = 0$:

$$\eta^*_\text{loc}(\alpha) = -\frac{\nabla_\Omega\mathcal{E}_g\cdot F_1}{2\,\nabla_\Omega\mathcal{E}_g\cdot F_2}.$$

This is the steepest instantaneous descent on a signal-to-noise ratio. However, it **stalls at the plateau**: the locally optimal $\eta$ cannot break the symmetry because $F_1 = 0$ there.

## Globally Optimal Rate via Pontryagin's Principle

Minimise $\mathcal{E}_g(\Omega(\alpha_F))$ subject to the ODE constraint $d\Omega/d\alpha = \eta F_1(\Omega) + \eta^2 F_2(\Omega)$. Introduce a costate (adjoint) variable $\Lambda(\alpha)$ and form the Hamiltonian:

$$\mathcal{H}_\text{ctrl} = \Lambda \cdot \big(\eta F_1(\Omega) + \eta^2 F_2(\Omega)\big).$$

**Pontryagin's Minimum Principle** gives:
- **State ODE:** $d\Omega/d\alpha = \eta F_1 + \eta^2 F_2$ (forward in $\alpha$).
- **Costate ODE:** $-d\Lambda/d\alpha = (\eta \nabla_\Omega F_1 + \eta^2 \nabla_\Omega F_2)^\top \Lambda$, with final condition $\Lambda(\alpha_F) = \nabla_\Omega\mathcal{E}_g(\Omega(\alpha_F))$ (backward in $\alpha$).
- **Optimality condition:** minimise $\mathcal{H}_\text{ctrl}$ over $\eta$ at each $\alpha$:

$$\eta^*_\text{glob}(\alpha) = -\frac{\Lambda(\alpha)\cdot F_1(\Omega)}{2\,\Lambda(\alpha)\cdot F_2(\Omega)}.$$

The costate $\Lambda(\alpha)$ encodes the *sensitivity of the final error* to the current state, propagated backward in time. This is the adjoint equation familiar from PDE-constrained optimisation. At the plateau, $\Lambda$ can be nonzero even when $F_1 = 0$ (because the final error is sensitive to the current state even when the state is not moving), allowing $\eta^*_\text{glob}$ to push the learning rate high enough to escape the symmetric fixed point. The global rate can therefore outperform the local rate near the plateau — at the cost of temporarily increasing $\mathcal{E}_g$ to set up a faster descent later {cite}`saad1997online`.

# Multi-Pass SGD via Dynamical Mean-Field Theory

*(Day 4)*

Online learning assumes each data point is seen exactly once. Multi-pass SGD reuses data across epochs, introducing **temporal memory**: the weight $w(t)$ depends on its own past through earlier gradients evaluated on the same data points.

## The Model

Consider a single-layer model with binary Gaussian mixture data:

$$y^\mu = \pm1, \qquad x^\mu = \frac{y^\mu w^*}{\sqrt{N}} + z^\mu, \quad z^\mu \sim \mathcal{N}(0,I_N).$$

Train with squared loss and $\ell_2$ regularisation. **Multi-pass SGD** with subsampling schedule $s_\mu(t) \in \{0,1\}$ (indicating whether sample $\mu$ is in the current mini-batch):

$$\frac{dw}{dt} = -\frac{1}{b}\sum_{\mu=1}^p s_\mu(t)\,\ell'(y^\mu h^\mu(t))\,x^\mu - \lambda w(t),$$

where $b = \mathbb{E}[s_\mu(t)]$ is the expected batch fraction and $\tau = dt/b$ is the persistence time. As $P, N \to \infty$ with $\alpha = P/N = \mathcal{O}(1)$, the temporal correlations between steps (absent in online learning) must be tracked explicitly.

## The Dynamical Cavity Method: Three Contributions

Decompose the weight vector along three directions:

1. **Lab frame** $v_i$: a fixed unit direction uncorrelated with data and teacher, $w_i(t) = w(t) \cdot v_i$.
2. **Data directions** $x^\mu / \sqrt{N}$: projections onto individual data points, $h^\mu(t) = x^\mu \cdot w(t)/\sqrt{N}$.
3. **Teacher direction** $w^*/N$: $m(t) = w(t)\cdot w^*/N$.

Adding the $v_i$ direction as a perturbation and expanding to first order in $1/\sqrt{N}$, one finds that the weight component $w_i(t)$ along $v_i$ satisfies an **effective stochastic process** with three terms {cite}`mignacco2020dynamical; bordelon2022self`:

$$\frac{dw_i(t)}{dt} = \underbrace{F_i(t)}_{\text{① random force}} - \underbrace{\alpha\int_0^t M_R(t,t')\,w_i(t')\,dt'}_{\text{② retarded friction}} - \underbrace{\big(\lambda + \alpha\,\upsilon(t)\big)\,w_i(t)}_{\text{③ dynamic renorm.}}$$

**① Random force** $F_i(t)$: the contribution of sample $\mu$'s input component along $v_i$, summed over all active samples. By the CLT, it is Gaussian with zero mean and two-time correlation

$$\langle F_i(t)F_i(t')\rangle = \alpha\,M_C(t,t'),$$

where $M_C(t,t') = \frac{1}{b}\langle s(t)s(t')\,\ell'(h(t))\,\ell'(h(t'))\rangle$ is the **correlation kernel**.

**② Retarded friction**: the past trajectory $w_i(t')$ for $t' < t$ feeds back through a **memory kernel** $M_R(t,t')$,

$$M_R(t,t') = \frac{1}{b}\left\langle s(t)\,\frac{\delta\ell'(h(t))}{\delta p(t')}\right\rangle,$$

encoding how the gradient at time $t$ is influenced by the weight perturbation at the earlier time $t'$. This is the Onsager response function of the dynamics, and it is the key new ingredient absent from online learning. The convolution $\int_0^t M_R(t,t') w_i(t')\,dt'$ is a **Volterra integral** of the first kind — exactly the structure of memory-dependent (history-dependent) ODEs / viscoelastic constitutive laws. In online learning, $M_R \equiv 0$ because past gradients were computed on different (independent) data; in multi-pass SGD they were computed on the same data, creating a nontrivial $M_R$.

**③ Dynamic renormalisation** of the regularisation: $\upsilon(t) = \frac{1}{b}\langle s(t)\,\ell''(h(t))\rangle$ accounts for the curvature of the loss.

## DMFT Closure and Self-Consistency

The effective process for $w_i(t)$ depends on $M_C$ and $M_R$, which in turn depend on the distribution of $h^\mu(t)$ — which depends on $w_i(t)$. Self-consistently closing this loop gives the **dynamical mean-field theory (DMFT)** equations: a closed system of integro-differential equations for the two-time correlation functions and response functions of the effective process. This is the dynamical analogue of the self-consistency equations for the replica order parameters in Lecture 1.

The DMFT equations are exact as $N \to \infty$. They capture:
- **Multi-pass memory**: the Volterra kernel $M_R(t,t')$ encodes how earlier training steps affect the current gradient.
- **Batch-size effects**: the persistence time $\tau = dt/b$ interpolates between full-batch GD ($\tau \to \infty$, $M_R$ long-lived) and online SGD ($\tau \to 0$, $M_R \to 0$).
- **Phase diagrams**: overfitting vs.\ underfitting as a function of $\alpha$, $\lambda$, $b$, and $\tau$ {cite}`mignacco2020dynamical; bordelon2022self`.

## Scaling Limits

The crossover between three qualitatively different regimes is controlled by $K \sim N^\kappa$ and $\eta \sim N^{-\delta}$ {cite}`veiga2022phase`:

| Regime | Condition | Outcome |
|--------|-----------|---------|
| Perfect learning | $\kappa + \delta > 0$ | Population error $\to 0$ |
| Bad learning | $-\tfrac{1}{2} < \kappa+\delta < 0$ | Noise dominates, nonzero error |
| No convergence | $\kappa + \delta < -\tfrac{1}{2}$ | ODE description breaks down |

The classical online-learning limit of these notes ($\eta = \mathcal{O}(1)$, $K = \mathcal{O}(1)$, $N\to\infty$) sits at $\kappa = \delta = 0$, on the boundary of the perfect-learning regime. The full phase diagram in the $(\kappa,\delta)$ plane is in {cite}`veiga2022phase`.
