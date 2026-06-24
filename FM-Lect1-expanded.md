---
title: Statistical Physics Approaches to High-Dimensional Learning
subtitle: "ICTP Summer School on Machine Learning — Lecture 1"
short_title: Motivation and Background
authors:
  - name: Francesca Mignacco (Princeton University & CUNY Graduate Center), transcribed by Max Hirsch with the Claude LLM
subject: Lecture Notes
venue: ICTP-INdAM-SLMath Summer Graduate School for Machine Learning, Trieste, 15–19 June 2026
bibliography: references.bib
---

These notes cover the first lecture. The central idea is that a trained neural network is a high-dimensional interacting system and the tools statistical physicists developed — disorder averaging, free energies, saddle-point methods, and the replica trick — give exact, quantitative predictions about learning and generalization. The analysis of the binary Gaussian mixture classification model follows {cite}`mignacco2020role`.

**A note on perspective.** Readers with a PDE background will find many familiar structures here, but with a probabilistic flavour. The "order parameters" that characterise the stationary states of the learning problem are analogous to macroscopic quantities (density, temperature) that collapse a high-dimensional PDE system to a low-dimensional description. The "thermodynamic limit" ($N \to \infty$) plays the role of a continuum limit. The replica method is a device for computing disorder-averaged logarithms, analogous in spirit to the method of characteristics but for a very different class of equations.

# Motivation: Why Physics Methods for Machine Learning?

Modern ML systems achieve striking results: large language models, protein structure prediction (AlphaFold, 2024 Nobel Prize in Chemistry), image generation, mathematical reasoning. Training-compute has grown exponentially, from ADALINE (1950s) to GPT-class models ($10^{25}$ FLOPs). Yet theoretical foundations remain poorly understood. Leo Breiman's 1995 essay "Reflections after refereeing papers for NIPS" listed open questions that remain largely unresolved:

- Why don't heavily over-parameterised neural networks overfit?
- What is the effective number of parameters?
- Why doesn't gradient descent get stuck in poor local minima?
- When should one stop training?

The first two concern the **statics** of learning (what the trained model looks like); the last two concern the **dynamics** (how it gets there). These two goals organise the entire course.

The **Hopfield model** {cite}`hopfield1982neural` is the founding example of the physics–ML connection. It is an associative memory: each stored pattern is an energy minimum of a disordered spin system, and retrieval is simply gradient descent relaxing into the nearest basin. Hopfield and Hinton shared the 2024 Nobel Prize in Physics; Parisi received the 2021 Nobel Prize for the replica theory that underpins it {cite}`mezard1987spin`. The capacity phase diagram — how many patterns can be stored before retrieval fails — was computed analytically by Amit, Gutfreund, and Sompolinsky {cite}`amit1985spin` using the replica method.

# Three Ingredients of Machine Learning

A supervised learning problem is defined by a dataset $\mathcal{D} = \{(x^\mu, y^\mu)\}_{\mu=1}^{p}$ with inputs $x^\mu \in \mathbb{R}^N$ and labels $y^\mu \in \mathbb{R}$. The goal is to learn a mapping $x \mapsto y$ that generalises to unseen data.

**① Data.** The training set $\mathcal{D}$. In the statistical physics approach, data are random samples from some distribution $P_{XY}$, which enables disorder averaging (see below). Concrete benchmarks: MNIST ($28\times28$ greyscale images, 70 k examples), CIFAR-10 ($32\times32$ RGB, 60 k), ImageNet (14 M images, 20 k categories), LAION-5B (5.8 B image–text pairs).

**② Architecture.** The function class $f_\mathbf{w}$, mapping inputs $x$ to predictions $\hat{y} = f_\mathbf{w}(x)$.

*The perceptron* (McCulloch–Pitts 1943; Rosenblatt 1958) is the simplest case: a single artificial neuron

$$\hat{y} = \phi(\mathbf{w}^\top x + b),$$

where $\mathbf{w}\in\mathbb{R}^N$ is a weight vector, $b$ is a bias, and $\phi$ is a nonlinearity. With $\phi = \operatorname{sign}$ it defines a linear classifier: the decision boundary is the hyperplane $\{\mathbf{w}^\top x + b = 0\}$ in input space, partitioning $\mathbb{R}^N$ into two half-spaces. The perceptron can only solve **linearly separable** tasks; Minsky & Papert's 1969 book showed it cannot solve XOR, contributing to the first AI winter.

*Deep neural networks* {cite}`lecun2015deep` compose many layers of such transforms:

$$\hat{y} = f_\mathbf{W}(x) = \phi^{(L)}\!\left(W_{(L)}\,\phi^{(L-1)}\!\left(\cdots\phi^{(1)}(W_{(1)} x)\cdots\right)\right).$$

Each layer applies an affine map followed by a pointwise nonlinearity $\phi$ (e.g.\ ReLU, $\tanh$, erf). The weight matrices $W_{(\ell)}$ are the parameters to be learned. This is a highly nonlinear composition: viewed as a discrete-time nonlinear map, each layer is one step of a flow.

*Transformers* {cite}`vaswani2017attention` process sequential data (language, code, proteins) via the **attention mechanism**:

$$\operatorname{Attention}(Q,K,V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V,$$

where $Q$ (queries), $K$ (keys), $V$ (values) are learned linear projections of the input tokens, and the softmax makes the row sums equal 1. Intuitively, each token attends to all others with weights proportional to query–key similarity. The training objective is next-token prediction.

**③ Algorithm.** Gradient descent on a regularised empirical loss:

$$\mathcal{L}(\mathbf{w};\mathcal{D}) = \sum_{\mu=1}^{p} \ell(y^\mu;\mathbf{w}, x^\mu) + \frac{\lambda}{2}\|\mathbf{w}\|_2^2, \qquad \mathbf{w}^{t+1} = \mathbf{w}^{t} - \eta\,\nabla_{\mathbf{w}}\mathcal{L}(\mathbf{w}^{t}).$$

For deep networks, $\nabla_\mathbf{w}\mathcal{L}$ requires **backpropagation** {cite}`rumelhart1986learning` — the chain rule applied layer-by-layer. With $a^{(0)} = x$, $z^{(\ell)} = W^{(\ell)}a^{(\ell-1)} + b^{(\ell)}$, $a^{(\ell)} = \phi(z^{(\ell)})$, $\hat{y} = a^{(L)}$, the backward pass computes error signals layer-by-layer:

$$\delta^{(L)} = \nabla_{z^{(L)}}\ell(y, a^{(L)}), \qquad \delta^{(\ell)} = \left(W^{(\ell+1)}\right)^\top\!\delta^{(\ell+1)}\odot\phi'(z^{(\ell)}), \qquad \frac{\partial\mathcal{L}}{\partial W^{(\ell)}} = \delta^{(\ell)}\!\left(a^{(\ell-1)}\right)^\top.$$

Zdeborová {cite}`zdeborova2020understanding` emphasises that building a theory of deep learning requires understanding the interplay of all three ingredients simultaneously.

# The Generalization Problem

A predictor is useful only if it performs well on **new** data, not just on the training set. The two relevant error measures are:

$$\varepsilon_\text{train}(\hat{\mathbf{w}}) = \frac{1}{p}\sum_{\mu=1}^{p}\ell\!\left(y_\mu, f_{\hat{\mathbf{w}}}(x_\mu)\right), \qquad \varepsilon_\text{gen}(\hat{\mathbf{w}}) = \mathbb{E}_{(x,y)\sim P_{XY}}\!\left[\ell\!\left(y, f_{\hat{\mathbf{w}}}(x)\right)\right].$$

The generalisation error $\varepsilon_\text{gen}$ is **unknown in practice** (we do not have access to $P_{XY}$, only to a finite sample from it). The statistical physics approach circumvents this: by working with an explicit generative model, it allows $\varepsilon_\text{gen}$ to be computed *exactly* in the thermodynamic limit, giving sharp predictions that can be checked against simulations.

# Exactly Solvable Models: The Statistical Physics Programme

The statistical physics approach rests on three pillars.

## 1. The Teacher–Student Setup

Rather than working with a black-box dataset, assume that labels are generated by a known *teacher* network with fixed weights $\mathbf{w}^*$:

$$y^\mu = f_{\mathbf{w}^*}(x^\mu) + \text{noise}, \qquad x^\mu \sim \mathcal{N}(0, I_N).$$

A *student* network with learnable $\mathbf{w}$ is trained to recover $\mathbf{w}^*$. This is analogous to solving an inverse problem when the forward operator is known. The teacher–student framework and its phase diagrams are described in {cite}`zdeborova2020understanding; engel2001statistical`.

## 2. Average-Case Rather Than Worst-Case Analysis

Classical learning theory (VC dimension, PAC learning, Rademacher complexity) provides **worst-case** bounds: guarantees that hold for every possible dataset. These bounds scale as $\mathcal{O}(\sqrt{N/p})$ and are often vacuous for modern large models.

The statistical physics approach instead asks: what happens for a **typical** dataset drawn from $P_{XY}$? In the large-$N$ limit, the distribution of $\varepsilon_\text{gen}$ over datasets concentrates into a sharp spike (self-averaging; see below), so the typical and mean values coincide. This gives **exact** predictions rather than inequalities.

## 3. The Thermodynamic Limit

Take $N, p \to\infty$ with $\alpha = p/N = \mathcal{O}(1)$ fixed. This is the analogue of a continuum limit in PDE numerics: the discrete, high-dimensional problem converges to a continuous, low-dimensional description in terms of a handful of **order parameters** — scalar quantities that capture the macroscopic state of the system (analogous to, say, the Reynolds number in fluid mechanics). Gardner's 1987–88 papers {cite}`gardner1987maximum; gardner1988space` are the founding works of this approach to learning.

The two goals of the course:
① **Statics:** which $\mathbf{w}$ does gradient descent converge to, and what is its $\varepsilon_\text{gen}$?
② **Dynamics:** how does the trajectory $\mathbf{w}^t$ evolve, and can we describe it with ODEs?

# Supervised Learning: Bayesian Formulation

Given $(x^\mu, y^\mu)\overset{\text{iid}}{\sim}P_{XY}(\cdot\mid\mathbf{w}^*)$, Bayes' theorem gives the **posterior distribution** over weights:

$$P(\mathbf{w}\mid\mathcal{D}) = \frac{P_0(\mathbf{w})\prod_\mu P_y(y^\mu\mid\mathbf{w},x^\mu)}{\mathcal{Z}(\mathcal{D})}, \qquad \mathcal{Z}(\mathcal{D}) = \int d\mathbf{w}\;P_0(\mathbf{w})\prod_\mu P_y(y^\mu\mid\mathbf{w},x^\mu).$$

Here $P_0(\mathbf{w})$ is the prior and $\mathcal{Z}(\mathcal{D})$ is the normalisation constant (partition function in physics language).

Identifying $-\log P_y \propto \ell$ and $-\log P_0 \propto \frac\lambda2\|\mathbf{w}\|^2$ gives a Gibbs distribution at inverse temperature $\beta$:

$$P(\mathbf{w}\mid\mathcal{D}) = \frac{e^{-\beta\,\mathcal{L}(\mathbf{w};\mathcal{D})}}{\mathcal{Z}(\mathcal{D})}.$$

**The fundamental analogy:** loss $\leftrightarrow$ energy, $\beta$ $\leftrightarrow$ inverse temperature, $\mathbf{w}$ $\leftrightarrow$ spin configuration, $\mathcal{D}$ $\leftrightarrow$ quenched disorder (frozen random couplings). The partition function $\mathcal{Z}(\mathcal{D})$ plays the role of the normalising constant for a Boltzmann distribution.

- **ERM limit ($\beta\to\infty$):** the Gibbs measure concentrates on the loss minimiser; this is what gradient descent implements.
- **Bayes-optimal setting:** if $P_0 = P_{\mathbf{w}^*}$ and $P_y = P_{Y|X}$ exactly match the true generative process, the posterior mean achieves the minimum possible $\varepsilon_\text{gen}$ among all algorithms (the MMSE estimator).

## Free Energy Density and Self-Averaging

The **free energy density** encodes all macroscopic information about the system:

$$f_N(\beta) = -\frac{1}{\beta N}\log\mathcal{Z}(\mathcal{D}).$$

**Self-averaging** is the key property that makes this tractable: in the limit $N\to\infty$,

$$f_N(\beta,\mathcal{D}) \xrightarrow{a.s.} f(\beta) = \lim_{N\to\infty}\mathbb{E}_\mathcal{D}[f_N(\beta)].$$

This is analogous to the law of large numbers, but for a nonlinear functional of the data. Self-averaging allows us to replace the hard problem (compute $f_N$ for a specific, random $\mathcal{D}$) with the easier problem (compute $\mathbb{E}_\mathcal{D}[f_N]$). Once $f(\beta)$ is known, all macroscopic observables follow by differentiation — exactly as in thermodynamics.

**Quenched vs annealed.** Self-averaging tells us we want $\mathbb{E}_\mathcal{D}[\log\mathcal{Z}]$ (the **quenched** average, where $\mathcal{D}$ is fixed before taking the log). The **annealed** average $\log\mathbb{E}_\mathcal{D}[\mathcal{Z}]$ is easier (Jensen: just exchange log and expectation) but is only an upper bound. Computing the quenched average exactly requires the replica method.

# Binary Gaussian Mixture Classification

We study the simplest nontrivial exactly solvable model {cite}`mignacco2020role`. Data come from a **binary Gaussian mixture**:

$$y^\mu = \pm1 \text{ with equal probability}, \qquad x^\mu = \frac{y^\mu\,\mathbf{w}^*}{\sqrt{N}} + \sqrt{\Delta}\,z^\mu, \qquad z^\mu\sim\mathcal{N}(0,I_N).$$

The two classes form isotropic Gaussian clouds of radius $\sqrt{\Delta}$ centred at $\pm\mathbf{w}^*/\sqrt{N}$; the signal-to-noise ratio is $\|\mathbf{w}^*\|^2/(\Delta N)$. The student uses a perceptron with threshold $\kappa$:

$$\hat{y}_\mathbf{w}(x) = \operatorname{sign}(h+\kappa), \qquad h = \frac{\mathbf{w}\cdot x}{\sqrt{N}}.$$

The pre-activation $h$ is a scalar summary of the input: it is the projection of $x$ onto the weight vector, normalised to $\mathcal{O}(1)$.

The limiting case $\mathbf{w}^*=0$ (pure noise labels) is the **Gardner capacity** problem: $p$ random $\pm1$ labels can be shattered by a linear classifier in $\mathbb{R}^N$ if and only if $\alpha < \alpha_c = 2$ {cite}`gardner1988space`. This is a sharp phase transition — entirely analogous to a subcritical/supercritical bifurcation in PDEs.

# The Replica Method

We need $\mathbb{E}_\mathcal{D}[\log\mathcal{Z}(\mathcal{D})]$. The **replica trick** {cite}`mezard1987spin; nishimori2001statistical; engel2001statistical` is an elegant algebraic device:

$$\log Z = \lim_{n\to 0} \frac{Z^n - 1}{n} = \lim_{n\to 0}\frac{\partial}{\partial n}\log Z^n \qquad \Longrightarrow \qquad \mathbb{E}[\log\mathcal{Z}] = \lim_{n\to 0}\frac{\partial}{\partial n}\mathbb{E}[\mathcal{Z}^n].$$

For integer $n$, $\mathcal{Z}^n$ is the partition function of $n$ *independent copies* (replicas) $\{\mathbf{w}_a\}_{a=1}^n$ of the system, all coupled through the same quenched disorder $\mathcal{D}$. One averages over $\mathcal{D}$ first (this couples the replicas to each other), then analytically continues to $n\to 0$.

**Step 1: Gaussian integration over data.** The local fields $h_a^\mu = \mathbf{w}_a\cdot x^\mu/\sqrt{N}$ are jointly Gaussian. Averaging the replicated partition function over the $p$ i.i.d. samples $\{x^\mu, y^\mu\}$ couples replicas through two macroscopic **order parameters**:

$$m_a = \frac{\mathbf{w}_a\cdot\mathbf{w}^*}{N} \quad \text{(teacher–student alignment)}, \qquad Q_{ab} = \frac{\mathbf{w}_a\cdot\mathbf{w}_b}{N} \quad \text{(replica overlap matrix)}.$$

These are scalar summaries: $m_a$ measures how well replica $a$ has recovered the teacher direction, and $Q_{ab}$ measures how similar two replica weight vectors are.

**Step 2: Saddle-point.** In large $N$ the integral over $(Q_{ab}, m_a)$ and conjugate variables is dominated by a saddle point, giving an effective free energy:

$$S = \underbrace{\text{(weight geometry / prior)}}_{\text{entropy term}} - \underbrace{\alpha\log\mathcal{Z}_y(Q,\mathbf{m})}_{\text{data / energy term}}.$$

The data term $\mathcal{Z}_y$ involves a one-dimensional Gaussian integral over a single effective field — all the $N$-dimensional complexity has been collapsed to a scalar problem.

**Step 3: Analytic continuation.** Extremise $S$ over $(Q_{ab}, m_a)$ and continue to $n\to 0$.

## Replica-Symmetric Ansatz

The simplest consistent ansatz is **replica symmetry (RS)**: all replicas are equivalent, so $Q_{ab} = r\,\delta_{ab} + q\,(1-\delta_{ab})$ and $m_a = m$. This reduces to three scalar order parameters:

| Symbol | Meaning |
|--------|---------|
| $r = \|\mathbf{w}_a\|^2/N$ | Mean squared norm of the weight vector |
| $q = \mathbf{w}_a\cdot\mathbf{w}_b/N$, $a\neq b$ | Overlap between two independent posterior samples |
| $m = \mathbf{w}_a\cdot\mathbf{w}^*/N$ | Teacher–student alignment (controls generalisation) |

RS is **exact for convex losses** (ridge regression, logistic regression) because convex losses have a unique global minimum, so all replicas converge to the same point ($q=r$). For non-convex losses (deep networks), RS may break down — a sign of a more complex loss landscape with many local minima.

Extremising the RS free energy gives three coupled algebraic self-consistency equations in $(r, q, m)$ — three equations in three unknowns.

**ERM limit ($\beta\to\infty$).** The posterior concentrates on the minimiser; $r - q \to 0$ but the susceptibility $\chi = \beta(r-q) = \mathcal{O}(1)$ remains finite (it measures the linear response of the order parameters to perturbations).

# Generalisation Error

Once $(r,q,m)$ are found from the saddle-point equations, the generalisation error follows in closed form. The pre-activation of the student on a test point is Gaussian with mean and variance set by the order parameters:

$$\tilde{h}\,|\,\tilde{y} \sim \mathcal{N}\!\left(\tilde{y}\,m,\; \Delta r\right),$$

so the generalisation error (fraction of test points misclassified, with bias $\kappa$) is:

$$\mathcal{E}_\text{gen} = \rho\,\Phi\!\left(\frac{-m-\kappa}{\sqrt{\Delta r}}\right) + (1-\rho)\,\Phi\!\left(\frac{m-\kappa}{\sqrt{\Delta r}}\right),$$

where $\Phi(t) = \frac{1}{\sqrt{2\pi}}\int_{-\infty}^t e^{-s^2/2}\,ds$ is the standard Gaussian CDF.

For the balanced ($\rho=\tfrac12$), unbiased ($\kappa=0$) case: $\mathcal{E}_\text{gen} = \Phi(-m/\sqrt{\Delta r})$, which depends only on the **signal-to-noise ratio** $m/\sqrt{\Delta r}$ of the student's pre-activation. As $\alpha\to\infty$ (more data), $m\to\sqrt{1/\Delta}$ and $\varepsilon_\text{gen}\to 0$. The full phase diagram as a function of $(\alpha,\Delta,\lambda)$ is in {cite}`mignacco2020role`.

**Key insight.** The entire $N$-dimensional learning problem has been reduced to a *three-parameter scalar system* whose solution gives exact predictions for $\varepsilon_\text{gen}$. This is the power of the thermodynamic limit combined with the RS ansatz.
