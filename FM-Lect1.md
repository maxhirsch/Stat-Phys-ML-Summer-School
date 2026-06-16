---
title: Statistical Physics Approaches to High-Dimensional Learning
subtitle: "ICTP Summer School on Machine Learning — Lecture 1"
short_title: Statistical Physics of Learning
authors:
  - name: Francesca Mignacco
subject: Lecture Notes
---

These notes cover the first lecture by Francesca Mignacco at the ICTP Summer School on Machine Learning. The lecture introduces the statistical physics perspective on supervised learning, reviews the Bayesian formalism and its connection to Gibbs distributions, and develops the replica method for the binary Gaussian mixture classification problem. The analysis is based in part on @mignacco2020role.

# Three Ingredients of Machine Learning

A supervised learning problem is defined by a dataset

$$
\mathcal{D} = \{(x^\mu, y^\mu)\}_{\mu=1}^{p}, \qquad x^\mu \in \mathbb{R}^N,\; y^\mu \in \mathbb{R},
$$

and the **goal** is to learn the mapping $x \mapsto y$, so that for a new input $x_\text{new}$ we can predict

$$
\hat{y}_\text{new} = f_{\mathbf{w}}(x_\text{new}).
$$

Three ingredients are needed:

**① Data.** The training set $\mathcal{D}$.

**② Architecture.** The function class $f_{\mathbf{w}}$.

- *Perceptron:* $\hat{y} = f_{\mathbf{w}}(x) = \phi(\mathbf{w}^\top x)$, e.g. $\phi(\cdot) = \operatorname{sign}(\cdot + \kappa)$. This can only solve linearly separable tasks.
- *Deep neural network (DNN):* $\phi^{(L)}\!\left(W_{(L)}\,\phi^{(L-1)}(\cdots)\right)$.
- *Transformers.*

**③ Algorithm.** Gradient descent on a regularized loss:

$$
\mathcal{L}(\mathbf{w};\mathcal{D}) = \sum_{\mu=1}^{p} \ell(y^\mu;\mathbf{w}, x^\mu) + \frac{\lambda}{2}\|\mathbf{w}\|_2^2,
$$

with update rule $\mathbf{w}^{t+1} = \mathbf{w}^{t} - \eta\,\nabla_{\mathbf{w}}\mathcal{L}(\mathbf{w}^{t})$, via backpropagation.

# The Generalization Problem

The central challenge is that minimizing the **training error** $\mathcal{E}_\text{train}$ does not automatically minimize the **generalization error** $\mathcal{E}_\text{gen}$. Understanding this gap is the core problem.

**Related model.** The *Hopfield model* [@hopfield1982neural] interprets memory retrieval as relaxation in an energy landscape — an early instance of the physics–ML connection.

# Exactly Solvable Models

To make analytical progress one adopts the **teacher–student** framework:

- Assume $(x, y) \sim P_{X,Y}$ with $x \sim \mathcal{N}(0, I)$.
- A **teacher network** with true weights $\mathbf{w}^*$ generates labels $y$ (the ground truth).
- A **student network** with weights $\mathbf{w}$ produces predictions $\hat{y} = f_{\mathbf{w}}(x)$.

The goal is to understand how well the student can recover the teacher as a function of data, architecture, and algorithm.

The statistical physics approach involves two key ideas:

1. **Average-case (vs. worst-case) analysis** — averaging over the random data $\mathcal{D}$.
2. The **thermodynamic (infinite-dimensional) limit** $N, p \to \infty$ with $\alpha = p/N = \mathcal{O}(1)$.

This leads to two complementary goals: understand (i) **statics** (properties of the loss landscape and its minimizers) and (ii) **dynamics** (behavior of gradient descent).

# Supervised Learning: Bayesian Formulation

Given a training set $\mathcal{D} = \{(x^\mu, y^\mu)\}_{\mu=1}^p$ with $(x^\mu, y^\mu) \overset{\text{iid}}{\sim} P_{XY}(\cdot \mid \mathbf{w}^*)$ and $\mathbf{w}^* \sim P_{\mathbf{w}^*}$, the network output is $\hat{y}(x) = f_{\mathbf{w}}(x)$, and we use $\mathcal{D}$ to learn $\mathbf{w}$.

## Bayesian Learning

The posterior over weights is

$$
P(\mathbf{w} \mid \mathcal{D}) = \frac{P_0(\mathbf{w})\prod_{\mu=1}^{p} P_y(y^\mu \mid \mathbf{w}, x^\mu)}{\mathcal{Z}(\mathcal{D})},
$$

where $P_0$ is the prior, $P_y$ the likelihood, and $\mathcal{Z}(\mathcal{D})$ is the partition function. Identifying $-\log P_y(y^\mu \mid \mathbf{w}, x^\mu) \propto \ell(y^\mu; \mathbf{w}, x^\mu)$ and $-\log P_0(\mathbf{w}) \propto \frac{\lambda}{2}\|\mathbf{w}\|^2$, this becomes the **Gibbs distribution**

$$
P(\mathbf{w} \mid \mathcal{D}) = \frac{e^{-\beta\,\mathcal{L}(\mathbf{w};\mathcal{D})}}{\int d\mathbf{w}\;e^{-\beta\,\mathcal{L}(\mathbf{w};\mathcal{D})}},
$$

where $\beta > 0$ is the inverse temperature.

## Special Cases

**(i) Empirical Risk Minimization (ERM).** In the zero-temperature limit $\beta \to \infty$, the Gibbs measure concentrates on the minimizer:

$$
-\frac{1}{\beta}\log\mathcal{Z}(\mathcal{D}) \xrightarrow{\beta\to\infty} \min_{\mathbf{w}}\mathcal{L}(\mathbf{w};\mathcal{D}).
$$

A common choice is logistic loss: $\ell(y;\mathbf{w},x) = \log(1 + e^{-y\,\mathbf{w}^\top x})$.

**(ii) Bayes-Optimal (BO) Setting.** Set $P_0 = P_{\mathbf{w}^*}$ and $P_y = P_{Y|X}$ so that the prior and likelihood match the true generative model. One then has

$$
\mathcal{E}_\text{gen}(\mathrm{BO}) \leq \mathcal{E}_\text{gen}(\mathrm{ERM}),
$$

where $\mathcal{E}_\text{gen} = \mathbb{E}_{X,Y}\!\left[(\hat{y}_{\mathbf{w}}(x) - y)^2\right]$.

## Free Energy Density

The **free energy density** acts as a cumulant generating function for the loss:

$$
f_N(\beta) = -\frac{1}{\beta N}\log\mathcal{Z}(\mathcal{D}), \qquad \frac{1}{N}\mathbb{E}_{\mathbf{w}|\mathcal{D}}[\mathcal{L}] = -\frac{\partial}{\partial\beta}\!\left(\beta f_N(\beta)\right).
$$

**Self-averaging.** In the thermodynamic limit $N, p \to \infty$ with $\alpha = p/N = \mathcal{O}(1)$, the free energy density concentrates around its mean almost surely:

$$
f_N(\beta) \xrightarrow{N\to\infty} f(\beta) = \lim_{N\to\infty}\mathbb{E}_{X,Y}[f_N(\beta)].
$$

This is the key property that makes the statistical physics approach tractable.

**Remark (quenched vs. annealed average).** By Jensen's inequality,

$$
\mathbb{E}_{\mathcal{D}}[\log\mathcal{Z}] \leq \log\mathbb{E}_{\mathcal{D}}[\mathcal{Z}(\mathcal{D})].
$$

The left-hand side is the *quenched* average (the physically correct one); the right-hand side is the *annealed* average (an upper bound). Computing the quenched average requires the replica method.

# Binary Gaussian Mixture Classification

We now study a concrete, exactly solvable model [@mignacco2020role].

**Data model.** Labels $y^\mu = \pm 1$ with equal probability, and

$$
x^\mu = \frac{y^\mu\,\mathbf{w}^*}{\sqrt{N}} + \sqrt{\Delta}\,z^\mu, \qquad z^\mu \sim \mathcal{N}(0, I_N),\quad \Delta > 0.
$$

The two classes form isotropic Gaussian clouds centered at $\pm \mathbf{w}^*/\sqrt{N}$, with noise level $\Delta$. The teacher weights lie on the sphere, $\mathbf{w}^* \sim S^{N-1}(\sqrt{N})$.

**Architecture.** A single-layer perceptron with bias/margin $\kappa$:

$$
\hat{y}_{\mathbf{w}}(x) = \operatorname{sign}(h + \kappa), \qquad h = \frac{\mathbf{w}\cdot x}{\sqrt{N}} \in \mathbb{R}.
$$

The random case $\mathbf{w}^* = 0$ (no signal) reduces to a **constraint satisfaction problem**: can $p$ random $\pm 1$ labels be shattered by a hyperplane in $\mathbb{R}^N$?

# The Replica Method

We need to compute the quenched free energy $\mathbb{E}_{\mathcal{D}}[\log\mathcal{Z}(\mathcal{D})]$. The **replica trick** [@mezard1987spin; @nishimori2001statistical; @engel2001statistical] exchanges the log for a derivative of a moment:

$$
\mathbb{E}_{\mathcal{D}}[\log\mathcal{Z}(\mathcal{D})] = \lim_{n\to 0^+}\frac{\partial}{\partial n}\mathbb{E}_{\mathcal{D}}\!\left[\mathcal{Z}(\mathcal{D})^n\right].
$$

The three main steps are:

1. **Treat $n$ as a positive integer** and write $\mathcal{Z}^n$ as a product over $n$ *replicas* $\{\mathbf{w}_a\}_{a=1}^n$ of the weights.
2. **Average over the data** $\mathcal{D}$, which decouples across samples and reduces the problem to an integral over finitely many *order parameters*.
3. **Analytically continue** to real $n$ and take $n\to 0^+$.

## Step 1: Averaging Over Data

Set $\Delta = 1$ for simplicity. Define the local fields $h_a^\mu = \mathbf{w}_a \cdot x^\mu / \sqrt{N}$. After averaging over the Gaussian inputs,

$$
\mathbf{h}^\mu \equiv (h_1^\mu, \ldots, h_n^\mu) \sim \mathcal{N}(y^\mu\,\mathbf{m},\, Q),
$$

where the two key order parameters are:

$$
m_a = \frac{\mathbf{w}_a \cdot \mathbf{w}^*}{N} \quad \text{(magnetization)}, \qquad Q_{ab} = \frac{\mathbf{w}_a \cdot \mathbf{w}_b}{N} \quad \text{(overlap matrix)}.
$$

One can then show

$$
\mathbb{E}_{\mathcal{D}}[\mathcal{Z}^n] = \int\!\left[\prod_a d\mathbf{w}_a\,P(\mathbf{w}_a)\right]\left[\mathcal{Z}_y(Q,\mathbf{m})\right]^p,
$$

where $\mathcal{Z}_y(Q,\mathbf{m}) = \mathbb{E}_{y,\mathbf{h}}\!\left[\prod_a P_y(y \mid h_a)\right]$ with $\mathbf{h} \sim \mathcal{N}(y\mathbf{m}, Q)$.

## Step 2: Introducing the Order Parameters

To enforce the definitions of $Q_{ab}$ and $m_a$ inside the integral, one inserts delta functions:

$$
1 = N\int dQ_{ab}\;\delta\!\left(NQ_{ab} - \mathbf{w}_a\cdot\mathbf{w}_b\right), \quad a \leq b,
$$

and similarly for $m_a$. Exponentiating via the Fourier representation of $\delta$ introduces conjugate variables $\hat{Q}_{ab}$, $\hat{m}_a$. In the large-$N$ limit the $\mathbf{w}$-integral becomes Gaussian and can be performed exactly, leaving an integral over $\mathcal{O}(n^2)$ scalar order parameters. The full action is

$$
\mathbb{E}_{\mathcal{D}}[\mathcal{Z}^n] \propto \int\!\left[\prod_{a\leq b}dQ_{ab}\,d\hat{Q}_{ab}\right]\!\left[\prod_a dm_a\,d\hat{m}_a\right] e^{-N\,S(Q,\hat{Q},\mathbf{m},\hat{\mathbf{m}})}, \tag{*}
$$

where the effective action is

$$
S(Q,\hat{Q},\mathbf{m},\hat{\mathbf{m}}) = \sum_{a\leq b}\hat{Q}_{ab}Q_{ab} + \sum_a\hat{m}_a m_a - \log\mathcal{Z}_{\mathbf{w}}(\hat{Q},\hat{\mathbf{m}}) - \alpha\log\mathcal{Z}_y(Q,\mathbf{m}),
$$

and the weight-space contribution is

$$
\log\mathcal{Z}_{\mathbf{w}}(\hat{Q},\hat{\mathbf{m}}) = \frac{1}{N}\log\int\prod_a d\mathbf{w}_a\,P_0(\mathbf{w}_a)\;\exp\!\left(\sum_{a\leq b}\hat{Q}_{ab}(\mathbf{w}_a\cdot\mathbf{w}_b) + \sum_a\hat{m}_a(\mathbf{w}_a\cdot\mathbf{w}^*)\right).
$$

For ridge regression, $P_0(\mathbf{w}) \propto e^{-\beta\lambda\|\mathbf{w}\|^2/2}$, the integral is Gaussian and gives

$$
\log\mathcal{Z}_{\mathbf{w}} = -\tfrac{1}{2}\log\det A + \tfrac{1}{2}\hat{\mathbf{m}}^\top A^{-1}\hat{\mathbf{m}} + \text{const},
\qquad A_{ab} = \begin{cases}\beta\lambda - 2\hat{Q}_{aa} & a = b \\ -2\hat{Q}_{ab} & a \neq b.\end{cases}
$$

## Step 3: Saddle-Point Equations

For $N \gg 1$, the integral $(*)$ is dominated by its saddle point. Setting $\partial S/\partial\hat{Q}_{ab} = 0$ and $\partial S/\partial\hat{m}_a = 0$ yields

$$
\hat{\mathbf{m}} = A\mathbf{m}, \qquad A^{-1} = Q - \mathbf{m}\mathbf{m}^\top,
$$

and the effective action reduces to

$$
S_\text{eff}(Q,\mathbf{m}) = \frac{\beta\lambda}{2}\operatorname{tr}Q - \frac{1}{2}\log\det(Q - \mathbf{m}\mathbf{m}^\top) - \alpha\log\mathcal{Z}_y(Q,\mathbf{m}) + \text{const}.
$$

# Replica-Symmetric Ansatz

For convex problems (e.g. logistic or ridge regression), the **replica-symmetric (RS) ansatz** is exact [@mezard1987spin]:

$$
Q_{ab} = r\,\delta_{ab} + q\,(1 - \delta_{ab}), \qquad m_a = m \quad \text{for all } a,
$$

where $r = \|\mathbf{w}_a\|^2/N$ (self-overlap) and $q = \mathbf{w}_a\cdot\mathbf{w}_b/N$ for $a\neq b$ (inter-replica overlap).

## Simplification Under RS

The trace and log-determinant become:

$$
\operatorname{tr}Q = nr, \qquad \log\det(Q - m^2\,\mathbf{1}\mathbf{1}^\top) \underset{n\to 0}{=} n\left[\log(r-q) + \frac{q - m^2}{r - q}\right] + \mathcal{O}(n^2).
$$

For $\mathcal{Z}_y$, the RS covariance $Q = (r-q)I_n + q\,\mathbf{1}\mathbf{1}^\top$ allows one to decouple the replicas via a shared Gaussian variable $\xi_0$:

$$
h_a = ym + \sqrt{r-q}\;\xi_a + \sqrt{q}\;\xi_0, \qquad \xi_a,\, \xi_0 \overset{\text{iid}}{\sim} \mathcal{N}(0,1).
$$

Conditioned on $\xi_0$, the replicas are independent and $\mathcal{Z}_y$ factorizes. After taking $n\to 0^+$:

$$
\log\mathcal{Z}_y(Q,m) = n\,\mathbb{E}_{y,\xi_0}\!\left[\log\mathbb{E}_\xi\!\left[P_y\!\left(y \;\Big|\; ym + \sqrt{r-q}\;\xi + \sqrt{q}\;\xi_0\right)\right]\right] + \mathcal{O}(n^2).
$$

## RS Free Energy

The full problem reduces to **three scalar order parameters** $(r, q, m)$:

$$
S_\text{eff}^{(\mathrm{RS})}(r,q,m) = \frac{\beta\lambda}{2}r - \frac{1}{2}\log(r-q) - \frac{q-m^2}{2(r-q)} - \alpha\,\mathbb{E}_{y,\xi_0}\!\left[\log\mathbb{E}_\xi\!\left[P_y(y\mid\cdots)\right]\right].
$$

All the $N$-dimensional geometry has been absorbed into three saddle-point equations $\partial S_\text{eff}^{(\mathrm{RS})}/\partial r = \partial S_\text{eff}^{(\mathrm{RS})}/\partial q = \partial S_\text{eff}^{(\mathrm{RS})}/\partial m = 0$.

**ERM limit ($\beta\to\infty$).** The combination $\chi = \beta(r - q) = \mathcal{O}(1)$ remains finite. It plays the role of a *susceptibility* — the linear response of the order parameter to a small perturbation of the effective field — a standard observable in disordered-systems physics.

# Generalization Error

For a fresh test sample $(\tilde{x}, \tilde{y}) \sim P_{XY}$, the pre-activation is $\tilde{h} = \mathbf{w}\cdot\tilde{x}/\sqrt{N}$. Expanding using the data model with $\Delta = 1$:

$$
\tilde{h} = \tilde{y}\,\frac{\mathbf{w}\cdot\mathbf{w}^*}{N} + \frac{\mathbf{w}\cdot z}{\sqrt{N}} \sim \mathcal{N}\!\left(\tilde{y}\,m,\; r\right),
$$

where $m = \mathbf{w}\cdot\mathbf{w}^*/N$ and $r = \|\mathbf{w}\|^2/N$ are the RS order parameters. The generalization error is therefore

$$
\mathcal{E}_\text{gen} = \mathbb{P}(\hat{y} \neq \tilde{y}) = \mathbb{P}\!\left(\tilde{y}\,(\tilde{h} + \kappa) < 0\right) = \mathbb{E}_{\tilde{y}}\!\left[\Phi\!\left(-\frac{\tilde{y}\,m + \kappa}{\sqrt{r}}\right)\right],
$$

where $\Phi$ is the standard Gaussian CDF. This depends only on the scalar order parameters $(m, r, \kappa)$ — all dimensionality has been absorbed. The full phase diagram as a function of $\alpha$, $\Delta$, and $\lambda$ is worked out in @mignacco2020role.
