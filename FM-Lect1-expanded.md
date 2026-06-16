---
title: Statistical Physics Approaches to High-Dimensional Learning
subtitle: "ICTP Summer School on Machine Learning — Lecture 1"
short_title: Motivation and Background
authors:
  - name: Francesca Mignacco, transcribed by Max Hirsch with the Claude LLM
subject: Lecture Notes
---

These notes cover the first lecture by Francesca Mignacco at the ICTP Summer School on Machine Learning. The central idea is that a trained neural network is a high-dimensional interacting system, and the tools statistical physicists developed to study such systems — disorder averaging, free energies, saddle-point methods, and the replica trick — give exact, quantitative predictions about learning and generalization. The analysis of the binary Gaussian mixture model follows @mignacco2020role.

# Three Ingredients of Machine Learning

A supervised learning problem is defined by a dataset

$$
\mathcal{D} = \{(x^\mu, y^\mu)\}_{\mu=1}^{p}, \qquad x^\mu \in \mathbb{R}^N,\; y^\mu \in \mathbb{R},
$$

and the **goal** is to learn a mapping $x \mapsto y$ that generalizes to unseen data: given a new input $x_\text{new}$, predict

$$
\hat{y}_\text{new} = f_{\mathbf{w}}(x_\text{new}).
$$

Three ingredients are needed:

**① Data.** The training set $\mathcal{D}$. In the statistical physics approach, data points are treated as random samples from some distribution $P_{XY}$, enabling disorder averaging.

**② Architecture.** The function class $f_{\mathbf{w}}$.

- *Perceptron:* $\hat{y} = \phi(\mathbf{w}^\top x)$, e.g. $\phi(\cdot) = \operatorname{sign}(\cdot + \kappa)$. Simple and exactly solvable, but limited to linearly separable tasks.
- *Deep neural network (DNN):* $\phi^{(L)}\!\left(W_{(L)}\,\phi^{(L-1)}(\cdots)\right)$. Composing many layers allows the network to learn hierarchical, nonlinear features.
- *Transformers.* Attention-based architectures suited to sequential data.

**③ Algorithm.** Gradient descent on a regularized empirical loss:

$$
\mathcal{L}(\mathbf{w};\mathcal{D}) = \sum_{\mu=1}^{p} \ell(y^\mu;\mathbf{w}, x^\mu) + \frac{\lambda}{2}\|\mathbf{w}\|_2^2,
$$

with update rule $\mathbf{w}^{t+1} = \mathbf{w}^{t} - \eta\,\nabla_{\mathbf{w}}\mathcal{L}(\mathbf{w}^{t})$, computed via backpropagation. The regularization term $\frac{\lambda}{2}\|\mathbf{w}\|^2$ penalizes large weights and corresponds to a Gaussian prior on $\mathbf{w}$.

# The Generalization Problem

The central challenge is that minimizing the **training error** $\mathcal{E}_\text{train}$ does not automatically minimize the **generalization error** $\mathcal{E}_\text{gen}$ on unseen data. Understanding this gap, and how it depends on the amount of data $p$, the dimension $N$, and the regularization $\lambda$, is the core problem these notes address.

**Connection to physics.** This is directly analogous to a problem in statistical mechanics: a system trained on $\mathcal{D}$ has been tuned to a particular realization of disorder (the random data). Generalization asks whether the resulting state also describes unseen samples from the same distribution — i.e. whether the learned $\mathbf{w}$ captures the underlying signal rather than the noise.

The *Hopfield model* [@hopfield1982neural] is an early and instructive example of the physics–ML connection: it models associative memory as energy minimization in a disordered system, where stored patterns are attractors of the dynamics. Retrieval of a memory corresponds to relaxation into the correct energy minimum — a direct physical picture of "inference."

# Exactly Solvable Models

To make analytical progress, one restricts to tractable settings via the **teacher–student** framework:

- Input data are Gaussian: $x \sim \mathcal{N}(0, I_N)$.
- A **teacher network** with fixed weights $\mathbf{w}^*$ generates the labels $y$ — it defines the ground truth.
- A **student network** with learnable weights $\mathbf{w}$ tries to recover the teacher.

The ratio $\alpha = p/N$ (sample complexity) is the key control parameter: with too little data ($\alpha \ll 1$) many weight vectors fit the training set equally well, and generalization is poor; with enough data ($\alpha \gg 1$) the teacher is recoverable.

The statistical physics approach rests on two ideas:

1. **Average-case analysis.** Rather than asking about worst-case data (as in PAC learning), one averages over the random draw of $\mathcal{D}$. This is the same philosophy as averaging over disorder in spin glasses.
2. **Thermodynamic limit.** Take $N, p \to \infty$ with $\alpha = p/N = \mathcal{O}(1)$ fixed. In this limit, fluctuations self-average and the problem becomes exactly solvable via a small number of macroscopic order parameters — much as thermodynamic quantities become sharp in the large-system limit of statistical mechanics.

The two goals are then: (i) understand **statics** — which $\mathbf{w}$ does the algorithm converge to, and what is its generalization error? — and (ii) understand **dynamics** — how does $\mathbf{w}^t$ evolve during training?

# Supervised Learning: Bayesian Formulation

Given $(x^\mu, y^\mu) \overset{\text{iid}}{\sim} P_{XY}(\cdot \mid \mathbf{w}^*)$ with $\mathbf{w}^* \sim P_{\mathbf{w}^*}$, Bayes' theorem gives the posterior over weights:

$$
P(\mathbf{w} \mid \mathcal{D}) = \frac{P_0(\mathbf{w})\prod_{\mu=1}^{p} P_y(y^\mu \mid \mathbf{w}, x^\mu)}{\mathcal{Z}(\mathcal{D})},
$$

where $P_0$ is the prior over weights, $P_y$ is the likelihood of an observed label, and

$$
\mathcal{Z}(\mathcal{D}) = \int d\mathbf{w}\; P_0(\mathbf{w})\prod_{\mu=1}^{p} P_y(y^\mu \mid \mathbf{w}, x^\mu)
$$

is the **partition function** — the normalizing constant that also encodes all thermodynamic information about the learning problem.

Identifying $-\log P_y(y^\mu \mid \mathbf{w}, x^\mu) \propto \ell(y^\mu; \mathbf{w}, x^\mu)$ and $-\log P_0(\mathbf{w}) \propto \frac{\lambda}{2}\|\mathbf{w}\|^2$, the posterior takes the form of a **Gibbs distribution**:

$$
P(\mathbf{w} \mid \mathcal{D}) = \frac{e^{-\beta\,\mathcal{L}(\mathbf{w};\mathcal{D})}}{\mathcal{Z}(\mathcal{D})}, \qquad \mathcal{Z}(\mathcal{D}) = \int d\mathbf{w}\;e^{-\beta\,\mathcal{L}(\mathbf{w};\mathcal{D})}.
$$

This is the fundamental bridge between machine learning and statistical physics: **the loss $\mathcal{L}$ plays the role of an energy, and $\beta$ is the inverse temperature.** The weights $\mathbf{w}$ are the microscopic degrees of freedom, and the dataset $\mathcal{D}$ is a realization of quenched disorder.

## Special Cases

**(i) Empirical Risk Minimization (ERM).** In the zero-temperature limit $\beta \to \infty$, the Gibbs measure collapses onto the loss minimizer:

$$
-\frac{1}{\beta}\log\mathcal{Z}(\mathcal{D}) \xrightarrow{\beta\to\infty} \min_{\mathbf{w}}\mathcal{L}(\mathbf{w};\mathcal{D}).
$$

Standard gradient descent, with a fixed learning rate run to convergence, implements ERM. A common loss for binary classification is logistic: $\ell(y;\mathbf{w},x) = \log(1 + e^{-y\,\mathbf{w}^\top x})$.

**(ii) Bayes-Optimal (BO) Setting.** If one sets $P_0 = P_{\mathbf{w}^*}$ and $P_y = P_{Y|X}$ — i.e. the prior and likelihood exactly match the true generative process — then the Bayesian posterior is optimal, and

$$
\mathcal{E}_\text{gen}(\mathrm{BO}) \leq \mathcal{E}_\text{gen}(\mathrm{ERM}),
$$

where $\mathcal{E}_\text{gen} = \mathbb{E}_{X,Y}\!\left[(\hat{y}_{\mathbf{w}}(x) - y)^2\right]$. The Bayes-optimal predictor is the best achievable by any algorithm; ERM with a misspecified prior or loss pays a price.

## Free Energy Density

The **free energy density** is the central quantity of interest:

$$
f_N(\beta) = -\frac{1}{\beta N}\log\mathcal{Z}(\mathcal{D}).
$$

Just as in thermodynamics, all macroscopic observables follow from derivatives of $f_N$. For instance, the mean loss under the posterior satisfies

$$
\frac{1}{N}\mathbb{E}_{\mathbf{w}|\mathcal{D}}[\mathcal{L}] = -\frac{\partial}{\partial\beta}\!\left(\beta f_N(\beta)\right),
$$

so $f_N$ is the cumulant generating function for the loss — knowledge of $f_N$ gives complete thermodynamic information.

**Self-averaging.** In the thermodynamic limit $N, p \to \infty$ with $\alpha = p/N = \mathcal{O}(1)$, the free energy density is *self-averaging*: sample-to-sample fluctuations vanish, and

$$
f_N(\beta) \xrightarrow{N\to\infty} f(\beta) = \lim_{N\to\infty}\mathbb{E}_{\mathcal{D}}[f_N(\beta)] \quad \text{almost surely.}
$$

This is the learning-theory analogue of the thermodynamic limit in physics: just as the free energy per spin of a large magnet is a sharp number independent of the specific spin configuration, the free energy per weight of a large learning problem is a sharp number independent of the specific dataset. It allows us to replace the hard problem of computing $f_N$ for a given $\mathcal{D}$ with the easier problem of computing its expectation over $\mathcal{D}$.

**Remark (quenched vs. annealed average).** The self-averaging property tells us we want $\mathbb{E}_{\mathcal{D}}[\log\mathcal{Z}]$. A naive approach would compute $\log \mathbb{E}_{\mathcal{D}}[\mathcal{Z}]$ instead — the *annealed* average, which is much easier since the log and expectation are exchanged. By Jensen's inequality this gives an upper bound:

$$
\mathbb{E}_{\mathcal{D}}[\log\mathcal{Z}] \leq \log\mathbb{E}_{\mathcal{D}}[\mathcal{Z}(\mathcal{D})].
$$

The left-hand side — the *quenched* average, where the disorder $\mathcal{D}$ is fixed before taking the log — is the physically correct one. The annealed average overcounts because it allows the partition function to be dominated by rare, atypically favorable datasets. Computing the quenched average is harder and requires the replica method.

# Binary Gaussian Mixture Classification

We now study the simplest nontrivial exactly solvable model [@mignacco2020role]. The data are drawn from a **binary Gaussian mixture**:

$$
y^\mu = \pm 1 \text{ with equal probability}, \qquad x^\mu = \frac{y^\mu\,\mathbf{w}^*}{\sqrt{N}} + \sqrt{\Delta}\,z^\mu,
$$

where $z^\mu \sim \mathcal{N}(0, I_N)$ is isotropic noise and $\Delta > 0$ controls the noise level. The teacher weights are normalized: $\mathbf{w}^* \sim S^{N-1}(\sqrt{N})$, so $\|\mathbf{w}^*\|^2 = N$.

**Physical picture.** The two classes form two isotropic Gaussian clouds in $\mathbb{R}^N$ centered at $\pm\mathbf{w}^*/\sqrt{N}$. The signal-to-noise ratio is set by $1/\Delta$: for small $\Delta$ the clouds are well separated and classification is easy; for large $\Delta$ the clouds heavily overlap and the problem becomes hard. The student must find a hyperplane that separates the two clouds, using only the $p$ labeled examples.

**Architecture.** A single-layer perceptron with bias/margin $\kappa$:

$$
\hat{y}_{\mathbf{w}}(x) = \operatorname{sign}(h + \kappa), \qquad h = \frac{\mathbf{w}\cdot x}{\sqrt{N}} \in \mathbb{R}.
$$

The scalar pre-activation $h$ is a sufficient statistic: all the information the student uses about $x$ is compressed into this single number.

The limiting case $\mathbf{w}^* = 0$ (no signal) is the **random labels** problem: can $p$ random $\pm 1$ labels be shattered by a linear classifier in $\mathbb{R}^N$? This is a **constraint satisfaction problem** — a central object in the statistical physics of disordered systems — and the answer is yes if and only if $\alpha = p/N < \alpha_c = 2$ (Gardner's capacity [@mezard1987spin]).

# The Replica Method

We need to compute the quenched free energy $\mathbb{E}_{\mathcal{D}}[\log\mathcal{Z}(\mathcal{D})]$. The difficulty is that the log sits inside the expectation over the random data — a standard obstacle in disordered systems.

The **replica trick** [@mezard1987spin; @nishimori2001statistical; @engel2001statistical] sidesteps this using the identity

$$
\log\mathcal{Z} = \lim_{n\to 0^+}\frac{\mathcal{Z}^n - 1}{n} = \lim_{n\to 0^+}\frac{\partial}{\partial n}\mathcal{Z}^n,
$$

so that

$$
\mathbb{E}_{\mathcal{D}}[\log\mathcal{Z}(\mathcal{D})] = \lim_{n\to 0^+}\frac{\partial}{\partial n}\mathbb{E}_{\mathcal{D}}\!\left[\mathcal{Z}(\mathcal{D})^n\right].
$$

The moment $\mathbb{E}[\mathcal{Z}^n]$ is far easier to compute than $\mathbb{E}[\log \mathcal{Z}]$: for integer $n$ one can write $\mathcal{Z}^n$ as a product of $n$ independent copies (replicas) of the system, average over $\mathcal{D}$, and then analytically continue to $n \to 0$. The procedure is non-rigorous but gives predictions confirmed by rigorous methods in many cases.

The three main steps are:

1. **Treat $n$ as a positive integer.** Write $\mathcal{Z}^n = \prod_{a=1}^n \mathcal{Z}(\mathcal{D})$ using $n$ independent replicas $\{\mathbf{w}_a\}_{a=1}^n$, each with the same loss but independent weight vectors.
2. **Average over the data $\mathcal{D}$.** The data enter only through the local fields $h_a^\mu = \mathbf{w}_a \cdot x^\mu/\sqrt{N}$, and since $x^\mu$ is Gaussian, this average can be done exactly. The replicas become coupled through a small number of macroscopic *order parameters*.
3. **Analytically continue to real $n$ and take $n\to 0^+$.**

## Step 1: Averaging Over Data

Set $\Delta = 1$. The local field for replica $a$ and sample $\mu$ is

$$
h_a^\mu = \frac{\mathbf{w}_a \cdot x^\mu}{\sqrt{N}} = y^\mu \frac{\mathbf{w}_a \cdot \mathbf{w}^*}{N} + \frac{\mathbf{w}_a \cdot z^\mu}{\sqrt{N}}.
$$

Averaging over the Gaussian noise $z^\mu$ (i.i.d. across samples), the joint distribution of the fields across replicas for a given sample becomes

$$
\mathbf{h}^\mu \equiv (h_1^\mu, \ldots, h_n^\mu) \sim \mathcal{N}(y^\mu\,\mathbf{m},\, Q),
$$

where the two key order parameters are:

$$
m_a = \frac{\mathbf{w}_a \cdot \mathbf{w}^*}{N} \quad \text{(magnetization)}, \qquad Q_{ab} = \frac{\mathbf{w}_a \cdot \mathbf{w}_b}{N} \quad \text{(overlap matrix)}.
$$

**Physical meaning.** The magnetization $m_a$ measures how well replica $a$ has aligned with the true teacher — it is the order parameter for learning. The off-diagonal overlap $Q_{ab}$ ($a \neq b$) measures how similar two independently trained replicas are to each other — it captures the geometry of the loss landscape. If the loss is convex, all replicas converge to the same minimum and $Q_{ab} \approx r$ (the self-overlap); in a rugged landscape, different replicas can get trapped in different minima and $Q_{ab} \ll r$.

After averaging over the $p$ i.i.d. samples, the data dependence factorizes:

$$
\mathbb{E}_{\mathcal{D}}[\mathcal{Z}^n] = \int\!\left[\prod_a d\mathbf{w}_a\,P(\mathbf{w}_a)\right]\left[\mathcal{Z}_y(Q,\mathbf{m})\right]^p,
$$

where $\mathcal{Z}_y(Q,\mathbf{m}) = \mathbb{E}_{y,\mathbf{h}}\!\left[\prod_a P_y(y \mid h_a)\right]$ with $\mathbf{h} \sim \mathcal{N}(y\mathbf{m}, Q)$.

## Step 2: Introducing the Order Parameters

The weight integral still runs over all of $\mathbb{R}^N$ for each replica. To reduce it to a finite-dimensional integral over $(Q, \mathbf{m})$, one inserts the identity

$$
1 = N\int dQ_{ab}\;\delta\!\left(NQ_{ab} - \mathbf{w}_a\cdot\mathbf{w}_b\right), \quad a \leq b,
$$

and similarly for $m_a$. Representing each $\delta$ function by its Fourier transform introduces conjugate variables $\hat{Q}_{ab}$ and $\hat{m}_a$. In the large-$N$ limit the $\mathbf{w}$-integral becomes Gaussian and can be performed exactly, giving

$$
\mathbb{E}_{\mathcal{D}}[\mathcal{Z}^n] \propto \int\!\left[\prod_{a\leq b}dQ_{ab}\,d\hat{Q}_{ab}\right]\!\left[\prod_a dm_a\,d\hat{m}_a\right] e^{-N\,S(Q,\hat{Q},\mathbf{m},\hat{\mathbf{m}})}, \tag{*}
$$

where the effective action separates into a **prior term** (encoding the weight geometry) and a **data term** (encoding the likelihood):

$$
S = \underbrace{\sum_{a\leq b}\hat{Q}_{ab}Q_{ab} + \sum_a\hat{m}_a m_a - \log\mathcal{Z}_{\mathbf{w}}(\hat{Q},\hat{\mathbf{m}})}_{\text{prior / weight geometry}} \;-\; \underbrace{\alpha\log\mathcal{Z}_y(Q,\mathbf{m})}_{\text{data}}.
$$

The weight-space partition function is

$$
\log\mathcal{Z}_{\mathbf{w}}(\hat{Q},\hat{\mathbf{m}}) = \frac{1}{N}\log\int\prod_a d\mathbf{w}_a\,P_0(\mathbf{w}_a)\;\exp\!\left(\sum_{a\leq b}\hat{Q}_{ab}(\mathbf{w}_a\cdot\mathbf{w}_b) + \sum_a\hat{m}_a(\mathbf{w}_a\cdot\mathbf{w}^*)\right).
$$

For ridge regression, $P_0(\mathbf{w}) \propto e^{-\beta\lambda\|\mathbf{w}\|^2/2}$, the integral is Gaussian and gives

$$
\log\mathcal{Z}_{\mathbf{w}} = -\tfrac{1}{2}\log\det A + \tfrac{1}{2}\hat{\mathbf{m}}^\top A^{-1}\hat{\mathbf{m}} + \text{const},
\qquad A_{ab} = \begin{cases}\beta\lambda - 2\hat{Q}_{aa} & a = b, \\ -2\hat{Q}_{ab} & a \neq b.\end{cases}
$$

## Step 3: Saddle-Point Equations

The integral $(*)$ has a prefactor $e^{-N(\cdots)}$, so for $N \gg 1$ it is dominated by the saddle point of the action. Extremizing over the conjugate variables $\hat{Q}_{ab}$ and $\hat{m}_a$ gives

$$
\hat{\mathbf{m}} = A\mathbf{m}, \qquad A^{-1} = Q - \mathbf{m}\mathbf{m}^\top,
$$

and the action reduces to an effective free energy in terms of $(Q, \mathbf{m})$ alone:

$$
S_\text{eff}(Q,\mathbf{m}) = \frac{\beta\lambda}{2}\operatorname{tr}Q - \frac{1}{2}\log\det(Q - \mathbf{m}\mathbf{m}^\top) - \alpha\log\mathcal{Z}_y(Q,\mathbf{m}) + \text{const}.
$$

The self-consistency equations $\partial S_\text{eff}/\partial Q_{ab} = 0$ and $\partial S_\text{eff}/\partial m_a = 0$ then determine the order parameters. This is the same procedure as finding the equations of state in a mean-field theory: one looks for a consistent solution where the macroscopic observables satisfy self-referential equations.

# Replica-Symmetric Ansatz

The order parameter matrix $Q$ has $\mathcal{O}(n^2)$ entries — infinitely many as $n\to 0^+$ if treated naively. One needs an ansatz for its structure. The simplest and most physically motivated is the **replica-symmetric (RS) ansatz**: all replicas are statistically equivalent,

$$
Q_{ab} = r\,\delta_{ab} + q\,(1 - \delta_{ab}), \qquad m_a = m \quad \text{for all } a.
$$

Here $r = \|\mathbf{w}_a\|^2/N$ is the self-overlap (mean-square norm per dimension) and $q = \mathbf{w}_a\cdot\mathbf{w}_b/N$ for $a\neq b$ is the inter-replica overlap. The RS ansatz is **exact for convex problems** [@mezard1987spin] (e.g. ridge regression, logistic regression) because convex losses have a unique minimum, so all replicas must converge to the same point: $q = r$. For non-convex losses a more complex replica-symmetry-breaking (RSB) structure may be needed.

## Simplification Under RS

The trace and log-determinant reduce to scalar expressions. Using the matrix determinant lemma on $Q - m^2\mathbf{1}\mathbf{1}^\top = (r-q)I_n + (q-m^2)\mathbf{1}\mathbf{1}^\top$:

$$
\operatorname{tr}Q = nr, \qquad \log\det(Q - m^2\,\mathbf{1}\mathbf{1}^\top) \underset{n\to 0}{=} n\left[\log(r-q) + \frac{q - m^2}{r - q}\right] + \mathcal{O}(n^2).
$$

For $\mathcal{Z}_y$, the RS covariance $Q = (r-q)I_n + q\,\mathbf{1}\mathbf{1}^\top$ has a useful structure: it says all replicas share the same "common noise" $\xi_0$ and have independent "private noise" $\xi_a$. Concretely, one can write

$$
h_a = ym + \sqrt{r-q}\;\xi_a + \sqrt{q}\;\xi_0, \qquad \xi_a,\, \xi_0 \overset{\text{iid}}{\sim} \mathcal{N}(0,1).
$$

**Physical picture.** The variable $\xi_0$ represents disorder shared by all replicas — the part of the weight configuration that is pinned by the data. The variable $\xi_a$ represents fluctuations private to each replica around this common background. When $q \to r$ (ERM limit), the private fluctuations $\sqrt{r-q}\,\xi_a$ vanish, all replicas collapse to the same solution, and there is no variance across the posterior.

Conditioned on $\xi_0$, the replicas are independent and $\mathcal{Z}_y$ factorizes over $a$. After taking $n\to 0^+$:

$$
\log\mathcal{Z}_y(Q,m) = n\,\mathbb{E}_{y,\xi_0}\!\left[\log\mathbb{E}_\xi\!\left[P_y\!\left(y \;\Big|\; ym + \sqrt{r-q}\;\xi + \sqrt{q}\;\xi_0\right)\right]\right] + \mathcal{O}(n^2).
$$

## RS Free Energy

Substituting into $S_\text{eff}$, all $N$-dimensional integrals have been absorbed, and the problem reduces entirely to **three scalar order parameters** $(r, q, m)$:

$$
S_\text{eff}^{(\mathrm{RS})}(r,q,m) = \frac{\beta\lambda}{2}r - \frac{1}{2}\log(r-q) - \frac{q-m^2}{2(r-q)} - \alpha\,\mathbb{E}_{y,\xi_0}\!\left[\log\mathbb{E}_\xi\!\left[P_y(y\mid\cdots)\right]\right].
$$

The saddle-point equations $\partial S_\text{eff}^{(\mathrm{RS})}/\partial r = \partial S_\text{eff}^{(\mathrm{RS})}/\partial q = \partial S_\text{eff}^{(\mathrm{RS})}/\partial m = 0$ are three coupled self-consistency equations that can be solved numerically. This is the main result of the replica calculation: a problem that started in $\mathbb{R}^N$ has been reduced to three equations in three unknowns.

**ERM limit ($\beta\to\infty$).** As $\beta\to\infty$ the Gibbs measure concentrates, the posterior variance $r - q \to 0$, but the product $\chi = \beta(r - q)$ remains $\mathcal{O}(1)$. The quantity $\chi$ is the **susceptibility** — it measures the linear response of the order parameters to a small perturbation of the effective field, and is a standard observable in spin glass theory. In ML terms, it is related to the sensitivity of the learned weights to small changes in the training data.

# Generalization Error

Once the order parameters $(r, q, m)$ are known from the saddle-point equations, the generalization error follows analytically. For a fresh test sample $(\tilde{x}, \tilde{y}) \sim P_{XY}$, the student's pre-activation is

$$
\tilde{h} = \frac{\mathbf{w}\cdot\tilde{x}}{\sqrt{N}} = \tilde{y}\,\frac{\mathbf{w}\cdot\mathbf{w}^*}{N} + \frac{\mathbf{w}\cdot \tilde{z}}{\sqrt{N}} \sim \mathcal{N}\!\left(\tilde{y}\,m,\; r\right),
$$

where the mean comes from the alignment with the teacher ($m$) and the variance from the residual weight norm ($r$). The generalization error is

$$
\mathcal{E}_\text{gen} = \mathbb{P}(\hat{y} \neq \tilde{y}) = \mathbb{P}\!\left(\tilde{y}\,(\tilde{h} + \kappa) < 0\right) = \mathbb{E}_{\tilde{y}}\!\left[\Phi\!\left(-\frac{\tilde{y}\,m + \kappa}{\sqrt{r}}\right)\right],
$$

where $\Phi$ is the standard Gaussian CDF. This is a closed-form expression: the generalization error is determined entirely by the ratio $m/\sqrt{r}$, the signal-to-noise ratio of the student's pre-activation on a test point.

**Physical interpretation.** The ratio $m/\sqrt{r}$ has a clean meaning: $m = \mathbf{w}\cdot\mathbf{w}^*/N$ is the overlap with the teacher (signal), while $\sqrt{r} = \|\mathbf{w}\|/\sqrt{N}$ is the total weight scale (signal + noise). A perfect student would have $\mathbf{w} = \mathbf{w}^*$, giving $m = r = 1$ and error $\Phi(-\kappa/\sqrt{1}) = \Phi(-\kappa)$ — determined only by the margin. As $\alpha \to \infty$ (infinite data), $m/\sqrt{r} \to 1/\sqrt{\Delta}$, recovering the Bayes-optimal error set by the intrinsic noise $\Delta$.

The full phase diagram of generalization error as a function of sample complexity $\alpha$, noise level $\Delta$, and regularization strength $\lambda$ is worked out in @mignacco2020role.
