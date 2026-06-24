---
title: Statistical Physics Approaches to High-Dimensional Learning
subtitle: "ICTP Summer School on Machine Learning — Lecture 2 (with Day 3 continuation)"
short_title: Clustering via Approximate Message Passing
authors:
  - name: Davide Ghio (Aston University), transcribed by Max Hirsch with the Claude LLM
subject: Lecture Notes
venue: ICTP-INdAM-SLMath Summer Graduate School for Machine Learning, Trieste, 15–19 June 2026
bibliography: references.bib
---

These notes cover the second lecture on clustering with approximate message passing (AMP), together with its continuation at the start of Day 3. The running theme is that clustering a Gaussian mixture can be reframed as a *planted* spin-glass problem, and the cavity method — originally developed for the Sherrington–Kirkpatrick spin glass {cite}`thouless1977solution` — gives both the optimal achievable accuracy and a practical algorithm (AMP) for achieving it. The analysis follows {cite}`lesieur2016phase`.

**A note on perspective for PDE readers.** AMP is an iterative algorithm whose *exact* large-$n$ behaviour is tracked by a scalar recursion called state evolution. The analogy to PDE numerics is this: state evolution is like a CFL-type analysis, but instead of stability it tracks the fixed points and basins of attraction of the algorithm. The "hard phase" — where reconstruction is information-theoretically possible but algorithmically intractable — is analogous to a regime where a numerical method is convergent but with a basin of attraction that shrinks as the mesh is refined.

The accompanying Jupyter notebook (`ICTP_day2_notebook.ipynb`) is organised into three parts. **Part 1** implements the planted SK model, AMP, and state evolution for the rank-one case, and traces the AMP overlap transition around the IT threshold. **Part 2** computes the Bethe free entropy as a function of the overlap order parameter for the Rademacher–Bernoulli prior, identifying the four regimes separated by $\lambda_\text{spi} < \lambda_\text{IT} < \lambda_\text{alg}$. **Part 3** generalises to rank-$r$ asymmetric matrix estimation and runs Low-rank AMP on sampled GMM data.

# Unsupervised Learning

The goal of unsupervised learning is to find hidden structure in unlabelled data. Three main families:

- **Dimensionality reduction** (PCA, t-SNE, autoencoders): represent high-dimensional data in a low-dimensional space that preserves relevant structure.
- **Clustering** (k-means, GMM): partition data into groups of similar points.
- **Generative modelling** (diffusion models, LLMs): learn the data distribution $p_\theta(\text{data})$ and sample new examples.

This lecture studies clustering for data drawn from a Gaussian mixture. The three central questions are: when is recovering the cluster memberships (i) information-theoretically (IT) possible, (ii) achievable by a polynomial-time algorithm, and (iii) is there a *gap* between these thresholds?

# Setting Up the Model

Throughout, $n$ (dimension) and $m$ (sample size) go to infinity with the rank $r = \mathcal{O}(1)$ and the load $\alpha = m/n = \mathcal{O}(1)$ fixed.

## Generative Model

Following {cite}`lesieur2016phase`, each data point is drawn as

$$x_i = \sqrt{\frac{\rho}{n}}\,V_{t_i} + u_i, \qquad t_i \sim \mathrm{Uniform}[r], \quad V_k \sim \mathcal{N}(0,I_n), \quad u_i \sim \mathcal{N}(0,I_n).$$

Here $V_k \in \mathbb{R}^n$ is the (unknown) centroid of cluster $k$, $t_i \in \{1,\ldots,r\}$ is the (unknown) cluster assignment of point $i$, and $u_i$ is observation noise. The signal strength $\rho$ plays the role of an inverse noise level: larger $\rho$ means more separated centroids and easier clustering.

## Reformulation as Low-Rank Matrix Factorisation

Pack the unknowns into a centroid matrix $V \in \mathbb{R}^{n \times r}$ (columns = centroids) and a one-hot assignment matrix $S \in \{0,1\}^{m \times r}$ (each row has exactly one 1). The observation matrix $X \in \mathbb{R}^{n \times m}$ is:

$$X = \sqrt{\frac{\rho}{n}}\underbrace{VS^\top}_{\text{rank-}r\text{ signal}} + \underbrace{U}_{\text{noise}} \in \mathbb{R}^{n\times m}.$$

This is a **low-rank-plus-noise** problem: recover $V$ and $S$ from a noisy rank-$r$ perturbation of a pure noise matrix. This structure appears throughout applied mathematics — in compressed sensing, matrix completion, community detection — and the physics tools apply to all of them.

## Bayesian Posterior

Bayes' rule gives the posterior over unknowns:

$$P(S,V\mid X) = \frac{1}{\mathcal{Z}(X)}\prod_i P_V(\underline{v}_i)\prod_j P_S(\underline{s}_j)\,e^{\mathcal{H}(S,V)},$$

where $\mathcal{H}$ is the log-likelihood of observing $X$ given $(S,V)$ and $\mathcal{Z}(X)$ is the partition function. This has exactly the Gibbs-measure structure of Lecture 1: posterior = Boltzmann distribution, negative log-likelihood = energy, data = quenched disorder.

# Nishimori Identities

In the **Bayes-optimal setting** (the assumed model perfectly matches the true one), there is a remarkable symmetry between the truth and posterior samples. For any test function $f$:

$$\mathbb{E}_{x^*,y}\big[f_y(x^*)\big] = \mathbb{E}_{x,y}\big[f_y(x)\big] \qquad \text{(Bayes-optimal),}$$

where $x^*$ is the true signal and $x$ is a sample from the posterior. The key practical consequence: the overlap between a posterior sample and the truth equals the overlap between two independent posterior samples,

$$m = q \qquad \text{(Bayes-optimal).}$$

In the replica language of Lecture 1, this collapses the two order parameters $m$ and $q$ into one, substantially simplifying the saddle-point equations. It is analogous to a symmetry that collapses the PDE system from order $2n$ to order $n$.

# From Clustering to the Sherrington–Kirkpatrick Model

For $r = 1$ the observation matrix is symmetric and the problem reduces to the **spiked Wigner model**:

$$Y = \sqrt{\frac{\lambda}{n}}\,\underline{x}\,\underline{x}^\top + \xi, \qquad \underline{x}\in\mathbb{R}^n, \quad \xi_{ij}=\xi_{ji}\overset{\text{iid}}{\sim}\mathcal{N}(0,1).$$

With an Ising ($\pm 1$) prior $P_X(x_i) = \frac{1}{2}(\delta_{x_i,1}+\delta_{x_i,-1})$, so $x_i^2 = 1$, the posterior Hamiltonian reduces to

$$\mathcal{H}(\underline{x}) = \sqrt{\frac{\lambda}{n}}\sum_{i<j}Y_{ij}\,x_i x_j,$$

which is precisely the **planted Sherrington–Kirkpatrick (SK) Hamiltonian** — a disordered ferromagnet with couplings $J_{ij} \propto Y_{ij}$ that contain a hidden rank-one signal $\sqrt{\lambda/n}\,x_i^*x_j^*$ planted in the noise. The pure SK model (no planted signal) was solved by the replica method {cite}`thouless1977solution; mezard1987spin`; here the structure is richer because the disorder is correlated.

## The Cavity Method

The cavity method is an alternative to replicas that gives the same saddle-point equations but with a more algorithmic flavour. The idea is to ask: what is the marginal distribution of a single spin $x_0$ given all the others?

**Step 1 (cavity field).** Remove spin $x_0$ from an $n$-spin equilibrium system. The field it receives from the remaining spins is

$$\sqrt{\frac{\lambda}{n}}\sum_i J_{0i}\,x_i.$$

This is a sum of $n-1$ weakly correlated terms; by the central limit theorem it converges as $n\to\infty$ to a Gaussian:

$$\sqrt{\frac{\lambda}{n}}\sum_i \xi_{0i}\,x_i \;\rightsquigarrow\; \sqrt{\lambda q}\;z, \qquad z\sim\mathcal{N}(0,1),$$

where $q = \frac{1}{n}\sum_i \langle x_i \rangle^2$ is the self-overlap of the posterior. This Gaussian approximation is the cavity assumption; it becomes exact as $n \to \infty$ due to the weak correlations between different spins.

**Step 2 (self-consistency).** Using the Nishimori identity $m = q$, the self-consistent **state evolution fixed-point equation** is:

$$m = \mathbb{E}_{z,x_0^*}\!\left[\tanh\!\left(\sqrt{\lambda m}\,z + \lambda m\,x_0^*\right)\,x_0^*\right],$$

where the expectation is over $z \sim \mathcal{N}(0,1)$ and $x_0^* = \pm 1$ uniformly. This is a scalar equation in one unknown $m \in [0,1]$; its fixed points and their stability determine whether the algorithm can recover the signal. The equation has the structure of a mean-field self-consistency equation — identical in spirit to those arising in continuum limits of interacting particle systems or mean-field PDEs.

# From the Cavity Method to AMP

The cavity recursion suggests an iterative algorithm. Define the **denoiser** (posterior mean of $x$ given a Gaussian observation):

$$\eta(A,B) = \frac{\int dx\,P_X(x)\,x\,e^{-\frac{A}{2}x^2+Bx}}{\int dx\,P_X(x)\,e^{-\frac{A}{2}x^2+Bx}}.$$

For the Ising prior, $\eta(A,B) = \tanh(B)/(1 + A\tanh(B)^2)$ approximately. The naive iteration of cavity equations suffers from a feedback problem: spin $i$'s own influence returns through its neighbours, causing double-counting. Correcting for this with a first-order Taylor expansion gives the **Onsager correction term**, and collecting everything yields **Approximate Message Passing (AMP)**:

$$\begin{cases} \underline{h}^t = \sqrt{\dfrac{\lambda}{n}}\,Y\,\underline{x}^t - \underbrace{\lambda\!\left(\dfrac{1}{n}\sum_i\eta_i'\right)}_{\text{Onsager correction}}\underline{x}^{t-1}, \\[3mm] \underline{x}^{t+1} = \eta\!\left(\dfrac{\lambda}{n}\underline{x}^t\cdot\underline{x}^t,\;\underline{h}^t\right). \end{cases}$$

The Onsager correction removes the self-feedback of each spin, restoring the validity of the cavity (Gaussian) approximation at each step. Without it the algorithm diverges; with it the trajectory is described *exactly* in the $n \to \infty$ limit by the scalar state evolution recursion. This is a uniquely strong theoretical guarantee: in contrast to most iterative numerical methods where convergence analysis is approximate, AMP's trajectory is provably tracked by a deterministic scalar map.

**TAP equations.** The fixed points of AMP are the solutions of the **Thouless–Anderson–Palmer (TAP) equations** {cite}`thouless1977solution`:

$$\langle x_i\rangle \approx \eta\!\left(\frac{\lambda}{n}\sum_{j\neq i}\langle x_j^2\rangle,\;\sqrt{\frac{\lambda}{n}}\sum_{j\neq i}Y_{ij}\langle x_j\rangle\right).$$

These are the cavity-corrected mean-field equations; they coincide with the saddle-point equations of the replica calculation — a strong cross-check of both methods.

## Low-Rank AMP for Clustering GMMs

The rank-one AMP extends straightforwardly to rank-$r$ asymmetric clustering by replacing scalars with length-$r$ vectors and matrices, giving the **Low-rank AMP (Low-rAMP)** algorithm of {cite}`lesieur2016phase`. Low-rAMP achieves the Bayes-optimal MMSE and runs in time linear in the data size per iteration.

# The Bethe Free Entropy

*(Day 3 continuation.)*

The cavity method gives fixed-point equations, but a single scalar potential — the **Bethe free entropy** — whose extremisation reproduces them also unlocks the full phase diagram.

**Definition:**

$$\phi_B = \lim_{n\to\infty}\mathbb{E}\!\left[\frac{1}{n}\log Z_n\right].$$

**Derivation via telescoping.** Write $\frac{1}{n}\log Z_n$ as a Cesàro mean of single-spin increments $\log(Z_i/Z_{i-1})$. By self-averaging, each increment has the same limiting value:

$$\phi_B = \lim_{n\to\infty}\mathbb{E}\!\left[\log\frac{Z_n}{Z_{n-1}}\right].$$

Using the CLT approximation for the cavity field and **Stein's lemma** ($\mathbb{E}[g(Z)Z] = \mathbb{E}[g'(Z)]$ for $Z \sim \mathcal{N}(0,1)$) to handle the disorder-average correction term $\mathbb{E}[\xi_{ij}\langle x_i x_j\rangle] = \mathbb{E}[\langle x_i^2 x_j^2\rangle - \langle x_i x_j\rangle^2]$, one arrives at the closed-form expression:

$$\boxed{\phi_B = \frac{\lambda}{4}m^2 - \frac{\lambda}{4}q^2 - \mathbb{E}_{z,x_0^*}\!\left[\log\int dx_0\,P_X(x_0)\,\exp\!\left(\lambda m\,x_0 x_0^* - \tfrac{\lambda}{2}x_0^2 + \sqrt{\lambda q}\,x_0\,z\right)\right].}$$

Extremising $\phi_B$ over $m$ (using $m = q$ from Nishimori) recovers the state evolution fixed-point equation. The Bethe free entropy is thus the *potential function* that AMP implicitly optimises — the analogue of a Lyapunov function for the iteration.

## Phase Diagrams and Statistical-to-Computational Gaps

The Bethe free entropy as a function of $m$ can have multiple local maxima, corresponding to distinct fixed points of state evolution. The global structure determines the phase diagram.

For $r < r_c = 4 + 2\alpha$ there is no hard phase: the IT and algorithmic thresholds coincide at $\rho_c(\alpha,r) = r/\alpha$. For $r > r_c$ three thresholds separate {cite}`lesieur2016phase`:

$$\rho_\text{IT}(\alpha,r) = \frac{2r\log r}{\alpha}(1+o_r(1)), \qquad \rho_c(\alpha,r) = \frac{r}{\alpha}, \qquad \rho_\text{sp}(\alpha,r) = \frac{2r\log r}{\alpha}(1+o_r(1)).$$

The **hard phase** $\rho_\text{IT} < \rho < \rho_c$ is the regime where:
- The Bethe free entropy has a global maximum at a nontrivial $m > 0$ (so Bayes-optimal recovery is possible).
- But AMP initialised from $m = 0$ converges to the trivial fixed point $m = 0$ (failure to recover).

The hard phase is a statistical-to-computational gap: reconstruction is IT-possible but no known polynomial-time algorithm achieves it from an uninformative initialisation. This is believed (but not proved in general) to be a fundamental obstruction, not an artefact of AMP.

**Analogy for PDE readers.** Think of the Bethe free entropy landscape as the energy of a nonlinear system with multiple equilibria. The IT threshold is where a nontrivial equilibrium first appears (a saddle-node bifurcation). The algorithmic threshold is where the trivial equilibrium loses stability (a transcritical bifurcation). Between these two thresholds, both equilibria coexist: depending on initialisation, the system converges to one or the other.
