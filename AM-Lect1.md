---
title: "Empirical Risk Minimization: Generalization, Rademacher Complexity, and Two-Layer Networks"
subtitle: "ICTP Summer School on Machine Learning — Lecture 4 (Day 6, Morning)"
short_title: ERM and Rademacher Complexity
authors:
  - name: Andrea Montanari, transcribed by Max Hirsch with the Claude LLM
subject: Lecture Notes
---

These notes cover the first lecture of Day 6 by Andrea Montanari. The topic is the classical statistical learning theory framework: empirical risk minimization (ERM), the decomposition of risk into approximation and generalization error, and Rademacher complexity as the tool for controlling the generalization gap. The lecture culminates in a sharp bound on the Rademacher complexity of two-layer neural networks (Bartlett 1996), and an abstract detour into Banach-space approximation theory that leads to the Barron norm and explains why shallow networks can efficiently approximate a broad function class.

## Supervised Learning Setup

We observe $n$ i.i.d. pairs $(x_i, y_i) \sim \mathbb{P}$, where

$$
x_i \in \mathbb{R}^d \quad \text{(feature vector)}, \qquad y_i \in \mathbb{R} \quad \text{(label or response)}.
$$

A **model** is a function $f : \mathbb{R}^d \to \mathbb{R}$. Given a new test pair $(x_{n+1}, y_{n+1})$ drawn from the same distribution, we want $f(x_{n+1}) \approx y_{n+1}$.

### Loss, Risk, and Empirical Risk

Fix a **loss function** $\ell : \mathbb{R} \times \mathbb{R} \to \mathbb{R}_{\geq 0}$, mapping $(y, \hat{y}) \mapsto \ell(y, \hat{y})$. The canonical example is the **square loss** $\ell(y, \hat{y}) = (y - \hat{y})^2$.

The **population risk** (or test loss) of $f$ is

$$
R(f) = \mathbb{E}\,\ell(y, f(x)),
$$

where the expectation is over $(x,y) \sim \mathbb{P}$. Since $\mathbb{P}$ is unknown, we cannot minimize $R(f)$ directly. We instead minimize the **empirical risk**

$$
\hat{R}_n(f) = \hat{\mathbb{E}}_n\,\ell(y, f(x)) = \frac{1}{n}\sum_{i=1}^n \ell(y_i, f(x_i)).
$$

By the law of large numbers, $\hat{R}_n(f) \to R(f)$ almost surely for any **fixed** $f$.

:::{admonition} Exercise
:class: tip

Take $\ell$ = square loss. Under $\mathbb{P}$:

1. What $f$ minimizes $R(f)$? The minimizer is $\hat{f}(x) = \mathbb{E}[y \mid x]$ (the conditional mean), since

$$
\operatorname*{arg\,min}_f \mathbb{E}\bigl[(y - f(x))^2\bigr] = \operatorname*{arg\,min}_f \|y - f(x)\|_{L^2(\mathbb{P})}^2.
$$

2. Suppose $y = f_*(x) + \sigma g$ where $g \sim \mathcal{N}(0,1)$ is independent of $x$. What is $\min_f R(f)$? Answer: $\sigma^2$ (the irreducible noise variance).

3. Suppose $x_1, \ldots, x_n$ are $n$ **distinct** points in $\mathbb{R}^d$. What is $\min_f \hat{R}_n(f)$? Answer: $0$, by interpolation.
:::

This third point is the fundamental tension of learning: by interpolating the training data one can drive $\hat{R}_n(f)$ to zero, yet such a function generalizes arbitrarily poorly. Minimizing $\hat{R}_n$ over all measurable functions does not minimize $R$.

## The Approximation–Generalization Decomposition

Let $\mathcal{F}$ be a hypothesis class (e.g. neural networks of a given architecture). Define

$$
\hat{f}_n = \operatorname*{arg\,min}_{f \in \mathcal{F}} \hat{R}_n(f), \qquad
f_* = \operatorname*{arg\,min}_{f \in \mathcal{F}} R(f), \qquad
f_{\mathcal{F}\infty} = \operatorname*{arg\,min}_{f \in \mathcal{F}} \hat{R}_n(f).
$$

(Here $f_*$ minimizes the true risk within $\mathcal{F}$, while $\hat{f}_n$ minimizes the empirical risk; they differ because $\hat{R}_n$ is a random function of the data.) We decompose the **excess risk**:

$$
R(\hat{f}_n) - R(f_*) = \underbrace{\bigl[R(\hat{f}_n) - \hat{R}_n(\hat{f}_n)\bigr]}_{\text{(I)}}
+ \underbrace{\bigl[\hat{R}_n(\hat{f}_n) - \hat{R}_n(f_*)\bigr]}_{\text{(II)} \leq 0}
+ \underbrace{\bigl[\hat{R}_n(f_*) - R(f_*)\bigr]}_{\text{(III)}}.
$$

Term (II) is $\leq 0$ because $\hat{f}_n$ minimizes $\hat{R}_n$ over $\mathcal{F}$, so it does no worse than $f_*$. Terms (I) and (III) are both instances of the fluctuation $|\hat{R}_n(f) - R(f)|$ at a particular $f$. Bounding them uniformly over $\mathcal{F}$ gives

$$
R(\hat{f}_n) - R(f_*) \lesssim \underbrace{\sup_{f \in \mathcal{F}} |R(f) - \hat{R}_n(f)|}_{\text{generalization error}} + \underbrace{\mathcal{O}\!\left(\tfrac{1}{\sqrt{n}}\right)}_{\text{fluctuations}}.
$$

If $f_*$ is not in $\mathcal{F}$, there is an additional **approximation error** term

$$
\inf_{f \in \mathcal{F}} R(f) - \inf_f R(f) \geq 0,
$$

representing the best-in-class deficit. The full bound therefore reads

$$
\boxed{R(\hat{f}_n) \leq \underbrace{\inf_{f \in \mathcal{F}} R(f)}_{\text{best in class}} + \underbrace{\sup_{f \in \mathcal{F}} |R(f) - \hat{R}_n(f)|}_{\text{generalization gap}} + \mathcal{O}\!\left(\tfrac{1}{\sqrt{n}}\right).}
$$

The picture below (from the lecture) shows the risk landscape: $R(f)$ has a true minimizer $f^*$, while $\hat{R}_n$ has a jagged empirical minimizer $\hat{f}$ that may be far from $f^*$ if $\mathcal{F}$ is too rich.

:::{note}
If $\mathcal{F}$ = all measurable functions, the generalization error in the example above is $\geq \sigma^2$: the model simply memorizes the noise. Controlling the generalization gap requires restricting $\mathcal{F}$.
:::

## Rademacher Complexity

The natural tool for bounding the generalization gap uniformly over $\mathcal{F}$ is the **Rademacher complexity**.

### Definition

For a set $A \subseteq \mathbb{R}^n$, define

$$
\mathrm{Rd}_n(A) := \mathbb{E}_\sigma \sup_{a \in A} \frac{1}{n} \langle \sigma, a \rangle,
$$

where $\sigma = (\sigma_1, \ldots, \sigma_n)$ are i.i.d. $\mathrm{Uniform}(\{+1,-1\})$ (Rademacher variables), independent of everything else. Geometrically, $\mathrm{Rd}_n(A)$ measures how well the set $A$ can be correlated with a random sign vector: it is large when $A$ contains vectors that "stick out" in every direction.

For a function class $G = \{g : S \to \mathbb{R}\}$ with data $z_1, \ldots, z_n \overset{\text{iid}}{\sim} \mathbb{P}$, define

$$
\mathrm{Rd}_n(G) := \mathbb{E}_z\,\mathbb{E}_\sigma \sup_{g \in G} \frac{1}{n}\sum_{i=1}^n \sigma_i g(z_i).
$$

The connection to the geometric definition is: fix $z_1, \ldots, z_n$ and let $A = \{(g(z_1), \ldots, g(z_n)) : g \in G\} \subseteq \mathbb{R}^n$ (the "evaluation set"), e.g. the output vectors of a neural network class on the training points. Then $\mathrm{Rd}_n(G) = \mathbb{E}_z[\mathrm{Rd}_n(A)]$.

### Rademacher Complexity Controls Uniform Convergence

:::{prf:lemma} Symmetrization
:label: lem-symmetrization

Assume $G$ is **symmetric**: $g \in G \Rightarrow -g \in G$. Then

$$
\frac{1}{2}\,\mathrm{Rd}_n(G) \leq \mathbb{E}\sup_{g \in G}\bigl|\hat{\mathbb{E}}_n g - \mathbb{E} g\bigr| \leq 2\,\mathrm{Rd}_n(G).
$$
:::

:::{prf:proof}

**Upper bound.** Let $z_i'$ be i.i.d. copies of $z_i$ (a "ghost" sample). By convexity of the supremum (Jensen),

$$
(*) := \mathbb{E}_z \sup_{g \in G} \left|\mathbb{E}_{z'} \frac{1}{n}\sum_{i=1}^n \bigl[g(z_i) - g(z_i')\bigr]\right|
\leq \mathbb{E}_{z,z'} \sup_{g \in G} \left|\frac{1}{n}\sum_{i=1}^n \bigl[g(z_i) - g(z_i')\bigr]\right|.
$$

Since $(z_i, z_i')$ and $(z_i', z_i)$ have the same joint distribution, the sign of each term $g(z_i) - g(z_i')$ can be replaced by an independent Rademacher $\sigma_i$ without changing the distribution:

$$
\leq \mathbb{E}_{z,z',\sigma} \sup_{g \in G} \left|\frac{1}{n}\sum_{i=1}^n \sigma_i \bigl[g(z_i) - g(z_i')\bigr]\right|
\leq 2\,\mathbb{E}_{z,\sigma} \sup_{g \in G} \left|\frac{1}{n}\sum_{i=1}^n \sigma_i g(z_i)\right|
= 2\,\mathrm{Rd}_n(G).
$$

The last inequality uses the triangle inequality and symmetry of $G$.

To connect $\mathbb{E}\sup |\hat{\mathbb{E}}_n g - \mathbb{E} g|$ to $(*)$: note $\mathbb{E} g = \mathbb{E}_{z'} \frac{1}{n}\sum_i g(z_i')$, so $\hat{\mathbb{E}}_n g - \mathbb{E} g = \mathbb{E}_{z'}[\frac{1}{n}\sum_i(g(z_i) - g(z_i'))]$. Taking the supremum and applying Jensen gives the bound. $\square$
:::

To apply this to learning, set $z_i = (y_i, x_i)$, $g(z_i) = \ell(y_i, f(x_i))$, and $G = \ell \circ \mathcal{F} := \{(y,x) \mapsto \ell(y,f(x)) : f \in \mathcal{F}\}$. Then $\hat{\mathbb{E}}_n g = \hat{R}_n(f)$ and $\mathbb{E} g = R(f)$, so the lemma gives

$$
R(\hat{f}_n) \leq \inf_{f \in \mathcal{F}} R(f) + 2\,\mathrm{Rd}_n(\ell \circ \mathcal{F}) + \mathcal{O}\!\left(\tfrac{1}{\sqrt{n}}\right).
$$

We want to compute $\mathrm{Rd}_n(\ell \circ \mathcal{F})$; the contraction lemma reduces this to $\mathrm{Rd}_n(\mathcal{F})$.

### Contraction Inequality

:::{prf:lemma} Contraction (Ledoux–Talagrand)
:label: lem-contraction

Let $\phi_1, \ldots, \phi_n : \mathbb{R} \to \mathbb{R}$ be $L$-Lipschitz with $\phi_i(0) = 0$. Then

$$
\mathbb{E}\sup_{g \in G} \left|\frac{1}{n}\sum_{i=1}^n \sigma_i \phi_i(g(z_i))\right|
\leq L\,\mathbb{E}\sup_{g \in G} \left|\frac{1}{n}\sum_{i=1}^n \sigma_i g(z_i)\right|.
$$

In other words, $\mathrm{Rd}_n(\phi \circ G) \leq L\,\mathrm{Rd}_n(G)$.
:::

:::{prf:proof}
By induction over the number of functions $\phi_i$ (the argument does not linearize). We give the base step ($n=1$) to illustrate the mechanism.

For $n=1$, the expression is $\mathbb{E}_\sigma |\sigma_1 \phi_1(g(z_1))|_{\sup_{g}} = \sup_g |\phi_1(g(z_1))|$. We need to compare this to $\sup_g |g(z_1)|$. Since $|\phi_1(t)| = |\phi_1(t) - \phi_1(0)| \leq L|t|$ by the Lipschitz condition and $\phi_1(0)=0$, the bound $\sup_g |\phi_1(g(z_1))| \leq L\sup_g |g(z_1)|$ follows.

The inductive step replaces the contribution from one coordinate at a time, using the fact that for a fixed supremum the sign flip $\sigma_i \to -\sigma_i$ only changes one term, and the Lipschitz condition bounds the resulting change. $\square$
:::

**Application.** Assume $\ell$ is $L$-Lipschitz in its second argument. Using the triangle inequality to introduce the reference $\ell(y_i, 0)$ (which handles the $\phi_i(0)=0$ requirement):

$$
\mathrm{Rd}_n(\ell \circ \mathcal{F})
\leq \frac{1}{\sqrt{n}}\,\mathbb{E}\bigl[\ell(y,0)^2\bigr]^{1/2} + L\,\mathrm{Rd}_n(\mathcal{F}).
$$

The first term is $\mathcal{O}(1/\sqrt{n})$ (for square-integrable $y$). So the generalization gap is bounded by $\mathrm{Rd}_n(\mathcal{F})$ up to $\mathcal{O}(1/\sqrt{n})$ corrections.

## Rademacher Complexity of Two-Layer Neural Networks

We now compute $\mathrm{Rd}_n(\mathcal{F})$ for a two-layer network class, following {cite}`bartlett1996valid`.

### Setup

A two-layer network with $m$ hidden units and activation $\sigma$ has the form

$$
f(x; \theta) = \frac{1}{m}\sum_{k=1}^m a_k\,\sigma(w_k^\top x),
\qquad \theta = \{(a_k, w_k)\}_{k \leq m}.
$$

We constrain the parameters by

$$
w_k \in B_2^d(C) \quad (\|w_k\|_2 \leq C), \qquad a \in B_1^m(r_0 \cdot m) \quad \left(\frac{1}{m}\sum_k |a_k| \leq r_0\right).
$$

Call this class $\mathcal{F}_m(C, r_0)$. The bound on $a$ in $\ell^1$ (rather than $\ell^2$) is the key structural choice; it limits the total variation of the output layer.

### Computation of $\mathrm{Rd}_n(\mathcal{F}_m(C, r_0))$

$$
\mathrm{Rd}_n(\mathcal{F}) = \mathbb{E}_{x,\sigma} \sup_{a, w} \left|\frac{1}{n}\sum_{i=1}^n \sigma_i \cdot \frac{1}{m}\sum_{k=1}^m a_k\,\sigma(w_k^\top x_i)\right|.
$$

Interchanging the sums and bounding by $\ell^1$–$\ell^\infty$ duality:

$$
\frac{1}{m}\left|\sum_k a_k \left(\frac{1}{n}\sum_i \sigma_i\,\sigma(w_k^\top x_i)\right)\right|
\leq \frac{1}{m}\|a\|_1 \cdot \max_k \left|\frac{1}{n}\sum_i \sigma_i\,\sigma(w_k^\top x_i)\right|
\leq r_0 \max_k \left|\frac{1}{n}\sum_i \sigma_i\,\sigma(w_k^\top x_i)\right|.
$$

Taking supremum over $a$ at cost $r_0$, and then supremum over all $w$ collapses the max over $k$:

$$
\mathrm{Rd}_n(\mathcal{F}) \leq r_0\,\mathbb{E}_\sigma \sup_{w:\|w\|_2 \leq C} \left|\frac{1}{n}\sum_{i=1}^n \sigma_i\,\sigma(w^\top x_i)\right|.
$$

Applying the contraction inequality with the $L$-Lipschitz activation $\sigma$ (e.g. $L=1$ for ReLU):

$$
\leq L r_0\,\mathbb{E}_\sigma \sup_{\|w\|_2 \leq C} \left|\frac{1}{n}\sum_{i=1}^n \sigma_i\,w^\top x_i\right|
= L r_0\,\mathbb{E}_\sigma \sup_{\|w\|_2 \leq C} \left|\left\langle w,\, \frac{1}{n}\sum_i \sigma_i x_i \right\rangle\right|.
$$

The sup over $\|w\|_2 \leq C$ of a linear form is $C$ times the $\ell^2$ norm of the argument:

$$
= L r_0 C\,\mathbb{E}_\sigma \left\|\frac{1}{n}\sum_i \sigma_i x_i\right\|_2.
$$

By Jensen's inequality (concavity of the square root):

$$
\leq L r_0 C \left(\mathbb{E}\left\|\frac{1}{n}\sum_i \sigma_i x_i\right\|_2^2\right)^{1/2}
= \frac{L r_0 C}{\sqrt{n}} \left(\mathbb{E}\|x\|_2^2\right)^{1/2}.
$$

(The last step uses that $\sigma_i$ are independent mean-zero, so cross-terms vanish.) Assuming $\mathbb{E}\|x\|_2^2 \leq C' d$, we obtain the **boxed result**:

$$
\boxed{\mathrm{Rd}_n(\mathcal{F}_m(C, r_0)) \leq \tilde{C}\,r_0\,C\sqrt{\frac{d}{n}}.}
$$

Crucially, this bound is **independent of $m$** (the number of hidden units). Plugging back into the generalization bound:

$$
R(\hat{f}_n) \leq \inf_{f \in \mathcal{F}_m(r_0)} R(f) + C\,r_0\sqrt{\frac{d}{n}} + \frac{C}{\sqrt{n}}.
$$

This is the result of {cite}`bartlett1996valid`.

:::{admonition} Remark 1: Approximation–Generalization Trade-off
:class: note

Increasing $m$ (more neurons) does not worsen the generalization bound, but it can improve the **approximation error** $\inf_{f \in \mathcal{F}_m(r_0)} R(f) - \inf_f R(f)$ by making the class richer. This is the classical trade-off, but it is controlled by $r_0$ (the $\ell^1$ norm of the output weights), not by $m$.
:::

:::{admonition} Remark 2: Overfitting and the Failure of the Bound
:class: warning

**The bound is not useful when $\mathcal{F}$ can overfit the data.** Consider the square loss under the fixed-design model $y_i = f_*(x_i) + \sigma g_i$ with $x_1, \ldots, x_n$ fixed and $g_i \overset{\text{iid}}{\sim} \mathcal{N}(0,1)$.

If $\hat{f}_n$ interpolates the data (i.e. $\hat{f}_n(x_i) = y_i$ for all $i$), then $\hat{R}_n(\hat{f}_n) = 0$, while

$$
R(\hat{f}_n) = \frac{1}{n}\sum_i \mathbb{E}_g\bigl[(y_i - \hat{f}_n(x_i))^2\bigr]
= \frac{1}{n}\|\mathbf{f}_*(X) - \hat{f}_n(X)\|^2 + \sigma^2.
$$

By the triangle inequality, $R(\hat{f}_n) - R(f_*) = \frac{1}{n}\|f_*(X) - \hat{f}_n(X)\|^2 \geq c\sigma^2$ for some constant $c > 0$ (the interpolating function has to "reach" each $y_i = f_*(x_i) + \sigma g_i$, moving distance $\sim \sigma$ from $f_*$ at each point). Moreover,

$$
\mathrm{Rd}_n(\mathcal{F}) = \mathbb{E}_\sigma \sup_{f \in \mathcal{F}} \frac{1}{n}\sum_i \sigma_i f(x_i) \geq c\sigma,
$$

so the Rademacher bound does not vanish as $n \to \infty$ when $\mathcal{F}$ is rich enough to interpolate. Overfitting $\Rightarrow$ $\mathrm{Rd}_n(\mathcal{F})$ does not go to 0.
:::

## Abstract Detour: Banach-Space Approximation and Barron Spaces

The Rademacher bound above controls generalization given $r_0$ but leaves open: how large must $m$ be to make $\inf_{f \in \mathcal{F}_m(r_0)} R(f)$ small? The following abstract framework, due to Maurey and Pisier, gives a clean answer.

### Convex Hulls in Banach Spaces

Let $(X, \|\cdot\|_X)$ be a Banach space, $G \subseteq X$ compact and **centrosymmetric** ($g \in G \Rightarrow -g \in G$). Define

$$
\mathrm{conv}(G) = \left\{\sum_{i=1}^n a_i g_i : a_i \geq 0,\; \sum_i a_i = 1,\; g_i \in G,\; n \in \mathbb{N}\right\} = \bigcap_{\substack{F \supseteq G \\ F \text{ halfspace}}} F,
$$

the closed convex hull of $G$. Define also

$$
B_G = \overline{\mathrm{conv}(G)}, \qquad
\tilde{B}_G = \left\{x = \int_G g\,\mu(dg) : \mu \text{ prob. measure on } G\right\}.
$$

:::{prf:lemma}
:label: lem-BG

$B_G = \tilde{B}_G$.
:::

:::{prf:proof}

$(\subseteq)$ Let $x \in B_G$. By definition of the closure of the convex hull, there exist $x_n = \sum_{i=1}^{m(n)} a_i^{(n)} g_i^{(n)} \to x$ with $a_i^{(n)} \geq 0$, $\sum_i a_i^{(n)} = 1$, $g_i^{(n)} \in G$. Define the discrete measures $\mu_n = \sum_{i=1}^{m(n)} a_i^{(n)} \delta_{g_i^{(n)}}$; then $x_n = \int g\,\mu_n(dg)$.

By Prokhorov's theorem (since $G$ is compact metric), the sequence $(\mu_n)$ is tight and has a weakly convergent subsequence $\mu_{n_k} \Rightarrow \mu$. For any $\lambda \in X^*$ (continuous linear functional),

$$
\lambda(x_{n_k}) = \int \lambda(g)\,\mu_{n_k}(dg) \to \int \lambda(g)\,\mu(dg),
$$

so $x = \int g\,\mu(dg) \in \tilde{B}_G$.

$(\supseteq)$ Let $x \in \tilde{B}_G$, so $x = \int g\,\mu(dg)$. Given $\varepsilon > 0$, let $\mathcal{N}_\varepsilon = \{g_1, \ldots, g_N\}$ be an $\varepsilon$-net of $G$, and partition $G = R_1 \cup \cdots \cup R_N$ with $R_i \subseteq B(g_i, \varepsilon)$. Set $a_i = \mu(R_i)$ and $x_\varepsilon = \sum_{i=1}^N a_i g_i \in \mathrm{conv}(G)$. Then

$$
\|x_\varepsilon - x\|_X = \left\|\sum_{i=1}^N \int_{R_i} (g_i - g)\,\mu(dg)\right\|_X \leq \sum_i \int_{R_i} \|g_i - g\|_X\,\mu(dg) \leq \varepsilon. \qquad \square
$$
:::

### The Atomic Norm

The **atomic norm** induced by $G$ is

$$
\|x\|_G = \inf\{\lambda > 0 : x \in B_G(\lambda)\} = \inf\{|\mu| : x = \int g\,\mu(dg),\; \mu \text{ not normalized}\},
$$

where $|\mu| = \mu(G)$ is the total mass.

**Example 1.** $X = \mathbb{R}^n$, $G = \{\pm e_1, \ldots, \pm e_n\}$. Then $B_G = \{x : \|x\|_1 \leq 1\}$, so $\|x\|_G = \|x\|_1$ (the $\ell^1$ norm).

**Example 2.** $X = \mathrm{Sym}_n \subseteq \mathbb{R}^{n \times n}$, $G = \{\pm uu^\top : \|u\|_2 = 1\}$. Then $B_G = \{X : \sum_i |\lambda_i(X)| \leq 1\}$ (unit ball in the Schatten-1 / nuclear norm), so $\|X\|_G = \|X\|_*$ (the nuclear norm).

### Maurey–Pisier Approximation Theorem

:::{prf:theorem} Maurey–Pisier
:label: thm-maurey-pisier

Assume $X$ is a **Hilbert space**. Then for every $f \in X$ there exist $g_1, \ldots, g_m \in G$ and $a \in \mathbb{R}$ such that

$$
\left\|f - \frac{a}{m}\sum_{i=1}^m g_i\right\|_X \leq \frac{B\|f\|_G}{\sqrt{m}},
$$

where $B = \sup_{g \in G} \|g\|_X$.
:::

The rate $1/\sqrt{m}$ is the **Monte Carlo rate**: approximating an integral by a sample average.

:::{prf:proof}

Since $f \in X = \overline{\mathrm{conv}(G)} \cdot \|f\|_G$ (after rescaling), by Lemma {ref}`lem-BG` we may write $f = \int_G g\,\mu(dg)$ for a positive measure $\mu$ with $|\mu| = (1+\varepsilon)\|f\|_G$ (for any $\varepsilon > 0$). Set $a = |\mu|$ and $\hat{\mu} = \mu/|\mu|$ (normalized to a probability measure). Then

$$
f = a\int_G g\,\hat{\mu}(dg) = \mathbb{E}_{g \sim \hat{\mu}}[a\,g].
$$

Draw $g_1, \ldots, g_m \overset{\text{iid}}{\sim} \hat{\mu}$ and set $\hat{f} = \frac{a}{m}\sum_{i=1}^m g_i$. Then $\mathbb{E}[\hat{f}] = f$, and

$$
\mathbb{E}\left[\|f - \hat{f}\|_X^2\right]
= \mathbb{E}\left\|\frac{a}{m}\sum_{i=1}^m (g_i - \mathbb{E}[g_i])\right\|_X^2
= \frac{a^2}{m^2}\sum_{i=1}^m \mathbb{E}\|g_i - \mathbb{E}[g_i]\|_X^2
\leq \frac{a^2}{m}\,\mathbb{E}\|g_i\|_X^2
\leq \frac{a^2 B^2}{m}.
$$

Since $\mathbb{E}[\hat{f}] = f$, the expectation of $\|f - \hat{f}\|_X^2$ being small implies the **existence** of a particular realization $g_1, \ldots, g_m$ achieving the bound. Taking $\varepsilon \to 0$ gives $a \to \|f\|_G$, completing the proof. $\square$
:::

:::{note}
The Hilbert space assumption is essential: the proof uses that $\mathbb{E}\|g_i - \mathbb{E}[g_i]\|^2 \leq \mathbb{E}\|g_i\|^2 = B^2$, which holds in any Hilbert space (or more generally in spaces of Rademacher type 2). For Banach spaces of type $p > 1$ but not Hilbert, the same conclusion holds with a worse dependence on the geometry.
:::

### Application: Barron Spaces and Two-Layer Networks

Set $X = L^2(\mathbb{R}^d; \mathbb{P})$ (the Hilbert space of square-integrable functions under the data distribution), and let

$$
G = \{\pm\sigma(\langle w, \cdot\rangle) : \|w\|_2 \leq 1\} = G_+ \cup G_-.
$$

The induced atomic norm is the **Barron norm**:

$$
\|f\|_\sigma = \inf\left\{|\nu| : f(\cdot) = \int_{G_+} \sigma(\langle w, \cdot\rangle)\,\nu_+(dw) - \int_{G_+} \sigma(\langle w, \cdot\rangle)\,\nu_-(dw)\right\},
$$

where $\nu = \nu_+ - \nu_-$ is a signed measure on $G_+ = \{w : \|w\|_2 \leq 1\}$. Equivalently,

$$
\|f\|_\sigma = \inf\left\{|\nu| : f(\cdot) = \int_{G_+} \sigma(\langle w, \cdot\rangle)\,\nu(dw),\; \nu \text{ signed measure on } G_+\right\}.
$$

The class $\mathcal{F}_m(r_0)$ defined earlier corresponds to elements of $B_G(r_0)$ (the ball of Barron norm $\leq r_0$) approximated by $m$-neuron networks.

:::{prf:corollary} Barron (1993)
:label: cor-barron

Let $f_* : \mathbb{R}^d \to \mathbb{R}$ with $\|f_*\|_\sigma < \infty$. Then for any $m \geq 1$,

$$
\inf_{f \in \mathcal{F}_m(r_0)} \|f_* - f\|_{L^2(\mathbb{P})} \leq \frac{C\|f_*\|_\sigma}{\sqrt{m}},
\qquad r_0 = \|f_*\|_\sigma.
$$
:::

Combining with the Rademacher bound from the previous section, the total excess risk of the ERM estimator over $\mathcal{F}_m(r_0)$ satisfies

$$
R(\hat{f}_n) - R(f_*) \lesssim \underbrace{\frac{C\|f_*\|_\sigma}{\sqrt{m}}}_{\text{approx. error}} + \underbrace{\|f_*\|_\sigma\cdot C\sqrt{\frac{d}{n}}}_{\text{gen. error}}.
$$

Balancing these two terms by choosing $m \asymp n/d$ gives $R(\hat{f}_n) - R(f_*) = \mathcal{O}(1/\sqrt{n/d})$, a rate independent of the ambient dimension $d$ (up to the implicit dependence in $\|f_*\|_\sigma$).

**Physical interpretation.** The Barron norm measures the "spectral complexity" of $f_*$ as a function of its Fourier/harmonic representation in the weight space. Functions that can be expressed as superpositions of neurons with bounded total variation in the weight space have small Barron norm. This is precisely the class for which shallow networks overcome the curse of dimensionality.
