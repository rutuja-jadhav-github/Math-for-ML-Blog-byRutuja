---
layout: post
title: Closed form solution to the OLS Regression
date: 2026-05-20
---

This post is continuation of an earlier derivation - 'Using Maximum Likelihood Estimation to Derive the OLS Loss function'. It shouldn't matter if you already are/ aren't familiar with the probabilistic predeccessors of this closed form solution. The point where we will pick up from now is often how it's taught in class anyway.

Based on all the classes where I learnt this closed form solution, there are **three main entry points up until here - MLE (link), ERM from the statistical learning theory (discussed here), and linear algebra with geometry** (I'll probably do a seaparate post on this later, but if that route interests you, I would highly  recommend watching Strang's lecture on Orthogonal Projections and Least Squares on the MIT open courses).

## I. Assumptions

**IID (Independent and Identically Distributed).** Most statistical learning theory assumes our observations are IID. *Independent* means each observation contributes genuinely new information, while *identically distributed* means all observations come from the same underlying population. For example, if you survey 1,000 people but 500 belong to just 100 families, knowing something about one family member tells you something about another—the observations are no longer truly independent. Similarly, if half your salary data comes from 2010 and half from 2024, the observations are not identically distributed because they are drawn from different populations.

**The Law of Large Numbers (LLN).** The practical importance of IID is that it allows the Law of Large Numbers to apply. As the sample size grows, quantities computed on the training set converge to their true population values. In machine learning, this is the bridge between training loss and real-world performance: minimizing empirical loss becomes a reasonable proxy for minimizing the loss we would observe on unseen data drawn from the same population.

**Gaussian Errors.** This assumption is far more subtle from the ERM persecptive than the MLE route for this derivation. Nevertheless, it still exists and is important. If your loss has its origins in sum of squared distances (as opposed to absolute distances, for example), then by deafault you have assumed a gaussian distribution of errors,even if you don't see it directly in this post. (The MLE to OLS post makes this more obious).

**Full rank feature matrix.** We assume the columns of $X$ are linearly independent. Equivalently, no feature can be written as an exact linear combination of the others.



## II. Empirical Risk Minimizer (ERM) for Squared Loss

In the previous post, we arrived at:

$$
\theta^*
= \arg\min_\theta \sum_{i=1}^n (y_i - x_i^\top \theta)^2
$$

using the Maximum Likelihood Estimation framework.

Now we re-derive the same objective from the Empirical Risk Minimization (ERM) perspective in statistical learning theory.



In ERM, we define a loss function over a finite dataset and choose parameters that minimize the average loss.

For squared loss, the empirical risk is:

$$
R(\theta)
=
\frac{1}{n} \sum_{i=1}^n (y_i - x_i^\top \theta)^2
$$

where:
- $n$ is the number of training samples  
- $y_i$ is the observed output (true value) for the $i$-th data point  
- $x_i$ is the feature vector for the $i$-th data point  
- $x_i^\top \theta$ is the model’s predicted output for the $i$-th data point

By definition of empirical risk minimization, an empirical risk minimizer is the parameter that minimizes this average loss over the dataset. In other words, it is the value of $\theta$ that makes the empirical risk as small as possible:

$$
\theta^* = \arg\min_\theta R(\theta)
$$

Substituting the definition of $R(\theta)$:

$$
\theta^*
=
\arg\min_\theta
\frac{1}{n} \sum_{i=1}^n (y_i - x_i^\top \theta)^2
$$

Since $\frac{1}{n}$ does not depend on $\theta$, it does not affect the minimizer. So this simplifies to:

$$
\theta^*
=
\arg\min_\theta
\sum_{i=1}^n (y_i - x_i^\top \theta)^2
$$

This is exactly the same optimization problem we obtained from maximum likelihood estimation.



## III. Defining the Objective Loss Function in Matrix notation

Notice that a sum of squared components can be written as the dot product of a vector with itself:

$$
m_1^2 + m_2^2 + \cdots + m_n^2
=
M^\top M
=
\|M\|^2
$$

where

$$
M =
\begin{bmatrix}
m_1 \\
m_2 \\
\vdots \\
m_n
\end{bmatrix}.
$$

If we define the residual vector as

$$
M = y - X\theta,
$$

then our objective becomes

$$
\theta^*
=
\arg\min_\theta
(y - X\theta)^\top (y - X\theta).
$$


Using the squared norm notation, we can write the objective even more compactly as:

$$
\theta^*
=
\arg\min_\theta
\|X\theta - y\|^2
$$

**Let's give this objective a name: Loss( $\theta$ )= F( $\theta$ )**

$$
F(\theta) = \|X\theta - y\|^2
$$

Our goal is now simple: find the value of $\theta$ that minimizes $F(\theta)$.

## IV. Convexity Detour to estbalish the existence of a Global Minima

Before jumping into calculus, it is worth appreciating that this is a particularly well-behaved optimization problem. The loss function is convex.

Why does that matter? Convexity is one of the reasons classical machine learning is so pleasant. A convex function has no local minima to trap us—only a single global minimum.

One way to verify convexity is through the Hessian of the loss function (the matrix of second derivatives):

$$
\nabla^2 F(\theta) = 2X^\top X
$$

Notice that this expression does not depend on $\theta$ at all. Moreover, for any vector $v$,

$$
v^\top (2X^\top X) v
=
2(Xv)^\top(Xv)
=
2\|Xv\|^2
\ge 0
$$

which means the Hessian is positive semidefinite.

By the first-order characterization of convexity (in a convex function, the tangent never overshoots the function— the curve always lies above its tangents), this tells us that $F(\theta)$ is convex. Geometrically, we are working with a nice bowl-shaped surface that has a global minima. 

So the Hessian is positive semidefinite.

But remember our full-rank assumption: the columns of $X$ are linearly independent. This means $Xv = 0$ only when $v = 0$.

Therefore,

$$
v^\top (2X^\top X) v > 0
\qquad \forall v \neq 0,
$$

which means the Hessian is actually positive definite (not just semi definite).

**This is an important upgrade. Positive semidefiniteness guarantees convexity, but positive definiteness guarantees strict convexity. In practical terms, it means our loss function has a single unique global minimum rather than multiple equally-good minimizers.**

So we are working with a **nice bowl-shaped function whose minimum is both global and unique.**

Where is that minimum?


## V. Computing the ERM for OLS - the closed form solution

For a differentiable convex function, the minimum occurs where the slope is zero. In other words, we need to find the value of $\theta$ for which

$$
\nabla F(\theta) = 0.
$$

So the next step is straightforward: differentiate

$$
F(\theta) = \|X\theta - y\|^2
$$

with respect to $\theta$, set the result equal to zero, and solve for the parameter vector that makes this happen.

Now we compute the gradient:

$$
\frac{dF(\theta)}{d\theta}
$$

To make differentiation easier, let’s expand the objective further.

We start with:

$$
F(\theta) = \|X\theta - y\|^2
= (X\theta - y)^\top (X\theta - y)
$$

Using the inner product form:

$$
F(\theta)
= \langle X\theta - y,\; X\theta - y \rangle
$$

Expanding:

$$
F(\theta)
= \langle X\theta, X\theta \rangle
- 2\langle X\theta, y \rangle
+ \langle y, y \rangle
$$

Writing in matrix form:

$$
F(\theta)
= (X\theta)^\top (X\theta)
- 2y^\top X\theta
+ y^\top y
$$

Now we differentiate term by term.

---

### First term

$$
(X\theta)^\top (X\theta) = \theta^\top X^\top X \theta
$$

Using:
$$
\frac{d}{d\theta} (\theta^\top A \theta) = (A + A^\top)\theta
$$

with $A = X^\top X$ (which is symmetric), we get:

$$
\frac{d}{d\theta} \theta^\top X^\top X \theta = 2X^\top X \theta
$$

---

### Second term

$$
-2y^\top X\theta
$$

Using:
$$
\frac{d}{d\theta}(v^\top \theta) = v
$$

we get:

$$
\frac{d}{d\theta}(-2y^\top X\theta) = -2X^\top y
$$

---

### Third term

$$
y^\top y
$$

This does not depend on $\theta$, so its derivative is:

$$
0
$$

---

### Final gradient

Putting everything together:

$$
\frac{dF(\theta)}{d\theta}
=
2X^\top X \theta - 2X^\top y
$$

Now we set the gradient to zero to find the minimizer:

$$
2X^\top X \theta - 2X^\top y = 0
$$


Divide both sides by 2:

$$
X^\top X \theta - X^\top y = 0
$$

Rearranging:

$$
X^\top X \theta = X^\top y
$$

Now, since we assumed $X$ has full column rank, $X^\top X$ is invertible. (Why? Short Answer: Because, Rank of X = Rank of $X^\top X$ , which means if X is full rank/ invertible, so is $X^\top X$. There is a proof for this and other nice properties about $X^\top X$  floating somewhere on this journal.)


Multiplying both sides by $(X^\top X)^{-1}$ gives:

$$
\theta = (X^\top X)^{-1} X^\top y
$$

where:
- $\theta$ is the vector of model parameters (weights/ slopes/ coefficients we want to learn)  
- $X$ is the design matrix of input features (should always be full rank, i.e., no dependent columns)
- $y$ is the vector of observed outputs (targets)

This is the closed-form solution for ordinary least squares.

















