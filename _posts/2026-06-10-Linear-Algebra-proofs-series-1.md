---
layout: post
title: "Useful Linear Algebra Proofs Series: Why is XᵀX Invertible When X is Full Rank?"
date: 2026-06-10
---

## I. Setting the scene

One of the most common matrices encountered in machine learning appears in the closed-form solution for Ordinary Least Squares (OLS):

$$
\theta^*
=

(X^TX)^{-1}X^Ty.
$$

The inverse of $X^TX$ is what allows us to solve for the optimal parameters directly. But this raises an important question:

> When is $X^TX$ invertible in the first place?

or more specifically,

> How does $X$ being full rank imply that $X^TX$ is also full rank and invertible?

## II. When Is $X$ Full Rank?

Recall that a matrix is **full rank** when its columns are linearly independent, or no combination of the columns produces another, or the kernel (null space) of the matrix is trivial.

There are two common situations in which this occurs.

### Square matrices

If

$$
X\in\mathbb R^{n\times n},
$$

then $X$ is full rank when all of its columns are linearly independent. In this case,

$$
\mathrm{rank}(X)=n,
$$

and $X$ is invertible.

### Tall rectangular matrices

If

$$
X\in\mathbb R^{m\times n},
\qquad m>n,
$$

then $X$ can still have linearly independent columns. In this case,

$$
\mathrm{rank}(X)=n,
$$

and we say that $X$ has **full column rank**.

Although such a matrix is not square and therefore cannot itself be inverted, none of its columns are redundant.

### Wide matrices are never full column rank

If

$$
X\in\mathbb R^{m\times n},
\qquad m<n,
$$

then there are more columns than rows.

In this case, some columns must necessarily be linear combinations of others, meaning the columns cannot all be independent. Equivalently,

$$
\mathrm{rank}(X)
\le m
<n.
$$

From a geometric point of view, many different input vectors are being compressed into a smaller output space. Information is lost, so the transformation cannot be one-to-one, and therefore cannot be invertible.

Now that we understand what makes $X$ full rank, let us prove that this also makes $X^TX$ full rank.

## III. Approach and Premise

Rather than working directly with ranks, we will **compare the kernels** of the two matrices.

Recall the **basic set theory** idea:

> If every element of set $A$ belongs to set $B$, and every element of set $B$ belongs to set $A$, then the two sets are equal.

Symbolically,

$$
A\subseteq B
\quad\text{and}\quad
B\subseteq A
\quad\Longrightarrow\quad
A=B.
$$

We will apply this idea **to show:**

$$
\mathrm{Ker}(X)
=

\mathrm{Ker}(X^TX).
$$

Once we establish that the two kernels are identical, the Rank-Nullity Theorem immediately implies

$$
\mathrm{rank}(X)
=

\mathrm{rank}(X^TX).
$$


## IV. Proof

First, to show

$$
\mathrm{Ker}(X)
\subseteq
\mathrm{Ker}(X^TX)
$$

Consider a vector $v$ such that,

$$
v\in\mathrm{Ker}(X).
$$

By definition,

$$
Xv=0.
$$

Multiplying both sides by $X^T$,

$$
X^TXv
=
X^T0
$$

Gives,

$$
(X^TX)v
=
0.
$$

Therefore,

$$
v\in\mathrm{Ker}(X^TX),
$$

which proves

$$
\boxed{
\mathrm{Ker}(X)
\subseteq
\mathrm{Ker}(X^TX).
}
$$




Now to proove the other direction also holds, suppose

$$
v\in\mathrm{Ker}(X^TX).
$$

Then

$$
X^TXv=0.
$$

Multiplying from the left by $v^T$,

$$
v^TX^TXv
=
0
$$

But

$$
v^TX^TXv
=
(Xv)^T(Xv)
=
||Xv||^2.
$$

Hence

$$
||Xv||^2=0.
$$

By the positive definiteness property of norms, a squared norm can equal zero only when the vector inside it is itself zero.

$$
Xv=0.
$$

Therefore,

$$
v\in\mathrm{Ker}(X),
$$

which proves

$$
\boxed{
\mathrm{Ker}(X^TX)
\subseteq
\mathrm{Ker}(X).
}
$$

**Since each null space is contained inside the other, we have,**

$$
\mathrm{Ker}(X)
=
\mathrm{Ker}(X^TX).
$$

**Applying the Rank-Nullity Theorem:**

$$
\mathrm{rank}(A)+\mathrm{nullity}(A)
=
\text{number of columns of }A,
$$

and since both $X$ and $X^TX$ have the same number of columns, it follows that

$$
\boxed{
\mathrm{rank}(X)
=
\mathrm{rank}(X^TX)
}
$$

Therefore, if $X$ has full column rank, then $X^TX$ also has full rank.

Hence, proven.

----

Linking back to the OLS closed form solution, we see that since $X^TX$ is square and full rank, it is invertible, which justifies the appearance of $(X^TX)^{-1}$ . 

in the closed-form solution for $\theta^*$

----



