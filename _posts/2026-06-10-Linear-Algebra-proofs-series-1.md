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

> When is $X^TX$ invertible? Are there any specific properties of $X$ that make $X^TX$ invertible?

In this post, we will prove this result from first principles using the concept of the **kernel (null space)** of a matrix.

---

## II. Proof Strategy

Rather than reasoning directly about invertibility, we will first prove the following result:

$$
\operatorname{Ker}(X)
=
\operatorname{Ker}(X^TX).
$$

At first glance, this may not seem to answer our original question. However, this equality turns out to be the key observation.

**Once we know that $X$ and $X^TX$ have exactly the same kernel, determining whether $X^TX$ is invertible, reduces to whether the kernel of $X$ is trivial.**

---

## III. Proof

Now let's **compare the kernels** of the two matrices to proove they are equal.

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
\boxed{
\mathrm{Ker}(X)
=
\mathrm{Ker}(X^TX).
}
$$


----

## IV. Does Full Rank X alone Make $X^TX$ Invertible?

From the previous section, we have established that

$$
\operatorname{Ker}(X)
=
\operatorname{Ker}(X^TX).
$$

In ML, a **common statement we encounter is 'if $X$ is full rank, then $X^TX$ is invertible'**. Almost, as if to say, if $X$ is full rank, its kernel is trivial, and since $X$ and $X^TX$ share the same kernel, $X^TX$ also has a trivial kernel. Therfore, it is invertible. 

**However, not all full rank matrices have a trivial kernel** (eg.wide rectangular full rank). **And not all matrices with a trivial kernel are invertible** (eg. tall rectangular matric with full rank is not invertible despite having a trivial kernel).

**Thus, $X$ being full rank alone, does not guarnatee a trivial kernel and invertibility! In addition to the rank of a matrix, whether or not a kernel is trivial and invertible, depends on the shape of the matrix.**.

Let us examine the three possible cases.

### Case 1: Square matrices

Suppose

$$
X\in\mathbb R^{n\times n},
$$

and $X$ is full rank.

Then

$$
\operatorname{rank}(X)=n.
$$

By the Rank-Nullity Theorem,

$$
\dim(\operatorname{Ker}(X))
=
n-n
=
0.
$$

Hence,

$$
\operatorname{Ker}(X)
=
\operatorname{Ker}(X^TX)
=
\{0\}.
$$

**Since both $X$ and $X^TX$ are square matrices (1-1 mapping) with trivial kernels, both are invertible**.

---

### Case 2: Tall rectangular matrices

Suppose

$$
X\in\mathbb R^{m\times n},
\qquad m>n.
$$

If $X$ is full rank, then

$$
\operatorname{rank}(X)=n.
$$

Again, by the Rank-Nullity Theorem,

$$
\dim(\operatorname{Ker}(X))
=
n-n
=
0.
$$

Hence,

$$
\operatorname{Ker}(X)
=
\operatorname{Ker}(X^TX)
=
\{0\}.
$$

Notice, however, that **$X$ itself is not invertible, despite being full rank**. The reason is that $X$ is tall rectangular, so there exist output vectors

$$
b\notin\operatorname{Col}(X),
$$

for which

$$
Xx=b
$$

has no solution in the input space and therefore cannot be inverted.

**However, unlike its tall rectangular full-rank parent X,**

$$
X^TX
\in
\mathbb R^{n\times n}
$$

**is square. And, since it also has a trivial kernel, $X^TX$ is invertible.**

---

### Case 3: Wide rectangular matrices

Suppose

$$
X\in\mathbb R^{m\times n},
\qquad m<n.
$$

If $X$ is full rank, then

$$
\operatorname{rank}(X)=m.
$$

Applying the Rank-Nullity Theorem,

$$
\dim(\operatorname{Ker}(X))
=
n-m
>
0.
$$

Therefore,

$$
\operatorname{Ker}(X)
=
\operatorname{Ker}(X^TX)
\neq
\{0\}.
$$

Again, **$X$ itself is not invertible**, despite being full rank.

Although every output vector is reachable, different input vectors can produce the same output because the kernel is nontrivial.

Since $X^TX$ inherits this same nontrivial kernel, it is also **not invertible**.

---

The three cases can be summarized as follows.

| Shape of $X$ | Full rank | $\operatorname{Ker}(X)$ | Is $X$ invertible? | Is $X^TX$ invertible? |
|:---|:---:|:---:|:---:|:---:|
| Square | ✓ | $\{0\}$ | ✓ | ✓ |
| Tall | ✓ | $\{0\}$ | ✗ | ✓ |
| Wide | ✓ | Nontrivial | ✗ | ✗ |

We therefore conclude that **a full rank matrix does not automatically imply that $X^TX$ is invertible**.

Rather, the **deciding factor is whether the common kernel of $X$ and $X^TX$ is trivial**.

**This occurs when $X$ is either**

- **a full rank square matrix**, or
- **a full rank tall matrix.**

In contrast, a full rank wide matrix always has a nontrivial kernel, so neither $X$ nor $X^TX$ is invertible.






---

Returning to the Ordinary Least Squares solution,

$$
\theta^*
=
(X^TX)^{-1}X^Ty,
$$

we now see why the inverse exists.

As long as the design matrix $X$ has **full column rank**, the matrix

$$
X^TX
$$

is **square, has a trivial kernel, and is therefore invertible**.

This is precisely what justifies the appearance of

$$
(X^TX)^{-1}
$$

in the closed-form solution for Ordinary Least Squares.

If X were wide and full rank, it would still have a non trivial kernel, not be invertible and result in an
$X^TX$ which, although square, would be rank deficit with the same non trivial kernel and therefore not invertible either.

----



