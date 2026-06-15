---
layout: post
title: 'Gradient Descent from First Principles L-Smoothness and the Descent Lemma'
date: 2026-05-20
---

## I. Motivation for Gradient Descent

In the previous post, we derived the closed-form solution for Ordinary Least Squares:

$$
\theta^* = (X^TX)^{-1}X^Ty.
$$

For linear regression, we are fortunate: the loss function is convex and admits an analytical solution. We can solve for the optimal parameters directly. As long as the feature matrix is full rank and observations are iid, we can obtain the the slope and intercept of the line in one go.

Unfortunately, not every machine learning problem is so kind.

**Differentiable but no closed form solution** For example, logistic regression uses the sigmoid function to model probabilities. Although the resulting loss function is differentiable and convex, the nonlinearity introduced by the sigmoid prevents us from isolating $\theta$ algebraically and obtaining a closed-form solution.

**Differentiable but non convex** Neural networks take things a step further. Not only do they lack a closed-form solution, but their loss surfaces are generally non-convex, meaning there is no longer a guarantee of a unique global minimum. The critical point we land on by equating the slope to zero, could be a saddle point, or a local min/max. 

So we need an alternative. Instead of computing the parameters with minimum loss in shot, we try a bunch of differnt paramters and pick the one where loss is the the lowest.


This idea gives rise to **Gradient Descent**.


## II. Direction of Descent

Suppose we have a current estimate $\theta^{(t)}$. Our goal in ML is to always choose model params that reduce the loss. Therefore, the question becomes:

> Given where we are now, how should we choose the next parameter value $\theta^{(t+1)}$ in a way that the loss decreases?

By definition gradient points in the direction of steepest ascent, i.e. the direction in which the loss increases most rapidly. For example, if we have a loss of 5 (scalar) with the current  Since our goal is to reduce the loss, the most natural thing to do is to move in the opposite direction.

This gives the familiar Gradient Descent update rule:

$$
\theta^{(t+1)}
=
\theta^{(t)}
-
\eta \nabla F(\theta^{(t)})
$$

where:

* $\theta^{(t)}$ is the current parameter estimate,
* $\nabla F(\theta^{(t)})$ is the gradient of the loss function at that point,
* $\eta > 0$ is the learning rate, which controls the size of the step.

Instead of solving for the optimal parameters in one shot, we search for them iteratively: Initialise the parameter with a suitable value (zero works) -> compute loss -> compute gradient -> step in a direction opposite to this graient -> compute loss at this new step -> repeat until loss consistently decreases and eventually plateaus. 

At first glance, this rule seems almost too simple.

Move opposite to the gradient, and repeat.

**But does moving opposite to the gradient actually guarantee that the loss decreases?**

After all, real loss functions are not straight lines descending indefinitely. They are curved surfaces. It is entirely possible that, while the gradient initially points downhill, taking too large a step could cause us to overshoot the minimum and climb back uphill on the other side. To illustrate this, think of the loss function as a bowl along the y axix, and you are controlling the parameter knob on the x-axis. Sure, as you take X from left to right, the slope of the bowl is pointing downhill, thus decreasing the function value. But the **minute you take too large a step on X-axis, you might completely miss the bottom of the bowl (the minimia) and overshoot to the other side of the bowl sloping uphill.**

In other words, how do we know that following the negative gradient truly moves us towards the bottom of the bowl, rather than sending us past it?

To answer this, we need to understand how rapidly the slope of the function itself is allowed to change. This **leads us to one of the most important assumptions in optimization: L-smoothness**.


## III. Deriving the L-smoothness condition for gradient descent using Taylor series approximation

Before proving that Gradient Descent actually decreases the loss, we need to answer a subtle question:

> How curved is the loss surface allowed to be?

Intuitively, if the surface bends too sharply, a step that initially points downhill may overshoot the minimum and send us climbing uphill on the other side. To avoid this, we would like to place a bound on the curvature of the function so that sufficiently small steps remain well-behaved.

How do we begin?

Recall that the value of a twice-differentiable function around a point $x$ is given by its second-order Taylor expansion:

$$
f(x+h)
=

f(x)
+
\nabla f(x)^T h
+
\frac{1}{2}h^T\nabla^2f(x)h.
$$

The three terms have a simple interpretation:

* $f(x)$ gives the function value at the current point;
* $\nabla f(x)^T h$ captures the local slope;
* $h^T\nabla^2f(x)h$ accounts for the curvature of the surface.

The curvature term is precisely what can cause us to overshoot. To keep it under control, we introduce the notion of **L-smoothness**.

A function is said to be **L-smooth** if its gradient is Lipschitz continuous:

$$
||\nabla f(x)-\nabla f(y)||
\le
L||x-y||
$$

for all $x,y$.

For twice-differentiable functions, this is equivalent to requiring that the Hessian be bounded above:

$$
\nabla^2f(x)\preceq LI,
$$

or equivalently,

$$
\lambda_{\max}\big(\nabla^2f(x)\big)
\le L.
$$

In other words, the largest eigenvalue of the Hessian—which measures the maximum curvature of the surface—is never allowed to exceed $L$.

To see how this affects the Taylor expansion, consider the curvature term

$$
h^T\nabla^2f(x)h.
$$

Writing

$$
h=||h||u,
$$

where $u$ is a unit vector satisfying

$$
||u||=1,
$$

we obtain

$$
h^T\nabla^2f(x)h
=

||h||^2u^T\nabla^2f(x)u.
\tag{1}
$$

Now consider the quadratic term on the right-hand side,

$$
u^T\nabla^2f(x)u.
$$

This term measures how much the Hessian matrix stretches a unit vector $u$. Since $u$ has unit length, the quantity above is simply a scalar that tells us the curvature of the function in the direction of $u$.

Naturally, this directional curvature cannot exceed the maximum curvature attainable in any direction, which is precisely given by the largest eigenvalue of the Hessian. In other words,

$$
u^T\nabla^2f(x)u
\le
\lambda_{\max}\big(\nabla^2f(x)\big).
$$

The quantity

$$
u^T\nabla^2f(x)u
$$

is known as the **Rayleigh quotient**, and it achieves its maximum value when $u$ points along the eigenvector corresponding to the largest eigenvalue.

Substituting this into Equation (1), we obtain

$$
h^T\nabla^2f(x)h
\le
||h||^2\lambda_{\max}\big(\nabla^2f(x)\big).
\tag{2}
$$

But recall that our function is assumed to be $L$-smooth. Therefore,

$$
\lambda_{\max}\big(\nabla^2f(x)\big)
\le L.
\tag{3}
$$

Combining Equations (2) and (3),

$$
h^T\nabla^2f(x)h
\le
||h||^2\lambda_{\max}\big(\nabla^2f(x)\big)
\le
L||h||^2,
$$

which implies

$$
\frac12h^T\nabla^2f(x)h
\le
\frac{L}{2}||h||^2.
\tag{4}
$$

Returning to the second-order Taylor expansion,

$$
f(x+h)
=

f(x)
+
\nabla f(x)^T h
+
\frac12h^T\nabla^2f(x)h,
$$

and using Equation (4), we arrive at

$$
\boxed{
f(x+h)
\le
f(x)
+
\nabla f(x)^T h
+
\frac{L}{2}||h||^2.
}
$$

Thus, an $L$-smooth function can never rise above a quadratic bowl of curvature $L$. This upper bound is the key ingredient that makes Gradient Descent work.

## IV. Choosing the Step Size

Having bounded the curvature of the function, we are now ready to answer the most important question:

> Given our current point $x$, what vector $h$ should we add so that the function value decreases?

Any vector can be decomposed into two components:

* a **direction**, and
* a **magnitude**.

From the previous section, we already know the direction. Since the gradient points in the direction of steepest ascent, the direction of steepest descent must be its negative:

$$
-\nabla f(x).
$$

What remains is to determine the magnitude of the step.

From the $L$-smoothness assumption, we know that taking a step that is too large may cause us to overshoot the minimum and climb back uphill. Therefore, we introduce a positive scalar $\eta$, called the **learning rate**, and write our step vector as

$$
h
=
\eta\big(-\nabla f(x)\big)
=
-\eta\nabla f(x).
$$

Substituting this choice of $h$ into the upper bound derived in the previous section,

$$
f(x+h)
\le
f(x)
+
\nabla f(x)^T h
+
\frac{L}{2}||h||^2.
$$


gives

$$
f(x-\eta\nabla f(x))
\le
f(x)
+
\nabla f(x)^T(-\eta\nabla f(x))
+
\frac{L}{2}
\left\|
-\eta\nabla f(x)
\right\|^2.
$$

Simplifying,

$$
f(x-\eta\nabla f(x))
\le
f(x)
-
\eta\|\nabla f(x)\|^2
+
\frac{L\eta^2}{2}
\|\nabla f(x)\|^2,
$$

or equivalently,

$$
f(x-\eta\nabla f(x))
\le
f(x)
-
\eta
\left(
1-\frac{L\eta}{2}
\right)
\|\nabla f(x)\|^2.
\tag{5}
$$

Our goal is to ensure that

$$
f(x+h)
<
f(x),
$$

that is, each step should strictly decrease the function value.

Since squared norms are positive semi definite,

$$
\|\nabla f(x)\|^2 > 0,
$$

therefore, the quantity being subtracted in Equation (5) must be positive to ensure the function value at the next step decreases:

$$
\eta
\left(
1-\frac{L\eta}{2}
\right)

> 0
$$

Because $\eta>0$, this reduces to

$$
1-\frac{L\eta}{2}>0,
$$

which implies

$$
\boxed{
0<\eta<\frac{2}{L}
}.
$$

Thus, any learning rate in this interval guarantees that Gradient Descent decreases the objective.

But it doesn't tell us which step size gives us optimal descent. In fact, some of the step sizes in this range converge extremly slowly, and some just return the same starting function value.

Looking again at Equation (5), we see that the amount by which the function decreases is controlled by:

$$
\eta
\left(
1-\frac{L\eta}{2}
\right).
$$

Therefore, the optimal learning rate is the value of $\eta$ that maximizes this expression.

Differentiating with respect to $\eta$,

$$
\frac{d}{d\eta}
\left[
\eta
\left(
1-\frac{L\eta}{2}
\right)
\right]
=

1-L\eta,
$$

and setting this equal to zero yields

$$
1-L\eta=0,
$$

from which we obtain

$$
\boxed{
\eta^*
=

\frac{1}{L}
}.
$$

This choice of learning rate produces the largest guaranteed decrease in the objective at every iteration.

Substituting $\eta=1/L$ into Equation (5), we finally obtain

$$
\boxed{
f\left(
x-\frac1L\nabla f(x)
\right)
\le
f(x)
-

\frac{1}{2L}
\|\nabla f(x)\|^2.
}
$$

This is the **Gradient Descent Lemma**.

This inequality tells us how the loss changes after one gradient step. The term f(x) is the current loss value. The squared norm of the gradient measures how steep the function is at that point, so it captures how strong the local direction of descent is — if the gradient is large, we are on a steep slope and can expect a bigger decrease.

The factor involving the learning rate (magnitude of step size) expressed in terms of the smoothness constant, L, controls how effective the update is. It reflects how step size and curvature together determine how much of that steepness actually translates into progress. So overall, the decrease in loss depends on both the local slope (the gradient) and the global curvature constraint (L-smoothness), which limits how aggressively we can move. 

