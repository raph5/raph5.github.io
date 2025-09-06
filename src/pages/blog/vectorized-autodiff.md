---
layout: ../../layouts/BaseLayout.astro
title: "Reverse vs. Vectorized Forward AD: A Performance Exploration in C"
description: "Performance comparison of reverse mode and vectorized forward mode automatic differentiation in C++. Includes benchmarks on polynomial fitting, cache-aware optimizations, and implementation details."
---

# Reverse vs. Vectorized Forward AD: A Performance Exploration in C

Automatic differentiation (AD) enables efficient and accurate gradient
computation for complex functions. While reverse mode is typically favored for
many-to-one problems like optimization due to its algorithmic efficiency, it
nevertheless often suffers from runtime overhead caused by memory allocation
and pointer indirection. This post explores whether a vectorized forward mode
can compete and shares some interesting optimization techniques.

For readers new to AD, [this video introduction](
https://youtu.be/watch?v=wG_nF1awSSY) gives an accessible overview. For a
deeper and more formal treatment, including implementation patterns, see
[Charles C. Margossian’s paper](https://arxiv.org/pdf/1811.05031).


## Implementation

I implemented both forward and reverse AD in C++. Writing an implementation in
a low level language is, in my opinion, a requirement to get somewhat relevant
results out of those micro-benchmarks.

The reverse AD implementation uses a tape-based design as described in [Charles
C. Margossian’s paper](https://arxiv.org/pdf/1811.05031). On the other hand,
forward AD uses a `struct {float value, float grad[GRADLEN]}` to propagate full
gradients in a single pass. This way of implementing forward AD is known as
vectorized forward AD.

The code is minimal (~300 lines per mode), designed to reflect core patterns of
open-source implementations. It's available here: [raph5/vectorized-autodiff](
https://github.com/raph5/vectorized-autodiff).


## Benchmark

I used a polynomial fitting task: minimize the L2 distance between a polynomial
and the function exp(1 / x²) on a discretized interval. This involved gradient
descent on the polynomial coefficients.

This problem is simple but sufficient to test gradient computation performance
as polynomial degree increases.

The benchmarks were compiled with clang (-O2) and ran on a MacBook Air M2. SIMD
usage was verified in assembly output. Runtime is averaged over 10 runs using
the `clock()` function from `time.h`.


## Initial Results

![results of the first benchmark](/images/chart_1.svg)

Performance was measured as a function of gradient size (i.e., the degree of
the polynomial):
- Forward vectorized AD outperforms reverse AD for gradients smaller than
  ~120-150.
- Above ~280 dimensions, forward AD slows dramatically.

This behavior suggests cache pressure. The forward mode stores a full gradient
vector at every intermediate node, which quickly exceeds L1/L2 capacity and
causes frequent cache misses.


## Chunked Vectorization

To address this, I modified the forward AD to split the gradient computation
across multiple passes. Instead of `{float value, grad[GRADLEN]}`, we compute
in blocks of size α: `{float value, grad[ALPHA]}`.

This dramatically improves cache temporal locality at the cost of
recomputing the function value multiple times -- what’s known as duplication of
the primal work.

I tested various α values (chart below) and found a sweet spot around 64.

![results of the second benchmark](/images/chart_2.svg)


## Updated Performance

Here are the performances of the chunked forward AD (in green) for α = 64

![results of the second benchmark](/images/chart_3.svg)

As you can see the chunked version performs way better. Though 2× slower than
reverse AD for gradients of size 500, forward AD is now viable choice for this
specific optimization problem.


## Parallelization Attempt

I attempted to parallelize the forward AD over gradient chunks using pthreads.
No speedup was observed. This may be due to inherent problems with the
parallelization approach or more probably to poor implementation choices. I
think there might be a cache issue once again as threads probably share the
same cache.


## Conclusion

First I should emphasize that the results here are not generalizable as they
were obtained on one specific micro-benchmark task on one specific computer.

That said I think this is an interesting little experiment that demonstrates
the relevance of forward AD when dealing with gradients of size 10^2 to 10^3.
It also shows the importance of considering cache pressure when implementing
vectorized forward AD.

An interesting follow up experiment would be to compare the performances of
those forward and reverse AD implementation on different problems running on
different architectures against the performances of real AD libraries. A tool
like [gradbench](https://github.com/gradbench/gradbench) is perfect for this
task.


## Resources

- [Github Repo Containing The Implementation And Benchmark Code](https://github.com/raph5/vectorized-autodiff)

- [Efficient GPU Implementation of Automatic Differentiation for Computational Fluid Dynamics](https://digitalcommons.odu.edu/cgi/viewcontent.cgi?article=1245&context=computerscience_fac_pubs)

- [A Review of Automatic Differentiation and its Efficient Implementation](https://arxiv.org/pdf/1811.05031).

- [Automatic Differentiation in Machine Learning: a Survey](https://arxiv.org/pdf/1502.05767)

- [Vector Forward Mode Automatic Differentiation on SIMD/SIMT architectures](https://jnamaral.github.io/icpp20/slides/Hueckelheim_Vector.pdf)

- [Gradbench](https://github.com/gradbench/gradbench)
