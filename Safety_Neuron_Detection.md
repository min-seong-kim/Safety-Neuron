# Safety Neuron Detection — Technical Summary

This document provides a structured summary of the safety neuron detection methodology described in the referenced paper. It includes the foundational formulation, accelerated detection (parallel computation), and detailed derivations for Feed-Forward Networks (FFN) and Self-Attention networks.

---

# 1. Foundational Safety Neuron Detection

## 1.1 Neuron Definition

- Let the \( l \)-th neuron in layer \( i \) be denoted as:
  \[
  N_i^{(l)}
  \]
- When processing a harmful query \( x \), the intermediate representation after layer \( i \) is:
  \[
  h_i(x)
  \]

---

## 1.2 Neuron Importance (Equation 1)

The importance of neuron \( N_i^{(l)} \) is defined as the change in the intermediate representation when that neuron is removed:

\[
\text{Imp}(N_i^{(l)}|x)
= \| h_i(x) - h^{\backslash N_i^{(l)}}_i(x) \|_2
\]

where:

- \( h^{\backslash N_i^{(l)}}_i(x) \): representation after deactivating neuron \( N_i^{(l)} \)

---

## 1.3 Activated Neuron Set for a Query (Equation 2)

A neuron is considered *activated* for harmful query \( x \) if:

\[
N_x = 
\left\{
N_i^{(l)} 
\ \bigg| \
\| h_i(x) - h^{\backslash N_i^{(l)}}_i(x) \|_2 \ge \varepsilon
\right\}
\]

where:

- \( \varepsilon \): predefined threshold

---

## 1.4 Safety Neuron Set Across Multiple Queries (Equation 3)

Given a corpus of harmful queries \( X \), safety neurons are defined as:

\[
N_{\text{safe}}
= 
\bigcap_{x \in X} N_x
\]

These neurons are *consistently important* across all harmful queries.

---

# 2. Accelerated Safety Neuron Detection

Sequentially deactivating every neuron is computationally expensive.  
To overcome this, the paper adopts a *parallel detection approach* inspired by Zhao et al. (2024b).

Acceleration is achieved by:
- Rewriting neuron-removal effects using algebraic transformations
- Constructing mask matrices for FFN
- Using rank-1 update analysis for Attention (Q/K)

---

# 3. Feed-Forward Network (FFN)

## 3.1 FFN Structure (Equation 7)

For an input sequence representation \( x \in \mathbb{R}^{l \times d_{\text{model}}} \):

\[
\text{FFN}(x)
=
\text{SiLU}(x W_{\text{gate}})
\cdot W_{\text{up}}
\cdot W_{\text{down}}
\]

where:
- \( W_{\text{gate}}, W_{\text{up}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{inter}}} \)
- \( W_{\text{down}} \in \mathbb{R}^{d_{\text{inter}} \times d_{\text{model}}} \)

Let:
\[
h_{\text{ffn}}(x) \in \mathbb{R}^{l \times d_{\text{inter}}}
\]
be the intermediate embedding before the down-projection.

---

## 3.2 Neuron Removal Effect (Equation 8)

Removing the \( k \)-th neuron of \( W_{\text{up}} \) yields:

\[
\text{Imp}(W_{\text{up}}[:,k] | x)
=
\| 
h_{\text{ffn}}(x) \cdot \text{Mask}[k] \cdot W_{\text{down}}
\|_2
\]

where:
- \( \text{Mask}[k] \) is a one-hot vector with 1 at index \( k \)

---

## 3.3 Parallel Computation via Diagonal Mask (Equation 9)

Using a diagonal mask matrix:

\[
Mask = \text{diag}(1,1,\dots,1) \in \mathbb{R}^{d_{\text{inter}} \times d_{\text{inter}}}
\]

the importance of all neurons can be computed simultaneously:

\[
\text{Imp}(W_{\text{up}}|x)
=
\|
 (h_{\text{ffn}}(x) \cdot Mask) W_{\text{down}}
\|_2
\]

This eliminates the need for per-neuron deactivation.

---

## 3.4 Equivalence of Deactivating W_up and W_down

The paper shows:

\[
W_{\text{down}}[:,k] \text{ deactivation }
\quad\Longleftrightarrow\quad
W_{\text{up}}[:,k] \text{ deactivation}
\]

Both set \( h_{\text{ffn}}(x)[k] = 0 \),  
so Eq. (9) also yields importance for \( W_{\text{down}} \).

---

# 4. Self-Attention Network

## 4.1 Attention Structure (Equation 10)

\[
\text{Attention}(x)
=
\text{Softmax}
\left(
\frac{W_Q(x) W_K(x)^T}{\sqrt{d}}
\right)
W_V(x)
\]

where:

- \( W_Q, W_K, W_V \in \mathbb{R}^{d_{\text{model}} \times d_{\text{mid}}} \)

Because \( W_V \) is *after* the softmax, its importance can also be computed using the FFN method (Eq. 9).

---

## 4.2 Effect of Removing Q Neuron (Equation 11)

Deactivating the \( k \)-th Q neuron yields:

\[
\Delta_k(x)
=
W_Q(x)[:,k] \cdot W_K(x)[k,:]
\]

- \( \Delta_k(x) \in \mathbb{R}^{l \times l} \)
- It represents a *rank-1 update* to the attention score matrix.

---

## 4.3 Approximate Importance via Softmax Change (Equation 12)

\[
\text{Imp}(W_Q[:,k] | x)
\approx
\left\|
\text{Softmax}\left(\frac{W_QW_K^T - \Delta_k}{\sqrt{d}}\right)
-
\text{Softmax}\left(\frac{W_QW_K^T}{\sqrt{d}}\right)
\right\|_2
\]

Importance is proportional to **how much the attention distribution changes** when the Q neuron is removed.

---

## 4.4 Parallel Computation of Δ(x) Tensor (Equation 13)

Reshape Q/K as:

\[
W_Q(x).resize(l, 1, d_{\text{mid}})
\]
\[
W_K(x).resize(1, l, d_{\text{mid}})
\]

Then:

\[
\Delta(x)
=
W_Q(x) \times W_K(x)
\in \mathbb{R}^{l \times l \times d_{\text{mid}}}
\]

- The \( j \)-th slice of Δ corresponds to \( \Delta_j(x) \)

---

## 4.5 Parallel Importance Computation for Q/K (Equation 14)

\[
\text{Imp}(W_Q|x)
\approx
\left\|
\text{Softmax}\left(
\frac{W_QW_K^T - \Delta(x)}{\sqrt{d}}
\right)
-
\text{Softmax}\left(
\frac{W_QW_K^T}{\sqrt{d}}
\right)
\right\|_2
\]

Since Q/K are symmetric in their contribution:

\[
\text{Imp}(W_K|x) \text{ is computed identically}
\]

---

# 5. Summary Table

| Layer Type | Neuron Removal Effect | Importance Formula | Parallelization Strategy |
|------------|-----------------------|---------------------|---------------------------|
| **FFN (W_up, W_down)** | Removes dimension in \(h_{\text{ffn}}\) | \(\|(h_{\text{ffn}} \cdot Mask) W_{\text{down}}\|_2\) | Diagonal Mask |
| **Attention Q** | Rank-1 update \( \Delta_k = Q_k K_k^T \) | Softmax change norm | Δ tensor (\(T \times T \times D\)) |
| **Attention K** | Same as Q | Same as Q | Same Δ tensor |
| **Attention V** | Linear output layer, no Softmax effect | FFN-style Eq. 9 | Mask matrix |

---

# End of Document