# Exact separable bounds for covariant entanglement witnesses in every dimension and the absence of non-decomposable mirrors

**with a correction to Quantum Sci. Technol. 11, 015037 (2026)**

Author: Ruqing Chen — Independent Researcher, Montreal, Quebec, Canada ·
ORCID [0009-0009-4622-6615](https://orcid.org/0009-0009-4622-6615) · ruqing@hotmail.com

Archived at Zenodo: [doi:10.5281/zenodo.24442820](https://doi.org/10.5281/zenodo.24442820)

This repository contains the manuscript, the deterministic verification script and the figure
script for a study of mirrored entanglement witnesses. It was prompted by one example in

> J. Seong, A. Bera, B. C. Hiesmayr, D. Chruściński, and J. Bae,
> *Mirrored entanglement witnesses for multipartite and high-dimensional quantum systems*,
> Quantum Sci. Technol. **11**, 015037 (2026),
> [doi:10.1088/2058-9565/ae30a3](https://doi.org/10.1088/2058-9565/ae30a3),
> [arXiv:2510.06863](https://arxiv.org/abs/2510.06863).

Equation numbers quoted as "SBHCB (n)" refer to arXiv version v2 of that paper.

## Setting

A mirrored entanglement witness is obtained from a witness W by reflection at its separable
upper bound: M = u_W·1 − W, where u_W is the maximum of ⟨e,f|W|e,f⟩ over product vectors.
The paper treats the circulant, diagonally covariant operators on C^n ⊗ C^n

    W_α = Σ_{i,k} α_k |i,i+k⟩⟨i,i+k|  −  Σ_{i≠j} |ii⟩⟨jj|        (indices mod n)

with arbitrary real α = (α_0, …, α_{n−1}). For n = 4 these are the operators W[a,b,c,d] of
SBHCB (72), for n = 3 the generalized Choi operators W[a,b,c].

## Main results

| Result | Statement |
|---|---|
| **Theorem 2** – decomposition certificate | If λ ∈ [0,1], v ≥ α_0 + λ, v ≥ α_k and (v − α_k)(v − α_{n−k}) ≥ (1 − λ)² for all k ≠ 0, then v·1 − W_α = P + Q^Γ with explicit P, Q ≥ 0 (Γ = partial transpose). |
| **Theorem 3** – exact separable bound, every n ≥ 2 | u_W = L_n(α) = max{ α_0, …, α_{n−1}, B_1, …, B_⌊n/2⌋ } with B_k = ((α_0+1)² − α_k α_{n−k}) / (2(α_0+1) − α_k − α_{n−k}), included only if α_0 + 1 > max(α_k, α_{n−k}). The maximum is attained on basis vectors or on product vectors supported on two levels {i, i+k}. Moreover u_W·1 − W_α = P + Q^Γ. |
| **Corollary 4** – no non-decomposable mirror | For every n and every real μ: μ·1 − W_α is block positive ⇔ μ ≥ L_n(α) ⇔ μ·1 − W_α is decomposable. A mirrored operator of this family is a decomposable witness or a positive operator, never a non-decomposable witness, with no optimality assumption. |
| **Remark 6** – dimensional jump | For W[1,…,1,0]: u_W = 4/3 for n = 3 (the Choi witness) and u_W = 3/2 for every n ≥ 4. |
| **Section 3** – counterexample | SBHCB give the mirror of the 4⊗4 witness W[1,1,1,0] as (4/3)·1 − W[1,1,1,0]. On the antipodal product vector ½(\|0⟩+\|2⟩)⊗(\|0⟩−\|2⟩) one has ⟨W[1,1,1,0]⟩ = 3/2, so that operator has expectation value **−1/6** on a separable state and is not an entanglement witness. |
| **Corollaries 8, 9** – the two 4⊗4 classes of SBHCB | Class I: u_W = 3/2 for all θ, with (3/2)·1 − W = 2(1+sin θ)\|φ⁺⟩⟨φ⁺\| + Q_θ^Γ. Class II: u_W = 3/2 on [0, π/2] and u_W = λ_max(W) on [π/2, π]. With the correct constant the PPT states ρ_x of SBHCB (75) are not detected. |
| **Proposition 10** – embedding | A non-decomposable mirrored pair (W_3, M_3) in 3⊗3 with W_3 + M_3 = μ·1 yields, by isometric embedding plus (μ/2)(1 − Π), a non-decomposable mirrored pair in every n⊗n, n ≥ 4. The three-dimensional pair itself is due to Chruściński, Bera, Bae and Hiesmayr, Sci. Rep. 15, 28205 (2025). |
| **Section 6, Appendix A** – recomputation | The 3⊗3 statements, the class-II classification, the pair SBHCB (77)–(78) and the three-qubit and GHZ identities of the audited paper are confirmed; six minor inconsistencies are listed. |

## Files

| File | Content |
|---|---|
| `mirrored_entanglement_witnesses_audit.pdf` | manuscript (12 pages) |
| `mirrored_entanglement_witnesses_audit.tex` | LaTeX source |
| `verify_mirrored_witnesses.py` | deterministic verification script (83 checks) |
| `verify_output.txt` | complete output of a run of the verification script |
| `make_figures.py` | generates the three figures |
| `fig_negative_region.pdf` | expectation of (4/3)·1 − W[1,1,1,0] on a family of product vectors |
| `fig_product_families.pdf` | the extremal product vectors for n = 4 |
| `fig_phase_diagram.pdf` | nature of μ·1 − W for the two 4⊗4 classes |
| `.zenodo.json` | metadata for the Zenodo archive |

## Reproducing the results

Requirements: Python ≥ 3.9 with NumPy and SymPy; Matplotlib for the figures.

```bash
pip install numpy sympy matplotlib
python verify_mirrored_witnesses.py > verify_output.txt
python make_figures.py
```

`verify_mirrored_witnesses.py` prints one `[PASS]`/`[FAIL]` line per check and a summary, and
exits with status 0 if and only if all 83 checks pass. It runs in about ninety seconds and was
last run with Python 3.12, NumPy 1.26 and SymPy 1.13. The checks cover:

- **A** – the 4⊗4 example: the value −1/6 and the decomposition of (3/2)·1 − W[1,1,1,0] in
  exact rational arithmetic; the trace formulas for ρ_x (symbolic).
- **B1–B3** – the certificate and the closed form for n = 4 and n = 3 on random parameters;
  the two classes of SBHCB (73); Proposition 6 of SBHCB.
- **B4** – arbitrary dimension: certificate, attainment on two-level product vectors and the
  λ_max formula for n = 2, …, 8 (150 random parameter vectors each); independent numerical
  maximization over product vectors for n = 2, 5, 6, 7; the jump 4/3 → 3/2.
- **C1–C3** – recomputation of the other examples of the audited paper.
- **D** – the embedded pair in 4⊗4.

To rebuild the PDF: `pdflatex mirrored_entanglement_witnesses_audit.tex` (three runs).

## Scope and limitations

- The audit used arXiv:2510.06863v2 (dated four days after the journal publication). It was
  not compared line by line with the version of record.
- Numerical optimization over product vectors is used only as an independent test of the
  closed forms; all theorems are proved analytically in the manuscript.
- For two of the listed inconsistencies (the three-qubit states ρ(x,y,z) and ρ(b,c)) the
  manuscript reports what the printed definitions give and does not decide which printed
  object is the intended one.

## Citation

> R. Chen, *Exact separable bounds for covariant entanglement witnesses in every dimension and
> the absence of non-decomposable mirrors, with a correction to Quantum Sci. Technol. 11,
> 015037 (2026)*, manuscript dated November 17, 2026.
> doi:10.5281/zenodo.24442820 ·
> https://github.com/Ruqing1963/seong-chruscinski-bae-2026-qst-audit

## AI assistance

The author conceived and directed this work. The AI tools Claude (Anthropic) and Gemini
(Google) assisted with English writing, LaTeX typesetting, cross-checking of derivations and
the writing of the scripts; see the declaration in the manuscript.
