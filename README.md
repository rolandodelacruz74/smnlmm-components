# Code for "Geometric ergodicity of Gibbs samplers for linear mixed models with several heavy-tailed random-effect components and unknown tail parameters"

Python 3 with numpy, scipy and pandas. Run the scripts from `code/`; results are written to `results/`.
All random seeds are fixed in the scripts.

## Data
The beta-HCG data of Section 5 are available from the first author upon request; place them as `data/bHCG.txt`.
The simulations of Section 6 and the verification tests of the Appendix do not need the data; `mvt_blockcond.py` does.

## Modules
- `smnlmm2.py`: the three-block Gibbs sampler with several components, component-specific mixing laws, tail
  parameters on finite grids drawn jointly with the weights (`GridMixing`), and an exact draw of theta.
- `smnlmm.py`: mixing distributions, WAIC/LPML, batch means, multivariate ESS and minESS.
- `data.py`: reads the beta-HCG data.

## Reproducing the paper
| Output | Command |
|---|---|
| Section 5, models M0 to M3 (Tables of fit and of the degrees of freedom) | `python3 p2_app.py M0 30000`, `python3 p2_app.py M3 30000`, and `python3 p2_app.py M 30000 scaled` for M in {M1, M2} |
| Section 5, sensitivity to the priors on the scales | `python3 p2_app.py M 30000` (Gamma(1, 0.1) for all precisions) and `python3 p2_app.py M 30000 0.001`, for M in {M0, ..., M3}; `python3 p2_prior_sens.py` |
| Section 6, simulated example with shared weights | `python3 p2_shared.py 105000` |
| Section 6, Table on the information per subject | `python3 p2_info.py N1 D0` for N1 in {0, 15, 30} and D0 in {1, 4} |
| Appendix, joint distribution tests | `python3 test_geweke2.py CASE 60000` for CASE in {multi, err, grid, gridlight, sl08} |
| Appendix, normalizing functions of the tilted laws (against adaptive quadrature) and joint draw of the tail parameter and the weights (against the exact conditional law on the grid) | `python3 test_grid.py` |
| Auxiliary check, not reported in the paper: block condition of Theorem 2 for multivariate Student-t random intercepts and slopes with a known shape matrix on the beta-HCG design (needs the data) | `python3 mvt_blockcond.py` |

## License

MIT License. See `LICENSE`.
