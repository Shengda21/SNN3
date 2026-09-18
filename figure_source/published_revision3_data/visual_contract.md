# R3 figure contract

Destination: ESWA manuscript, full-width quantitative vector figures.

- Figure 2: target-adaptation versus cross-window checkpoint reuse, raw CE. Language retains three paired R2.1 adaptation seeds at 512/4096; vision uses R3 initialization A, five paired seeds at 512/1024/2048. Dot observations plus paired Student 95% intervals; positive P favors target adaptation.
- Figure 3: effect of matched-window BN recalibration on P2/P4 across the same three vision A checkpoints, followed by the raw-to-BN movement in the P4/accuracy plane at 512 and 2048. Changes are paired before interval calculation. Metric disagreement is represented by the observed CE and accuracy signs.
- Supplement: separate A (five seeds) and B (three seeds) P4 budget trajectories, plus three calibration subsets across S0/S2/S4 and both inference windows at 2048. Positive transfer penalty means other-window calibration raises CE. S0 has one checkpoint and no training-seed interval. Calibration subsets do not increase training n.

Existing figure grammar is retained with neutral BN encodings and distinct P2/P4 markers. Full-width output is approximately 7.3 inches. PDF/PGF/SVG are vector, PNG is a preview. Source CSV identities and seed-level interval checks are stored in figure_audit.json. No main tex, prior generator, timing figure, or Figure 1 is modified. English captions are supplied separately for the writing owner. Final float/page QA belongs to the writing owner after integration.
