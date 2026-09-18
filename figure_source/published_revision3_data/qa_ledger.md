# R3 figure QA

| Check | Result |
| --- | --- |
| Source scope | Vision R3 A/B; language retained R2.1; no mixed runtime training cohort |
| Numeric integrity | All plotted P and BN-transfer intervals independently rederived from paired seed values and matched to source intervals |
| Derived BN change | BN minus raw within each seed before Student interval calculation |
| Calibration sample size | Three subsets kept separate; adaptation n is A=5, B=3 |
| Initial checkpoint | One state per initialization; no artificial training-seed interval |
| Vector export | PDF, SVG, PGF generated; all three compiled PGF PDFs contain zero image XObjects |
| Rendering | All three PNGs and standalone PGF-derived PDFs visually inspected; no clipped labels, legend collisions, or unreadable markers |
| Boundary correction | Figure 3 panel a right boundary expanded to retain the largest individual 1,024-update P2 change |
| Grayscale | P2/P4 use different marker shapes; BN uses open/filled; subsets use three shapes/fills |
| Preservation | Main tex, old figure generators, timing plots, and Figure 1 untouched |
| Remaining integration | Writing owner must embed PGF, update captions/cross-references, and check manuscript page layout |

Standalone PGF-derived PDF sizes: Figure 2 526.80 x 235.20 pt; Figure 3 526.80 x 217.20 pt; supplementary figure 526.80 x 411.56 pt. All three compile successfully with pdflatex and the manuscript-compatible PGF/Latin Modern packages. Preview and compile checks are in support/revision3_render_check and need not be included in the submission package.
