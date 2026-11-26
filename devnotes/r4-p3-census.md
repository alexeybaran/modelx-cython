# R4 P3 census: layers each regional model needs on line B (experiment E1)

Method and summary: `r4-p3-reconcile.md` section 7. Every model needs B4 (table base) and B5 (row fields, without
the inliner), so those are left out below. L5/7 = B6 extremes and membership; L3/8 = B6 sub-tables and walks;
math = native math calls (p1-math, assumed in line A). In brackets: reads left on pandas in per-step cells on
line B itself (lean/original text); no bracket means none on either text.

| lean text needs | original text needs | n | models |
|---|---|---|---|
| B1 L5/7 | L5/7 | 9 | Basis_DE_S, Endowment_JP_S (1/0), KLV_DE_S, Obseques_FR_S, SPIA_US_S, Term_JP_S, UC_FR_S (1/1), ULB_UK_S, WholeLife_JP_S |
| B1 L5/7 L3/8 | L5/7 L3/8 | 6 | CI_UK_S (3/1), EC_FR_S (1/1), PER_FR_S (4/4), Sofort_DE_S (2/0), TD_FR_S (1/1), WOL_UK_S (1/1) |
| B1 L5/7 L3/8 math | L5/7 L3/8 math | 5 | Cancer_KR_S (14/10), FIA_US_S (3/2), LTC_JP_S (2/0), MYGA_US_S (2/1), RILA_US_S (2/1) |
| B1 | - | 4 | FRV_DE_S, FXWholeLife_JP_S (2/1), PA_UK_S, RLV_DE_S (1/0) |
| B1 B2 | B2 | 3 | Index_DE_S, Rente_FR_S, Riester_DE_S |
| B1 B2 L5/7 | B2 L5/7 | 3 | RV_DE_S (1/1), Term_KR_S, Term_UK_S |
| B1 B3 L5/7 L3/8 | B3 L5/7 L3/8 | 3 | IP_UK_S (1/1), Medical_JP_S (1/1), WP_UK_S (1/1) |
| B1 L5/7 math | L5/7 math | 3 | Cancer_JP_S (11/11), IUL_US_S, UL_US_S |
| B0 B1 | B0 | 2 | DIA_US_S, WholeLife_US_S |
| B1 L3/8 | L3/8 | 2 | Immediate_KR_S, WholeLife_KR_S |
| B1 L3/8 math | L3/8 math | 2 | Child_KR_S (12/9), LTC_KR_S (6/4) |
| B0 B1 B2 B3 L5/7 L3/8 math | B0 B2 L5/7 L3/8 math | 1 | Annuity_JP_S (9/6) |
| B0 B1 B2 L3/8 math | B0 B2 L3/8 math | 1 | Pension_KR_S (8/2) |
| B0 B1 B2 L5/7 | B0 B2 L5/7 | 1 | IncomeTerm_JP_S (3/3) |
| B0 B1 B3 L5/7 L3/8 | B0 L5/7 L3/8 | 1 | ADE_FR_S (2/2) |
| B0 B1 L5/7 math | B0 L5/7 math | 1 | ULSG_US_S |
| B1 B2 L5/7 L3/8 | B2 L5/7 L3/8 | 1 | Euro_FR_S (2/2) |
| B1 B2 L5/7 L3/8 math | B2 L5/7 L3/8 math | 1 | VA_US_S (3/3) |
| B1 B2 math | B2 math | 1 | CI_KR_S (1/0) |
| B1 B3 L3/8 | B3 L3/8 | 1 | VA_KR_S (11/10) |
| B1 B3 L5/7 | B3 L5/7 | 1 | BU_DE_S |
| B1 B3 L5/7 L3/8 | B2 L5/7 L3/8 | 1 | Medical_KR_S (2/1) |
| B1 B3 L5/7 L3/8 math | B2 L5/7 L3/8 math | 1 | VUL_US_S (1/1) |
| B1 B3 L5/7 L3/8 math | L5/7 L3/8 math | 1 | Dep_FR_S (2/0) |
| B1 math | math | 1 | Pflege_DE_S |
| L3/8 | L3/8 | 1 | Term_US_S |

Lowered read sites per model, lean text (identical on the original text except one L8 site fewer in one model):

| model | L1 | L2 | L3 | L5 | L7 | L8 | reductions | vector calls | per-step pandas (lean/original) |
|---|---|---|---|---|---|---|---|---|---|
| ADE_FR_S | 5 | 20 | 8 | 2 | 0 | 3 | 0/0 | 57/0 | 2/2 |
| Annuity_JP_S | 6 | 16 | 1 | 1 | 0 | 1 | 1/1 | 10/0 | 9/6 |
| BU_DE_S | 7 | 20 | 0 | 7 | 0 | 0 | 0/0 | 231/60 | 0/0 |
| Basis_DE_S | 16 | 80 | 0 | 5 | 0 | 0 | 0/0 | 0/0 | 0/0 |
| CI_KR_S | 2 | 17 | 0 | 0 | 0 | 0 | 2/2 | 0/0 | 1/0 |
| CI_UK_S | 3 | 17 | 0 | 1 | 0 | 1 | 0/0 | 0/0 | 3/1 |
| Cancer_JP_S | 7 | 21 | 0 | 6 | 0 | 0 | 0/0 | 0/0 | 11/11 |
| Cancer_KR_S | 18 | 17 | 2 | 1 | 0 | 2 | 0/0 | 0/0 | 14/10 |
| Child_KR_S | 2 | 30 | 0 | 0 | 0 | 5 | 0/0 | 0/0 | 12/9 |
| DIA_US_S | 3 | 26 | 0 | 0 | 0 | 0 | 0/0 | 0/0 | 0/0 |
| Dep_FR_S | 9 | 21 | 0 | 4 | 0 | 3 | 0/0 | 92/0 | 2/0 |
| EC_FR_S | 1 | 32 | 2 | 2 | 0 | 3 | 0/0 | 0/0 | 1/1 |
| Endowment_JP_S | 3 | 21 | 0 | 2 | 0 | 0 | 0/0 | 0/0 | 1/0 |
| Euro_FR_S | 2 | 19 | 2 | 3 | 0 | 0 | 1/1 | 0/0 | 2/2 |
| FIA_US_S | 5 | 32 | 1 | 3 | 1 | 3 | 0/0 | 0/0 | 3/2 |
| FRV_DE_S | 6 | 34 | 0 | 0 | 0 | 0 | 0/0 | 0/0 | 0/0 |
| FXWholeLife_JP_S | 2 | 26 | 0 | 0 | 0 | 0 | 0/0 | 0/0 | 2/1 |
| IP_UK_S | 4 | 14 | 2 | 3 | 0 | 1 | 0/0 | 67/67 | 1/1 |
| IUL_US_S | 6 | 28 | 0 | 6 | 0 | 0 | 0/0 | 0/0 | 0/0 |
| Immediate_KR_S | 1 | 29 | 0 | 0 | 0 | 1 | 0/0 | 0/0 | 0/0 |
| IncomeTerm_JP_S | 3 | 12 | 0 | 1 | 1 | 0 | 1/1 | 0/0 | 3/3 |
| Index_DE_S | 8 | 24 | 0 | 0 | 0 | 0 | 2/2 | 0/0 | 0/0 |
| KLV_DE_S | 17 | 36 | 0 | 9 | 1 | 0 | 0/0 | 0/0 | 0/0 |
| LTC_JP_S | 4 | 20 | 0 | 1 | 0 | 1 | 0/0 | 0/0 | 2/0 |
| LTC_KR_S | 7 | 21 | 10 | 0 | 0 | 4 | 0/0 | 0/0 | 6/4 |
| MYGA_US_S | 5 | 23 | 0 | 2 | 2 | 1 | 0/0 | 0/0 | 2/1 |
| Medical_JP_S | 2 | 21 | 0 | 2 | 0 | 4 | 0/0 | 38/38 | 1/1 |
| Medical_KR_S | 6 | 16 | 0 | 3 | 0 | 3 | 0/3 | 21/0 | 2/1 |
| Obseques_FR_S | 4 | 22 | 0 | 4 | 0 | 0 | 0/0 | 0/0 | 0/0 |
| PA_UK_S | 1 | 22 | 0 | 0 | 0 | 0 | 0/0 | 0/0 | 0/0 |
| PER_FR_S | 1 | 19 | 4 | 4 | 0 | 0 | 0/0 | 0/0 | 4/4 |
| Pension_KR_S | 9 | 19 | 2 | 0 | 0 | 3 | 1/1 | 0/0 | 8/2 |
| Pflege_DE_S | 51 | 17 | 0 | 0 | 0 | 0 | 0/0 | 0/0 | 0/0 |
| RILA_US_S | 6 | 24 | 0 | 1 | 2 | 3 | 0/0 | 0/0 | 2/1 |
| RLV_DE_S | 6 | 27 | 0 | 0 | 0 | 0 | 0/0 | 0/0 | 1/0 |
| RV_DE_S | 25 | 92 | 0 | 3 | 0 | 0 | 2/2 | 0/0 | 1/1 |
| Rente_FR_S | 1 | 20 | 0 | 0 | 0 | 0 | 1/1 | 0/0 | 0/0 |
| Riester_DE_S | 8 | 42 | 0 | 0 | 0 | 0 | 1/1 | 0/0 | 0/0 |
| SPIA_US_S | 2 | 22 | 0 | 2 | 0 | 0 | 0/0 | 0/0 | 0/0 |
| Sofort_DE_S | 6 | 20 | 0 | 4 | 0 | 1 | 0/0 | 0/0 | 2/0 |
| TD_FR_S | 6 | 17 | 1 | 2 | 1 | 0 | 0/0 | 0/0 | 1/1 |
| Term_JP_S | 4 | 25 | 0 | 2 | 1 | 0 | 0/0 | 0/0 | 0/0 |
| Term_KR_S | 8 | 27 | 0 | 0 | 1 | 0 | 2/2 | 0/0 | 0/0 |
| Term_UK_S | 2 | 20 | 0 | 1 | 0 | 0 | 1/1 | 0/0 | 0/0 |
| Term_US_S | 3 | 11 | 0 | 0 | 0 | 1 | 0/0 | 0/0 | 0/0 |
| UC_FR_S | 3 | 30 | 0 | 1 | 1 | 0 | 0/0 | 0/0 | 1/1 |
| ULB_UK_S | 2 | 15 | 0 | 1 | 0 | 0 | 0/0 | 0/0 | 0/0 |
| ULSG_US_S | 7 | 24 | 0 | 6 | 1 | 0 | 0/0 | 0/0 | 0/0 |
| UL_US_S | 7 | 22 | 0 | 7 | 0 | 0 | 0/0 | 0/0 | 0/0 |
| VA_KR_S | 7 | 14 | 0 | 0 | 0 | 1 | 0/0 | 44/44 | 11/10 |
| VA_US_S | 5 | 30 | 3 | 2 | 2 | 4 | 4/4 | 0/0 | 3/3 |
| VUL_US_S | 7 | 25 | 1 | 7 | 0 | 1 | 0/5 | 37/0 | 1/1 |
| WOL_UK_S | 1 | 12 | 1 | 1 | 0 | 0 | 0/0 | 0/0 | 1/1 |
| WP_UK_S | 1 | 19 | 1 | 1 | 0 | 0 | 0/0 | 5/5 | 1/1 |
| WholeLife_JP_S | 2 | 17 | 0 | 1 | 0 | 0 | 0/0 | 0/0 | 0/0 |
| WholeLife_KR_S | 1 | 22 | 0 | 0 | 0 | 1 | 0/0 | 0/0 | 0/0 |
| WholeLife_US_S | 4 | 17 | 0 | 0 | 0 | 0 | 0/0 | 0/0 | 0/0 |
