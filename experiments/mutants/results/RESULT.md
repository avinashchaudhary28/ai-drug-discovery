# Asli mutant sequences dene se model behtar hota hai?

Davis mutants ko unke wild-type ki hubahu sequence deta hai. Naam se mutation parse karke asli sequence banayi, phir dono taraf ek hi model/features/splits pe naapa.

- unique sequences: **379 -> 423** (50 badli)
- mutation parse status: {'wt': 367, 'ok': 50, 'skip': 25}
- ESM: facebook/esm2_t33_650M_UR50D, dono taraf CPU float32 me (purani GPU-half embeddings mix nahi ki)
- 5 seeds, splits: random aur cold-target

## Contradictory labels (ek hi input, alag jawab)

| tag   |   groups |   rows |   pct_rows |   mean_std |   mean_range |
|:------|---------:|-------:|-----------:|-----------:|-------------:|
| WT    |      546 |   2894 |       9.63 |     0.5566 |       1.1203 |
| MUT   |      555 |   1192 |       3.97 |     0.5051 |       0.7356 |

## Nateeja (t = mean/SE, |t|>=2 = shor se bada)

| split       | metric                    |     WT |    MUT |   delta | jeete   |     t |
|:------------|:--------------------------|-------:|-------:|--------:|:--------|------:|
| random      | CI                        | 0.8862 | 0.8878 |  0.0016 | 4/5     |  2.31 |
| random      | CI (sirf mutant proteins) | 0.9229 | 0.9402 |  0.0173 | 5/5     | 23.16 |
| cold-target | CI                        | 0.8483 | 0.8497 |  0.0014 | 3/5     |  0.93 |
| cold-target | CI (sirf mutant proteins) | 0.9187 | 0.9258 |  0.0071 | 5/5     |  2.86 |

**SUDHAAR ASLI HAI - mutant proteins pe CI +0.0173 (t=23.16), 5/5 seeds. Asli mutant sequences dena faydemand hai.**
