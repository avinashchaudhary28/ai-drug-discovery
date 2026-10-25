# Asli mutant sequences dene se model behtar hota hai?

Davis mutants ko unke wild-type ki hubahu sequence deta hai. Naam se mutation parse karke asli sequence banayi, phir dono taraf ek hi model/features/splits pe naapa.

- unique sequences: **379 -> 421** (47 badli)
- mutation parse status: {'wt': 367, 'skip': 28, 'ok-shifted': 18, 'ok': 29}
- ESM: facebook/esm2_t33_650M_UR50D, dono taraf CPU float32 me (purani GPU-half embeddings mix nahi ki)
- 5 seeds, splits: random aur cold-target

## Contradictory labels (ek hi input, alag jawab)

| tag   |   groups |   rows |   pct_rows |   mean_std |   mean_range |
|:------|---------:|-------:|-----------:|-----------:|-------------:|
| WT    |      546 |   2894 |       9.63 |     0.5566 |       1.1203 |
| MUT   |      515 |   1241 |       4.13 |     0.5154 |       0.7771 |

## Nateeja (t = mean/SE, |t|>=2 = shor se bada)

| split       | metric                    |     WT |    MUT |   delta | jeete   |     t |
|:------------|:--------------------------|-------:|-------:|--------:|:--------|------:|
| random      | CI                        | 0.8862 | 0.8871 |  0.0009 | 4/5     |  2.29 |
| random      | CI (sirf mutant proteins) | 0.9216 | 0.9385 |  0.0169 | 5/5     | 12.83 |
| cold-target | CI                        | 0.8483 | 0.8491 |  0.0008 | 4/5     |  2.19 |
| cold-target | CI (sirf mutant proteins) | 0.9178 | 0.9249 |  0.0071 | 5/5     |  3.08 |

**SUDHAAR ASLI HAI - mutant proteins pe CI +0.0169 (t=12.83), 5/5 seeds. Asli mutant sequences dena faydemand hai.**
