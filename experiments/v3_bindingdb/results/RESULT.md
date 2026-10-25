# v3 experiment: kya zyada data se binding model behtar hota hai?

**Design:** model, features, hyperparameters sab variants me bilkul same. Sirf training data badla.

- Baseline **A** = sirf Davis train pairs
- **B** = A + BindingDB (Kd+Ki) · **B2** = A + sirf Kd · **B3/B4** = wahi, par model ko assay type bata ke
- Test: Davis **cold-drug** split, 12 seeds (mean ± std)
- BindingDB se match hue proteins: 351
- Leakage hataya: test dawaiyon ke canonical SMILES, aur Davis ke saath duplicate jode

## Har variant, average

| model                 |   CI_mean |   CI_std |   MSE_mean |   MSE_std |   Pearson_mean |   Pearson_std |   n_train |
|:----------------------|----------:|---------:|-----------:|----------:|---------------:|--------------:|----------:|
| A  Davis sirf         |    0.7281 |   0.0327 |     0.6663 |    0.1193 |         0.4655 |        0.0732 |     23868 |
| A  Davis sirf + assay |    0.7223 |   0.0269 |     0.6829 |    0.1102 |         0.4511 |        0.0674 |     23868 |
| B  all assays         |    0.7104 |   0.0472 |     0.7114 |    0.1784 |         0.4683 |        0.1253 |     58060 |
| B2 sirf Kd            |    0.7459 |   0.0452 |     0.6282 |    0.1162 |         0.5455 |        0.0971 |     40809 |
| B3 all + assay        |    0.7214 |   0.0362 |     0.652  |    0.1122 |         0.4974 |        0.103  |     58060 |
| B4 Kd + assay         |    0.7365 |   0.0447 |     0.6533 |    0.1133 |         0.5277 |        0.1017 |     40809 |

## Baseline se tulna

| variant        | baseline              |   seeds |   jeete |   CI_farak |    std |
|:---------------|:----------------------|--------:|--------:|-----------:|-------:|
| B  all assays  | A  Davis sirf         |      12 |       2 |    -0.0177 | 0.0484 |
| B2 sirf Kd     | A  Davis sirf         |      12 |       8 |     0.0178 | 0.0441 |
| B3 all + assay | A  Davis sirf + assay |      12 |       6 |    -0.0009 | 0.0359 |
| B4 Kd + assay  | A  Davis sirf + assay |      12 |       8 |     0.0142 | 0.0419 |

## Har seed ka CI

|   seed |   A  Davis sirf |   A  Davis sirf + assay |   B  all assays |   B2 sirf Kd |   B3 all + assay |   B4 Kd + assay |
|-------:|----------------:|------------------------:|----------------:|-------------:|-----------------:|----------------:|
|      0 |          0.7322 |                  0.7342 |          0.7893 |       0.7752 |           0.7806 |          0.7753 |
|      1 |          0.7994 |                  0.7785 |          0.73   |       0.7714 |           0.7423 |          0.727  |
|      2 |          0.7139 |                  0.7011 |          0.6771 |       0.6714 |           0.689  |          0.6663 |
|      3 |          0.7789 |                  0.756  |          0.7653 |       0.8151 |           0.7563 |          0.8117 |
|      4 |          0.6942 |                  0.7069 |          0.7498 |       0.762  |           0.7623 |          0.7885 |
|      5 |          0.7335 |                  0.7217 |          0.6133 |       0.6672 |           0.6478 |          0.6762 |
|      6 |          0.7083 |                  0.6966 |          0.6678 |       0.7332 |           0.7058 |          0.7306 |
|      7 |          0.7235 |                  0.716  |          0.7177 |       0.7146 |           0.6947 |          0.7239 |
|      8 |          0.7212 |                  0.7296 |          0.7202 |       0.7871 |           0.7144 |          0.7727 |
|      9 |          0.7402 |                  0.7368 |          0.7143 |       0.7747 |           0.73   |          0.7512 |
|     10 |          0.695  |                  0.6812 |          0.6903 |       0.7249 |           0.7147 |          0.7127 |
|     11 |          0.697  |                  0.7087 |          0.6899 |       0.7545 |           0.7185 |          0.7018 |

**KISI BHI tarah ke data se pakka faida nahi. Sabse achha variant 'B2 sirf Kd' bhi sirf 8/12 seeds jeeta (CI +0.0178 ±0.0441) - farak shor ke andar hai. Matlab mila-jula assay data wajah nahi thi; bottleneck PROTEIN REPRESENTATION hai (pocket-aware raasta).**
