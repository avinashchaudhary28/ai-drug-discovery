# v3 experiment: kya zyada data se binding model behtar hota hai?

**Design:** model, features, hyperparameters sab variants me bilkul same. Sirf training data badla.

- Baseline **A** = sirf Davis train pairs
- **B** = A + BindingDB (Kd+Ki) · **B2** = A + sirf Kd · **B3/B4** = wahi, par model ko assay type bata ke
- Test: Davis **cold-drug** split, 5 seeds (mean ± std)
- BindingDB se match hue proteins: 351
- Leakage hataya: test dawaiyon ke canonical SMILES, aur Davis ke saath duplicate jode

## Har variant, average

| model                 |   CI_mean |   CI_std |   MSE_mean |   MSE_std |   Pearson_mean |   Pearson_std |   n_train |
|:----------------------|----------:|---------:|-----------:|----------:|---------------:|--------------:|----------:|
| A  Davis sirf         |    0.7437 |   0.0442 |     0.6674 |    0.1734 |         0.5281 |        0.062  |     23868 |
| A  Davis sirf + assay |    0.7353 |   0.0327 |     0.6913 |    0.1404 |         0.5107 |        0.0407 |     23868 |
| B  all assays         |    0.7423 |   0.0424 |     0.6387 |    0.1204 |         0.5674 |        0.0839 |     58060 |
| B2 sirf Kd            |    0.759  |   0.053  |     0.6182 |    0.1585 |         0.6016 |        0.0802 |     40809 |
| B3 all + assay        |    0.7461 |   0.0347 |     0.6151 |    0.1181 |         0.5797 |        0.059  |     58060 |
| B4 Kd + assay         |    0.7538 |   0.0579 |     0.6304 |    0.157  |         0.5976 |        0.0957 |     40809 |

## Baseline se tulna

| variant        | baseline              |   seeds |   jeete |   CI_farak |    std |
|:---------------|:----------------------|--------:|--------:|-----------:|-------:|
| B  all assays  | A  Davis sirf         |       5 |       2 |    -0.0014 | 0.0564 |
| B2 sirf Kd     | A  Davis sirf         |       5 |       3 |     0.0153 | 0.0479 |
| B3 all + assay | A  Davis sirf + assay |       5 |       3 |     0.0107 | 0.0391 |
| B4 Kd + assay  | A  Davis sirf + assay |       5 |       3 |     0.0184 | 0.0584 |

## Har seed ka CI

|   seed |   A  Davis sirf |   A  Davis sirf + assay |   B  all assays |   B2 sirf Kd |   B3 all + assay |   B4 Kd + assay |
|-------:|----------------:|------------------------:|----------------:|-------------:|-----------------:|----------------:|
|      0 |          0.7322 |                  0.7342 |          0.7893 |       0.7752 |           0.7806 |          0.7753 |
|      1 |          0.7994 |                  0.7785 |          0.73   |       0.7714 |           0.7423 |          0.727  |
|      2 |          0.7139 |                  0.7011 |          0.6771 |       0.6714 |           0.689  |          0.6663 |
|      3 |          0.7789 |                  0.756  |          0.7653 |       0.8151 |           0.7563 |          0.8117 |
|      4 |          0.6942 |                  0.7069 |          0.7498 |       0.762  |           0.7623 |          0.7885 |

**KISI BHI tarah ke data se pakka faida nahi. Sabse achha variant 'B4 Kd + assay' bhi sirf 3/5 seeds jeeta (CI +0.0184 ±0.0584) - farak shor ke andar hai. Matlab mila-jula assay data wajah nahi thi; bottleneck PROTEIN REPRESENTATION hai (pocket-aware raasta).**
