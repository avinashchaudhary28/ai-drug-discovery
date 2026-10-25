# v3 experiment: kya zyada data se binding model behtar hota hai?

**Design:** model, features, hyperparameters dono taraf bilkul same. Sirf training data badla.

- A = Davis train pairs
- B = Davis train pairs + BindingDB (Kd+Ki)
- Test: Davis **cold-drug** split, 5 seeds
- BindingDB se match hue proteins: 351
- Leakage hataya: test dawaiyon ke canonical SMILES, aur Davis ke saath duplicate jode

## Average (± std)

| model   |   ('CI', 'mean') |   ('CI', 'std') |   ('MSE', 'mean') |   ('MSE', 'std') |   ('Pearson', 'mean') |   ('Pearson', 'std') |
|:--------|-----------------:|----------------:|------------------:|-----------------:|----------------------:|---------------------:|
| A       |           0.7437 |          0.0442 |            0.6674 |           0.1734 |                0.5281 |               0.062  |
| B       |           0.7315 |          0.0478 |            0.6629 |           0.1428 |                0.5419 |               0.0936 |

## Har seed ka CI

|   seed |      A |      B |   farak (B-A) |
|-------:|-------:|-------:|--------------:|
|      0 | 0.7322 | 0.7517 |        0.0195 |
|      1 | 0.7994 | 0.7096 |       -0.0898 |
|      2 | 0.7139 | 0.6731 |       -0.0408 |
|      3 | 0.7789 | 0.8005 |        0.0216 |
|      4 | 0.6942 | 0.7225 |        0.0283 |

**B jeeta 3/5 seeds me · CI farak -0.0122 (±0.0515)**

**DATA se koi pakka farak nahi - shor ke andar hai. Bottleneck data nahi, protein representation hai (pocket-aware raasta).**
