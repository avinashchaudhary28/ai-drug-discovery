# v3 experiment: kya zyada data se binding model behtar hota hai?

- Test set: **Davis cold-drug** (6,188 jode, 14 nayi dawaiyan jo training me nahi thi)
- BindingDB se jude jode: **33,982** (351 proteins pe)
- Model, features, hyperparameters: **dono me same** (sirf data badla)

| model              |   n_train |     CI |    MSE |   Pearson |
|:-------------------|----------:|-------:|-------:|----------:|
| A: Davis sirf      |     23868 | 0.7322 | 0.6911 |    0.5301 |
| B: Davis+BindingDB |     57850 | 0.7517 | 0.5865 |    0.6118 |

**CI farak: +0.0195**

**DATA se faida hua - BindingDB jodne se model behtar hua.**
