# KLIFS position-level resistance propensity

Which aligned kinase pocket positions produce drug resistance when
mutated. Built from 4,992 measured (drug, mutation) pairs over
34 genes; a resistance event is ΔΔG ≤ -1.0
(the drug loses ≥10× affinity). Pooled base rate **0.178**.

**This is a lookup table, not a model, and that is deliberate.**
A model was considered and abandoned after counting: there are only
**85 (gene, position) pairs**, and at n ≥ 5 just **1 of 20 genes** has
both hotspot and non-hotspot positions — so leave-one-gene-out AUROC
is undefined almost everywhere and the task cannot be evaluated
honestly. The cause is biological: each gene has measurements at only
1–13 positions, because the field measures the resistance sites it
already knows about.

## Does it transfer to an unseen gene?

Leave-one-gene-out: the table is built from the other genes and scored
on the held-out one. Mean AUROC **0.541** over
20 genes (chance 0.500), beating chance by +0.041 at
t = +0.92, winning 12/20 genes; removing the
most influential gene (AURKA) leaves t = +0.59.

## Where the signal actually comes from — a correction

The per-position rate scored **0.583** in the Step 7 classification
run (17 folds, all rows), beating a per-drug rate (0.556) and nearly
matching a trained model (0.596). I described that as *position
identity beating drug identity*. Decomposing it shows that reading was
too generous:

| Predictor | genes | AUROC | t vs chance | jackknife |
|---|---|---|---|---|
| per-position, with *not-in-pocket* as its own category | 17 | **0.583** | +3.02 | +2.67 |
| **just the one bit: is the mutation inside the pocket?** | 17 | **0.545** | **+2.51** | **+2.18** |
| per-position, restricted to in-pocket rows | 12 | 0.555 | +1.70 | +1.28 |

So **most of that 0.583 is a single binary fact** — whether the
mutation falls inside the ATP pocket at all — which is real,
jackknife-robust, and needs no model. The fine-grained 85-position
resolution that this table provides is **not established**: with the
in/out bit held constant it reaches t = 1.70 on 12 genes.

That is the honest status of this table. It is a usable prior for
ranking positions, not a validated predictor.

## Columns

| Column | Meaning |
|---|---|
| `klifs_pos` | KLIFS pocket position, 1–85, aligned across the kinome |
| `region` | structural region of that position |
| `n_measurements` / `n_genes` / `n_mutations` | how much evidence |
| `resistance_events` | measurements with ΔΔG ≤ −1 |
| `rate_raw` | events / measurements |
| `rate_smoothed` | shrunk toward the pooled rate; use this for ranking |
| `ci95_low` / `ci95_high` | Wilson 95% interval on `rate_raw` |
| `wt_residues` | wild-type residues seen at this position |
| `genes` | contributing genes (up to 8) |

**Read the interval, not the rate.** Many positions rest on a handful
of measurements, where `rate_raw` is almost unconstrained — a position
at 1.00 on 2 measurements has a 95% interval running down past 0.3.

## Top positions by smoothed rate

|   klifs_pos | region   |   n_measurements |   n_genes |   resistance_events |   rate_raw |   ci95_low |   ci95_high | wt_residues   |
|------------:|:---------|-----------------:|----------:|--------------------:|-----------:|-----------:|------------:|:--------------|
|          81 | xDFG     |               22 |         1 |                  21 |     0.9545 |     0.782  |      0.9919 | D             |
|          19 | III      |               30 |         1 |                  14 |     0.4667 |     0.3023 |      0.6386 | L             |
|          51 | linker   |              338 |         5 |                 116 |     0.3432 |     0.2946 |      0.3953 | G             |
|          47 | hinge    |              186 |         5 |                  56 |     0.3011 |     0.2397 |      0.3705 | CFLY          |
|          11 | II       |                1 |         1 |                   1 |     1      |     0.2065 |      1      | C             |
|          36 | b.l      |               16 |         4 |                   4 |     0.25   |     0.1018 |      0.495  | IV            |
|          45 | GK       |             1705 |        18 |                 375 |     0.2199 |     0.2009 |      0.2402 | FLQTV         |
|          52 | linker   |              304 |         5 |                  67 |     0.2204 |     0.1774 |      0.2703 | CDET          |
|          49 | linker   |                3 |         2 |                   1 |     0.3333 |     0.0615 |      0.7923 | HS            |
|          10 | II       |               76 |         1 |                  16 |     0.2105 |     0.134  |      0.315  | E             |
|          37 | b.l      |               24 |         2 |                   5 |     0.2083 |     0.0924 |      0.4047 | N             |
|          28 | αC       |                4 |         1 |                   1 |     0.25   |     0.0456 |      0.6994 | I             |
|           5 | g.l      |               22 |         1 |                   4 |     0.1818 |     0.0731 |      0.3852 | G             |
|          85 | a.l      |                6 |         2 |                   1 |     0.1667 |     0.0301 |      0.5635 | CS            |
|          23 | αC       |                6 |         1 |                   1 |     0.1667 |     0.0301 |      0.5635 | D             |
