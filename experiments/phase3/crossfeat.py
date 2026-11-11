"""Drug x mutation INTERACTION features - jo chaar koshishon me nahi tha.

Phase 3 ka diagnosis saaf hai:

    "All four feature sets describe the drug and the mutation separately
     plus pocket context; none describes the interaction geometry
     between them."

Par seedha interaction dena kaam nahi karega: 1024-bit fingerprint x
85 pocket positions = lakhon combinations, aur hamare paas 8,110 rows
hain. Tree usme sahi cross dhoondh hi nahi sakta.

To interaction ko ek AISI granularity pe laana hai jo 8,110 rows se
seekhi ja sake, aur jiske DONO axes gene se azaad hon (warna gene
badalne pe feature bemaani ho jayega):

    dawa ka axis     : scaffold cluster (Morgan FP pe KMeans)
    mutation ka axis : KLIFS pocket position - pehle se saare kinases
                       me aligned, yahi KLIFS ka poora matlab hai
                       (aur AA badlav ka class)

Phir (cluster, position) jodi ka training me dekha gaya resistance rate
ek feature banta hai. Yahi "kya is kism ki dawa is jagah ki mutation se
tootti hai" ka seedha jawab hai.

DO JAGAH LEAKAGE AA SAKTA HAI, DONO CHUP-CHAAP:

1. Cluster TEST data pe fit kar diya -> test ka dhaancha training me
   ghus gaya. (ESM ke PCA me yahi sabak mila tha: projection bhi
   training folds pe hi fit hota hai.)
2. Target encoding usi row ke label se bana jo encode ho rahi hai ->
   model ko apna jawab feature me mil jata hai. Training ka score
   shandaar, LOGO ka bekaar, aur koi error nahi.

Isliye: cluster SIRF train pe fit, aur encoding train ke ANDAR
out-of-fold - har row ki encoding un rows se banti hai jo us inner
fold me nahi hain.
"""
import numpy as np

N_CLUSTERS = 40
INNER_FOLDS = 5
SMOOTH = 20.0        # kitne observations pe local rate ko poora maana jaye


def fit_clusters(fp_train, n_clusters=N_CLUSTERS, seed=0):
    """Morgan FP pe KMeans - SIRF training rows pe.

    -> fitted object; `assign` se baad me test ko bhi label milta hai.
    """
    from sklearn.cluster import KMeans
    n = min(n_clusters, max(2, len(np.unique(fp_train, axis=0))))
    km = KMeans(n_clusters=n, n_init=4, random_state=seed)
    km.fit(fp_train)
    return km


def assign(km, fp):
    return km.predict(fp).astype(int)


def _rate_table(keys, y, smooth=SMOOTH, prior=None):
    """{key: smoothed rate}. prior ke taraf sikudta hai jab observations
    kam hon - warna ek hi row wala key 0 ya 1 bolkar shor bhar dega."""
    keys = np.asarray(keys)
    y = np.asarray(y, dtype=float)
    p0 = float(y.mean()) if prior is None else float(prior)
    out = {}
    for k in np.unique(keys):
        m = keys == k
        n = int(m.sum())
        out[k] = (y[m].sum() + smooth * p0) / (n + smooth)
    return out, p0


def oof_encode(keys, y, n_folds=INNER_FOLDS, smooth=SMOOTH, seed=0):
    """Train ke ANDAR out-of-fold encoding.

    Har row ki value un rows se banti hai jo uske inner fold me NAHI
    hain. Seedhe poore train pe encode karne se row ko apna hi label
    mil jata, aur wo leakage hai jo koi error nahi deta.

    -> (encoded array, full-train table, prior) - table test ke liye.
    """
    keys = np.asarray(keys)
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, n_folds, len(y))
    enc = np.empty(len(y), dtype=np.float32)
    for f in range(n_folds):
        te = fold == f
        tr = ~te
        if not tr.any():
            enc[te] = float(y.mean())
            continue
        tbl, p0 = _rate_table(keys[tr], y[tr], smooth)
        enc[te] = [tbl.get(k, p0) for k in keys[te]]
    full, prior = _rate_table(keys, y, smooth)
    return enc, full, prior


def apply_encode(keys, table, prior):
    """Test rows pe train ki table lagao. Andekha key -> prior."""
    return np.asarray([table.get(k, prior) for k in np.asarray(keys)],
                      dtype=np.float32)


def count_table(keys):
    keys = np.asarray(keys)
    u, c = np.unique(keys, return_counts=True)
    return dict(zip(u.tolist(), c.tolist()))


NAMES = ["x_cluster_pos_rate", "x_cluster_pos_n",
         "x_cluster_sub_rate", "x_cluster_sub_n",
         "x_pos_rate", "x_cluster_rate"]


def build(cl_tr, pos_tr, sub_tr, y_tr, cl_te, pos_te, sub_te, seed=0):
    """Training aur test ke cross features ek saath.

    cl_* : cluster id, pos_* : KLIFS position (ya -1), sub_* : AA class
    y_tr : training ka binary label

    -> (X_train, X_test) dono (n, 6)
    """
    def key2(a, b):
        return np.asarray([f"{x}|{z}" for x, z in zip(a, b)])

    k_cp_tr, k_cp_te = key2(cl_tr, pos_tr), key2(cl_te, pos_te)
    k_cs_tr, k_cs_te = key2(cl_tr, sub_tr), key2(cl_te, sub_te)
    p_tr = np.asarray([str(x) for x in pos_tr])
    p_te = np.asarray([str(x) for x in pos_te])
    c_tr = np.asarray([str(x) for x in cl_tr])
    c_te = np.asarray([str(x) for x in cl_te])

    # (train keys, test keys, count wala column bhi chahiye?)
    # Kram NAMES se milna chahiye - isliye yahan saaf likha hai,
    # `is` se pehchanne ke bharose nahi.
    specs = [(k_cp_tr, k_cp_te, True),      # cluster x position
             (k_cs_tr, k_cs_te, True),      # cluster x AA badlav
             (p_tr, p_te, False),           # sirf position
             (c_tr, c_te, False)]           # sirf cluster
    cols_tr, cols_te = [], []
    for ktr, kte, want_n in specs:
        enc, tbl, prior = oof_encode(ktr, y_tr, seed=seed)
        cols_tr.append(enc)
        cols_te.append(apply_encode(kte, tbl, prior))
        if want_n:
            cnt = count_table(ktr)
            cols_tr.append(np.asarray([cnt.get(k, 0) for k in ktr],
                                      dtype=np.float32))
            cols_te.append(np.asarray([cnt.get(k, 0) for k in kte],
                                      dtype=np.float32))
    Xtr = np.column_stack(cols_tr).astype(np.float32)
    Xte = np.column_stack(cols_te).astype(np.float32)
    assert Xtr.shape[1] == len(NAMES), (Xtr.shape[1], len(NAMES))
    return Xtr, Xte
