"""Bina network ke test: top-k hit rate.

Do khatre jo CHUP-CHAAP galat nateeja dete hain:

  1. DISHA ulti - "tikne wali" ki jagah "toot jane wali" dawa chun lena.
     Model ki sabse bekaar compounds "sabse achhi" dikhengi, number
     plausible hoga, error kabhi nahi aayega.
  2. BARABAR values ka kram - agar koi model sab ko ek hi number de
     (jaise `0 bolo`), aur ties pehle-aaya-pehle kram me rahein, to usko
     data ka apna kram muft mil jata hai aur wo jhoothi jeet dikhata hai.

Dono ki jaanch yahan seedhe taur pe hai.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from topk import MIN_GROUP, TOP, group_scores      # noqa: E402

FAIL = []


def check(name, got, want):
    if got == want:
        print(f"  ✓ {name}")
    else:
        print(f"  ✗ {name}\n      mila : {got!r}\n      chahiye: {want!r}")
        FAIL.append(name)


R = lambda s=0: np.random.default_rng(s)           # noqa: E731

print("DISHA: dhanatmak ddG = dawa TIKI, rinatmak = dawa FAIL")
# 6 dawaiyan. d0 ka ddG sabse DHANATMAK -> wahi sabse achhi tiki.
y = [+0.9, +0.4, 0.0, -0.5, -1.2, -2.0]
# Model bilkul sahi kram deta hai
s = group_scores(y, y, R())
check("sahi kram -> hit", s["hit"], 1.0)
check("sahi kram -> poora overlap", s["overlap"], 1.0)

# Model BILKUL ULTA kram deta hai - hit 0 hona chahiye
s = group_scores(y, [-v for v in y], R())
check("ulta kram -> hit NAHI", s["hit"], 0.0)
check("ulta kram -> overlap 0", s["overlap"], 0.0)

# Ye wahi jaanch hai jo disha ki galti pakdegi: agar koi `-` bhool jaye
# to ye test fail hoga, na ki chup-chaap plausible number aayega.

print("\nasli sabse-tikne-wali top-3 me hai ya nahi")
# Model d0 ko teesre sthan pe rakhta hai -> phir bhi hit
s = group_scores(y, [0.2, 0.9, 0.5, -0.5, -1.2, -2.0], R())
check("teesre sthan pe bhi hit", s["hit"], 1.0)
# Model d0 ko chauthe sthan pe -> hit nahi
s = group_scores(y, [0.1, 0.9, 0.5, 0.3, -1.2, -2.0], R())
check("chauthe sthan pe -> hit nahi", s["hit"], 0.0)

print("\nBARABAR values: model sab ko ek number de to muft jeet na mile")
flat = [0.0] * 8
yy = [+2.0, +1.0, +0.5, 0.0, -0.5, -1.0, -1.5, -2.0]
# 3000 seeds, SE ~0.009. 400 pe SE 0.024 tha aur test flaky tha
# (ek baar 0.427 aaya, mauka 0.375 - z=2.1). Uniform hona 8000 samples
# pe alag se jaancha gaya tha: z=+0.48, aur pehla sthan har index pe
# barabar. To shor tha, bias nahi - par test ko shor pe nahi tikne dena.
hits = [group_scores(yy, flat, R(s))["hit"] for s in range(3000)]
rate = float(np.mean(hits))
exp = TOP / len(yy)
check(f"ek-hi-number model ka hit ~ mauka ({exp:.3f})",
      bool(abs(rate - exp) < 0.03), True)
print(f"      (mila {rate:.3f}, mauka {exp:.3f})")

# Aur wahi model har baar WAHI jawab na de - warna wo kram hai, mauka nahi
uniq = {tuple(sorted(group_scores(yy, flat, R(s)).items())) for s in range(5)}
check("alag seed -> alag nateeja (kram tay nahi)", len(uniq) > 1, True)

print("\nchhote samooh chhoot jate hain")
check("3 dawa -> None (top-3 ka matlab nahi)",
      group_scores([1.0, 0.0, -1.0], [1.0, 0.0, -1.0], R()), None)
check("4 dawa -> chalta hai",
      group_scores([1.0, 0.5, 0.0, -1.0], [1.0, 0.5, 0.0, -1.0],
                   R())["hit"], 1.0)
check("MIN_GROUP 3 se bada", bool(MIN_GROUP > TOP), True)

print("\nNaN prediction -> samooh chhodo, 0 maan kar score NA karo")
check("NaN -> None",
      group_scores(y, [0.9, np.nan, 0.0, -0.5, -1.2, -2.0], R()), None)

print("\nrandom_hit sahi nikalta hai")
s = group_scores(yy, yy, R())
check("random_hit = 3/n", round(s["random_hit"], 4), round(3 / 8, 4))
check("n sahi", s["n"], 8)

print("\noverlap ka ganit")
# asli top3 = d0,d1,d2. Model top3 = d0,d1,d5 -> 2/3
s = group_scores(yy, [+9, +8, -9, -8, -7, +7], R())
check("2 me se 3 -> overlap 2/3", round(s["overlap"], 4),
      round(2 / 3, 4))

# ---------------------------------------------------------------- run_topk
print("\nstable_seed: har process me WAHI (hash() nahi chalta)")
import subprocess                                            # noqa: E402
HERE = os.path.dirname(os.path.abspath(__file__))
code = ("import sys; sys.path.insert(0, %r); "
        "from run_topk import stable_seed; print(stable_seed('ABL1','T315I',0,20))"
        % HERE)
outs = {subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True).stdout.strip() for _ in range(3)}
check("teen alag process -> ek hi seed", len(outs), 1)
pyh = {subprocess.run([sys.executable, "-c", "print(hash('ABL1'))"],
                      capture_output=True, text=True).stdout.strip()
       for _ in range(3)}
check("(aur hash() sach me badalta hai)", len(pyh) > 1, True)

print("\ndedupe: ek dawa, ek pankti")
import pandas as pd                                          # noqa: E402
from run_topk import dedupe                                  # noqa: E402
sub = pd.DataFrame({"drug_key": ["D1", "D1", "D2", "D3"],
                    "ddG": [-1.0, -3.0, 0.5, 0.0],
                    "pred": [-1.0, -1.0, 0.4, 0.1]})
g = dedupe(sub)
check("4 rows -> 3 dawa", len(g), 3)
check("D1 ka ddG median", float(g.loc["D1", "ddG"]), -2.0)
check("D1 ka pred ausat", float(g.loc["D1", "pred"]), -1.0)

# Asli khatra: bina dedupe ek dawa top-3 ki kai seat le leti
many = pd.DataFrame({"drug_key": ["D1"] * 3 + ["D2", "D3", "D4", "D5"],
                     "ddG": [2.0, 2.0, 2.0, 1.0, 0.5, 0.0, -1.0],
                     "pred": [9.0, 9.0, 9.0, 1.0, 0.5, 0.0, -1.0]})
check("bina dedupe D1 teen seat leti", int((many.pred == 9.0).sum()), 3)
check("dedupe ke baad ek", int((dedupe(many).pred == 9.0).sum()), 1)

print()
if FAIL:
    print(f"✗ {len(FAIL)} test fail: {FAIL}")
    sys.exit(1)
print("✓ saare test paas")
