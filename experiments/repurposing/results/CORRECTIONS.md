# Bina retraining kitne kinases 'badhaye' ja sakte hain?

Paimana: **kitne checkable targets (146) pe koi jaani-pehchani dawa top-5 me aati hai.** Model, features, training — kuch nahi badla; sirf ranking ka transform aur molecule filter badla.

## top-5 recall (jitna zyada utna behtar)

| transform             |   MW 250-700 |   koi nahi |
|:----------------------|-------------:|-----------:|
| dawa ka mean ghatao   |           43 |         38 |
| dawa ka z-score       |           43 |         42 |
| dawa z + target z     |           42 |         38 |
| dono mean ghatao      |           43 |         38 |
| raw (jaisa hai)       |           45 |         23 |
| target ka mean ghatao |           45 |         23 |
| target ka z-score     |           45 |         23 |

## Poora table

| transform             | filter     | top5   | top10   | top20   | top50   |
|:----------------------|:-----------|:-------|:--------|:--------|:--------|
| raw (jaisa hai)       | koi nahi   | 23/134 | 24/134  | 24/134  | 41/134  |
| raw (jaisa hai)       | MW 250-700 | 45/134 | 46/134  | 49/134  | 58/134  |
| dawa ka mean ghatao   | koi nahi   | 38/134 | 42/134  | 48/134  | 57/134  |
| dawa ka mean ghatao   | MW 250-700 | 43/134 | 52/134  | 56/134  | 59/134  |
| dawa ka z-score       | koi nahi   | 42/134 | 46/134  | 49/134  | 55/134  |
| dawa ka z-score       | MW 250-700 | 43/134 | 52/134  | 52/134  | 57/134  |
| target ka mean ghatao | koi nahi   | 23/134 | 24/134  | 24/134  | 41/134  |
| target ka mean ghatao | MW 250-700 | 45/134 | 46/134  | 49/134  | 58/134  |
| target ka z-score     | koi nahi   | 23/134 | 24/134  | 24/134  | 41/134  |
| target ka z-score     | MW 250-700 | 45/134 | 46/134  | 49/134  | 58/134  |
| dono mean ghatao      | koi nahi   | 38/134 | 42/134  | 48/134  | 57/134  |
| dono mean ghatao      | MW 250-700 | 43/134 | 52/134  | 56/134  | 59/134  |
| dawa z + target z     | koi nahi   | 38/134 | 41/134  | 42/134  | 50/134  |
| dawa z + target z     | MW 250-700 | 42/134 | 47/134  | 52/134  | 59/134  |

**Shuruaat:** 23/134 · **Sabse achha:** 45/134 (raw (jaisa hai) + MW 250-700)
