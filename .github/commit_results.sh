#!/usr/bin/env bash
# Nateeja repo me commit karo - bharose ke saath.
#
# Kyun ye script: pehle har workflow khud `git add` -> `git commit` ->
# `git pull --rebase --autostash` -> `git push` karta tha. v3 ka 4-ghante
# wala 12-seed run isi me nateeja kha gaya:
#   - run ke dauraan maine main pe naye commits push kiye
#   - commit ke baad `pull --rebase` ne conflict mara (dono taraf wahi
#     results/ files badli thi)
#   - conflict `|| true` me dab gaya, rebase adhoora chhoot gaya
#   - us haalat me HEAD remote ke commit pe tha, to `git push` ne
#     "Everything up-to-date" kaha aur SUCCESS de diya
#   - job green, step green, nateeja gayab
#
# Isliye ab rebase hi nahi karte. Nateeja alag jagah rakh kar branch ko
# origin pe reset karte hain, phir nateeja wapas rakh ke commit karte hain -
# conflict ka mauka hi nahi. Aur aakhir me VERIFY karte hain ki commit sach
# me remote pe pahuncha; na pahuncha to exit 1, taaki job red ho.
#
# istemaal: commit_results.sh <branch> "<commit message>" <path> [path...]
set -uo pipefail

BRANCH="$1"; shift
MSG="$1"; shift
PATHS=("$@")

if [ "${#PATHS[@]}" -eq 0 ]; then
  echo "koi path nahi diya"; exit 1
fi

git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"

STAGE="$(mktemp -d)"
FOUND=0
for p in "${PATHS[@]}"; do
  if [ -e "$p" ]; then
    mkdir -p "$STAGE/$(dirname "$p")"
    cp -r "$p" "$STAGE/$(dirname "$p")/"
    FOUND=1
  else
    echo "note: $p maujood nahi, chhod rahe hain"
  fi
done
if [ "$FOUND" -eq 0 ]; then
  echo "::warning::commit karne layak kuch nahi mila"; exit 0
fi

for attempt in 1 2 3 4 5; do
  echo "--- koshish $attempt ---"
  git fetch -q origin "$BRANCH" || { sleep 5; continue; }
  # -f zaroori hai: nateeja working tree me pada hai aur plain checkout use
  # overwrite karne se mana kar deta hai. Safe hai kyunki nateeja $STAGE me
  # alag rakha hua hai aur neeche wapas copy hota hai.
  git checkout -f -q -B "$BRANCH" "origin/$BRANCH" || { sleep 5; continue; }
  git reset -q --hard "origin/$BRANCH" || { sleep 5; continue; }

  for p in "${PATHS[@]}"; do
    [ -e "$STAGE/$p" ] || continue
    mkdir -p "$(dirname "$p")"
    cp -r "$STAGE/$p" "$(dirname "$p")/"
  done

  git add -f -- "${PATHS[@]}" 2>/dev/null || true
  if git diff --cached --quiet; then
    echo "remote pe already wahi hai - commit ki zaroorat nahi"
    exit 0
  fi

  git commit -q -m "$MSG" || { echo "commit fail"; sleep 5; continue; }
  LOCAL="$(git rev-parse HEAD)"

  if git push -q origin "HEAD:$BRANCH"; then
    git fetch -q origin "$BRANCH"
    REMOTE="$(git rev-parse "origin/$BRANCH")"
    if [ "$LOCAL" = "$REMOTE" ]; then
      echo "✓ commit remote pe pahunch gaya: ${LOCAL:0:7}"
      exit 0
    fi
    echo "push ke baad bhi remote pe nahi ($LOCAL vs $REMOTE) - dobara"
  else
    echo "push reject hua (remote aage nikal gaya) - dobara"
  fi
  sleep $((attempt * 5))
done

echo "::error::5 koshish ke baad bhi nateeja commit nahi hua - result khone se behtar hai job red ho"
exit 1
