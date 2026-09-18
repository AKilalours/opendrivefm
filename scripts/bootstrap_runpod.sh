#!/usr/bin/env bash
# OpenDriveFM :: RunPod bootstrap
#
# Gets nuScenes val onto this pod as a compact packed array, using far less
# disk than the dataset itself.
#
# The idea
# --------
# nuScenes trainval ships as ten ~35 GB blobs. We need the 150 VALIDATION
# scenes only: published checkpoints are evaluated on val, and every analysis
# downstream of that runs post-hoc on cached logits. The blobs are not split by
# train/val, so each one still has to be downloaded -- but it does NOT have to
# be kept. This script downloads one blob, extracts only the keyframes that
# belong to val scenes, appends them to a packed uint8 array at 256x448, then
# deletes the blob before touching the next one.
#
#   peak disk  = one blob (~35 GB) + the growing pack (~14 GB) + metadata
#              ~= 50 GB, instead of the 350 GB the dataset occupies
#
# Sweeps are skipped entirely: the temporal window is four KEYFRAMES 0.5 s
# apart, all of which live under samples/. Sweeps are ~70% of the dataset and
# this project never reads one.
#
# Usage
# -----
#   1. Log in at nuscenes.org, open the download page for "Full dataset (v1.0)
#      -> trainval". Copy the link for EACH of the ten blobs plus the metadata
#      archive. The links carry a signed token and expire after a few hours, so
#      copy them right before running this.
#   2. Paste them, one per line, into urls.txt (metadata line first).
#   3. bash bootstrap_runpod.sh
#
# Re-running is safe: finished blobs are recorded in .done and skipped.
set -euo pipefail

# Everything is resolved to an absolute path BEFORE the cd below. The script
# works from $DATA, so a relative urls.txt or pack_nuscenes.py would be looked
# for inside the dataset directory rather than beside this script.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="${DATA:-/workspace/nuscenes}"
PACK="${PACK:-/workspace/pack}"
URLS="${URLS:-$HERE/urls.txt}"
[ -f "$URLS" ] || [ ! -f "$HERE/urls.txt" ] || URLS="$HERE/urls.txt"
PACKER="$HERE/pack_nuscenes.py"
H="${H:-256}"
W="${W:-448}"
# val  = 150 scenes, ~14 GB packed -- everything the paper strictly needs
# all  = 850 scenes, ~72 GB packed -- adds the training split
# The blobs are downloaded either way; SPLIT only decides what is kept out of
# them. Choosing "val" now and wanting "train" in week 5 means a second 350 GB
# download, so pick "all" if the volume has ~120 GB free.
SPLIT="${SPLIT:-val}"
# 0 lets the packer choose min(nproc, 64). Decoding 200k JPEGs is the only
# CPU-heavy step here and it parallelises almost perfectly.
WORKERS="${WORKERS:-0}"
# RunPod's /workspace is MooseFS over FUSE. Sequential I/O on it is fine -- the
# 29 GB blobs and the 70 GB memmap live there happily -- but each blob yields
# ~23,600 individual ~130 KB files, and writing then re-reading those across a
# network filesystem took 32 minutes per blob against 8 minutes to download
# them. So the small-file churn is done on the container's local disk and only
# the packed array goes back to the network volume. SCRATCH is wiped after each
# blob, so it never needs more than one blob's worth (~4 GB).
SCRATCH="${SCRATCH:-/root/odfm_scratch}"

mkdir -p "$DATA" "$PACK" "$SCRATCH"
cd "$DATA"

say() { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31m!! %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- 0. sanity
say "environment"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "no GPU visible (fine for this phase)"
python3 -c "import sys; print('python', sys.version.split()[0])"
echo "cpu cores: $(nproc)"
AVAIL=$(df -BG --output=avail "$DATA" | tail -1 | tr -dc '0-9')
echo "disk available at $DATA: ${AVAIL} GB (network volume)"
SAVAIL=$(df -BG --output=avail "$SCRATCH" | tail -1 | tr -dc '0-9')
echo "scratch at $SCRATCH: ${SAVAIL} GB (local disk)"
[ "$SAVAIL" -lt 10 ] && die "scratch needs ~10 GB of LOCAL disk; found ${SAVAIL} GB at $SCRATCH.
   Set SCRATCH=/some/local/path, or SCRATCH=$DATA to extract on the network volume (much slower)."
echo "split: $SPLIT"
if [ "$SPLIT" != "val" ] && [ "$AVAIL" -lt 120 ]; then
  die "SPLIT=$SPLIT needs ~120 GB (one blob plus a 72 GB pack); found ${AVAIL} GB"
fi
[ "$AVAIL" -lt 60 ] && die "need ~60 GB free to hold one blob plus the pack; found ${AVAIL} GB"

[ -f "$URLS" ] || die "no urls.txt found at $URLS -- put it beside this script, or set URLS=/path/to/urls.txt"
[ -f "$PACKER" ] || die "pack_nuscenes.py not found at $PACKER -- it must sit beside this script"
NURL=$(grep -cve '^\s*$' "$URLS")
echo "$NURL url(s) queued"

# ---------------------------------------------------------------- 1. deps
say "dependencies"
# pillow-simd is deliberately NOT used. It builds from source, and on a fresh
# image that build can stall for many minutes or hang outright -- which it did,
# silently, because the error was being discarded into /dev/null. Plain Pillow
# across 64 processes is not the bottleneck here; the download is.
if python3 -c "import nuscenes, PIL, numpy" 2>/dev/null; then
  echo "dependencies already present"
else
  timeout 900 pip install --no-input --progress-bar off nuscenes-devkit pillow \
    || die "dependency install failed or timed out; run it in the foreground to see why:
       pip install --no-input nuscenes-devkit pillow"
fi
# pigz decompresses a 35 GB blob across all cores instead of one. Without it
# gzip is single-threaded and becomes the bottleneck on a 128-core box.
# pigz is a nice-to-have, never a blocker: apt can hang on a lock or a stale
# mirror, and single-threaded gzip still gets the job done.
if ! command -v pigz >/dev/null || ! command -v aria2c >/dev/null; then
  timeout 300 bash -c "apt-get update -qq && apt-get install -y -qq pigz aria2" >/dev/null 2>&1 || true
fi
command -v aria2c >/dev/null \
  && echo "aria2c: parallel download enabled" \
  || echo "aria2c unavailable, curl single-stream (expect ~15 MB/s from S3)"
if command -v pigz >/dev/null; then echo "pigz: parallel decompression available";
else echo "pigz unavailable, gzip will be single-threaded"; fi

# nuScenes serves archives named ".tar" that are in fact gzip-compressed, and
# has served genuinely uncompressed tars in the past. Sniff each file instead
# of trusting the extension -- guessing wrong fails the extract with a useless
# error after the whole blob has already been downloaded.
tar_flags_for() {
  case "$(file -b --mime-type "$1")" in
    application/gzip|application/x-gzip)
        if command -v pigz >/dev/null; then echo "-I pigz"; else echo "-z"; fi ;;
    *)  echo "" ;;
  esac
}
python3 -c "import nuscenes, PIL, numpy; print('devkit ok')"

# A single S3 connection tops out around 15 MB/s, which turns 315 GB into six
# hours. The limit is per-connection, not per-host, so sixteen parallel range
# requests against the same object go roughly an order of magnitude faster.
# aria2c does that; curl is kept as a fallback so the script still works
# without it.
fetch() {   # fetch <url> <destination>
  local url="$1" out="$2"
  if command -v aria2c >/dev/null; then
    aria2c -x 16 -s 16 -k 10M --file-allocation=none --console-log-level=warn \
           --summary-interval=60 --retry-wait=5 -m 5 -c \
           -d "$(dirname "$out")" -o "$(basename "$out")" "$url"
  else
    curl -fL --retry 3 -C - --progress-bar -o "$out" "$url"
  fi
}

# ---------------------------------------------------------------- 2. metadata
# The first line must be the metadata archive: it is small and everything else
# depends on it, so a bad token fails here in seconds rather than after 35 GB.
say "metadata"
META_URL=$(head -1 "$URLS")
if [ ! -d "$DATA/v1.0-trainval" ]; then
  fetch "$META_URL" "$DATA/meta.tgz" || die "metadata download failed -- check the url in urls.txt"
  # shellcheck disable=SC2046
  tar $(tar_flags_for meta.tgz) -xf meta.tgz && rm -f meta.tgz
fi
[ -d "$DATA/v1.0-trainval" ] || die "v1.0-trainval/ missing after extract"
echo "metadata ok: $(ls "$DATA"/v1.0-trainval | wc -l) json files"

# ---------------------------------------------------------------- 3. manifest
# Which files do we actually want? Built once, from metadata alone, before a
# single image is downloaded.
say "building val manifest"
python3 "$PACKER" --stage manifest --data "$DATA" --pack "$PACK" --split "$SPLIT"

# ---------------------------------------------------------------- 4. blobs
say "blobs: download -> extract wanted -> pack -> delete"
touch .done
i=0
while read -r url; do
  [ -z "$url" ] && continue
  i=$((i + 1))
  [ "$i" -eq 1 ] && continue                      # line 1 was the metadata
  tag="blob$(printf '%02d' $((i - 1)))"
  grep -qx "$tag" .done && { echo "  $tag already done, skipping"; continue; }

  echo ""
  echo "  --- $tag : downloading"
  # Resumable either way: aria2c -c and curl -C - both continue a partial file.
  fetch "$url" "$DATA/$tag.tgz" \
    || die "$tag download failed -- re-run to resume; finished blobs are skipped"

  echo "  --- $tag : extracting only the keyframes val needs (-> local scratch)"
  rm -rf "${SCRATCH:?}"/samples
  mkdir -p "$SCRATCH"
  # -T feeds tar the wanted-path list; missing members are not an error because
  # any given blob holds only a slice of the dataset.
  # shellcheck disable=SC2046
  # --occurrence=1 is the difference between 30 minutes and 4. Without it tar
  # reads the ENTIRE 28 GiB archive even after it has already extracted every
  # file we asked for, because it cannot know there are no further matches --
  # and most of what it then reads is sweeps/, the ~70% this project never
  # touches. With it, tar exits as soon as each name has been seen once.
  # Measured on a synthetic archive of the same shape: 7.3x faster, byte-identical output.
  tar $(tar_flags_for "$tag.tgz") -xf "$tag.tgz" -C "$SCRATCH" \
      -T "$PACK/wanted_paths.txt" --occurrence=1 --ignore-failed-read 2>/dev/null || true

  echo "  --- $tag : packing"
  python3 "$PACKER" --stage pack --data "$SCRATCH" --pack "$PACK" --height "$H" --width "$W" --workers "$WORKERS"

  echo "  --- $tag : reclaiming disk"
  rm -f "$tag.tgz"
  rm -rf "${SCRATCH:?}"/samples "${SCRATCH:?}"/sweeps   # already copied into the pack
  rm -rf "$DATA/samples" "$DATA/sweeps"                 # from older runs, if any
  echo "$tag" >> .done
  df -BG --output=avail "$DATA" | tail -1 | xargs echo "     free now:"
done < "$URLS"

# ---------------------------------------------------------------- 5. verify
say "verify"
python3 "$PACKER" --stage verify --data "$DATA" --pack "$PACK"

cat <<EOF

Done. The pack lives in $PACK and is the only thing you need from here on:

  images_${H}x${W}.u8      packed uint8, (keyframes, 6, H, W, 3)
  lidar.npz                keyframe LiDAR for the occlusion ray-cast
  calib.json               intrinsics, extrinsics, ego poses per keyframe
  index.json               token / scene / split per row

Next: stop this pod. The pack is small enough that the baseline inference runs
on the cheap A6000, and everything after that runs on CPU.
EOF
