#!/usr/bin/env bash
# Add FlashOcc to a pod that already ran setup_fbocc_pod.sh.
#
# Why this is a second script rather than an edit to the first: the FB-OCC
# environment is what produced every published number in this project, and it
# must stay reproducible byte for byte. This script only ADDS, and switches
# backbone by PYTHONPATH order. Nothing FB-OCC depends on is modified.
#
# Discovered live on 2026-10-29, A40, RunPod PyTorch 2.8.0 / Ubuntu 24.04.
# Six walls, none of them in any documentation:
#
#   1. FB-BEV's mmdet3d is missing LSSViewTransformerBEVStereo and BEVStereo4D,
#      which BEVStereo4DOCC inherits from.          -> BEVDet dev3.0 alongside
#   2. BEVDet's mmdet3d imports lyft_dataset_sdk and pandas, neither installed
#      by the FB-OCC script.                        -> --no-deps installs
#   3. FlashOCC configs inherit from _base_ files at
#      $FLASHOCC/mmdetection3d/.                    -> symlink to BEVDet
#   4. Both repos register MEGVIIEMAHook and friends; mmcv refuses duplicates.
#                                                   -> register_module(force=True)
#   5. ray_metrics.py JIT-builds dvr from RELATIVE paths and dies unless cwd is
#      the FlashOCC root. RayIoU is a different benchmark we never use.
#                                                   -> absolute paths, fail soft
#   6. FlashOCC ships its own CUDA ops, unbuilt, behind projects/setup.py
#      (not a root setup.py, which is why it is easy to miss).
#                                                   -> build_ext --inplace
#
# Verified end state:
#   detector BEVStereo4DOCC | head BEVOCCHead2D | num_classes 18
#
#   source /opt/env.sh && bash setup_flashocc_pod.sh
set -euo pipefail

say() { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }

export CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11

# ------------------------------------------------- wall 2: missing imports
say "dependencies BEVDet's mmdet3d needs and FB-OCC never installed"
# --no-deps throughout: the full dependency sets try to bump numpy off 1.23.5,
# which takes the whole stack down with it.
pip install --no-deps lyft_dataset_sdk "pandas==1.5.3" pytz python-dateutil
python -c "import numpy; assert numpy.__version__=='1.23.5', numpy.__version__; print('numpy pin intact')"

# ------------------------------------------------- wall 1: the base classes
say "BEVDet dev3.0 (FlashOcc's actual base)"
[ -d /opt/BEVDet ] || git clone -b dev3.0 --depth 1 \
  https://github.com/HuangJunJie2017/BEVDet /opt/BEVDet

python - <<'PY'
import pathlib, re
p = pathlib.Path('/opt/BEVDet/mmdet3d/__init__.py')
s = p.read_text()
t = re.sub(r"mmcv_maximum_version\s*=\s*'[^']+'", "mmcv_maximum_version = '1.8.0'", s)
p.write_text(t)
print('mmcv ceiling lifted' if t != s else 'no mmcv ceiling found')
PY

say "BEVDet CUDA ops, in place (no pip install -e: it would re-register mmdet3d)"
( cd /opt/BEVDet && python setup.py build_ext --inplace >/tmp/bevdet_ops.log 2>&1 ) \
  || { tail -20 /tmp/bevdet_ops.log; exit 1; }
find /opt/BEVDet -name "bev_pool_v2_ext*.so" | head -1

# ------------------------------------------------- FlashOCC itself
say "FlashOCC"
[ -d /opt/FlashOCC ] || git clone --depth 1 https://github.com/Yzichen/FlashOCC /opt/FlashOCC

# wall 3
[ -e /opt/FlashOCC/mmdetection3d ] || ln -s /opt/BEVDet /opt/FlashOCC/mmdetection3d
test -f /opt/FlashOCC/mmdetection3d/configs/_base_/datasets/nus-3d.py

# wall 4 -- where both repos define a name, the plugin is the one the configs
# were written against, so the plugin winning is the correct resolution.
say "letting the plugin override duplicate registrations"
if ! grep -rq "register_module(force=True)" /opt/FlashOCC/projects/mmdet3d_plugin/; then
  cp -r /opt/FlashOCC/projects/mmdet3d_plugin /opt/FlashOCC/projects/mmdet3d_plugin.orig
  grep -rl "register_module()" /opt/FlashOCC/projects/mmdet3d_plugin/ \
    | xargs sed -i 's/register_module()/register_module(force=True)/g'
fi

# wall 5
say "ray_metrics: absolute paths, non-fatal JIT"
python - <<'PY'
import pathlib, re
p = pathlib.Path('/opt/FlashOCC/projects/mmdet3d_plugin/core/evaluation/ray_metrics.py')
s = p.read_text()
if '[patch] dvr' in s:
    print('already patched'); raise SystemExit
m = re.search(r'^dvr = load\(.*?\)\s*$', s, re.M)
assert m, 'dvr load() line not found'
new = (
"import os as _os\n"
"_R = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', '..', '..'))\n"
"try:\n"
"    dvr = load('dvr',\n"
"               sources=[_os.path.join(_R, 'lib/dvr/dvr.cpp'), _os.path.join(_R, 'lib/dvr/dvr.cu')],\n"
"               verbose=True, extra_cuda_cflags=['-allow-unsupported-compiler'])\n"
"except Exception as _e:\n"
"    print('[patch] dvr JIT build skipped:', type(_e).__name__, _e)\n"
"    dvr = None"
)
p.write_text(s[:m.start()] + new + s[m.end():])
print('patched')
PY

# wall 6
say "FlashOCC plugin CUDA ops (projects/setup.py, not a root setup.py)"
( cd /opt/FlashOCC/projects && python setup.py build_ext --inplace >/tmp/flashocc_ops.log 2>&1 ) \
  || { tail -20 /tmp/flashocc_ops.log; exit 1; }
find /opt/FlashOCC/projects/mmdet3d_plugin/ops -name "*.so"

# ------------------------------------------------- checkpoints
say "checkpoints (Occ3D-nuScenes table of the FlashOCC README)"
pip install -q gdown
mkdir -p /opt/ckpts
# M2 = FlashOCC-4D-Stereo 2f, mIoU 37.84.  M1 = FlashOCC 1f, mIoU 32.08.
# Do NOT take the second README table: its "M1" row is 15.41 on another
# benchmark with a different file id.
[ -f /opt/ckpts/flashocc-m2.pth ] || gdown 12WYaCdoZA8-A6_oh6vdLgOmqyEc3PNCe -O /opt/ckpts/flashocc-m2.pth
[ -f /opt/ckpts/flashocc-m1.pth ] || gdown 1k9BzXB2nRyvXhqf7GQx3XNSej6Oq6I-B -O /opt/ckpts/flashocc-m1.pth
ls -lh /opt/ckpts/

# ------------------------------------------------- verify
say "building the model, which is the only proof any of the above worked"
cd /opt
PYTHONPATH=/opt/BEVDet:/opt/FlashOCC:${PYTHONPATH:-} python - <<'PY'
from mmcv import Config
from mmdet3d.models import build_model
import importlib, mmdet3d
print('mmdet3d from', mmdet3d.__file__)
cfg = Config.fromfile('/opt/FlashOCC/projects/configs/flashocc/flashocc-r50-4d-stereo.py')
if cfg.get('plugin'):
    importlib.import_module(cfg.plugin_dir.replace('/', '.').rstrip('.'))
cfg.model.train_cfg = None
m = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
assert type(m).__name__ == 'BEVStereo4DOCC', type(m).__name__
assert m.occ_head.num_classes == 18, m.occ_head.num_classes
assert hasattr(type(m), 'simple_test_occ')
print('OK  BEVStereo4DOCC / BEVOCCHead2D / num_classes 18 / simple_test_occ present')
PY

cat <<'EOF'

done. Every FlashOcc command needs BEVDet FIRST on the path:

  export PYTHONPATH=/opt/BEVDet:/opt/FlashOCC:$PYTHONPATH

Leaving FB-BEV first silently gives you the wrong mmdet3d.
EOF
