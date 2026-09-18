#!/usr/bin/env bash
# Rebuild the FB-OCC inference environment on a bare RunPod box.
#
# Why this file exists
# --------------------
# A RunPod pod without a network volume loses its container disk the moment it
# stops. The environment below took about three hours to work out the first
# time, almost all of it spent discovering that a 2023 research stack does not
# install on a 2026 image without five separate version fights. None of that is
# worth repeating, so it is written down.
#
# Tested on: RunPod PyTorch 2.8.0 image, RTX A6000, CUDA 12.8 host, driver 580.
# Takes ~25 minutes, most of it downloading.
#
#   bash setup_fbocc_pod.sh
#
# The five walls, in the order they appear, so a failure is diagnosable:
#
#   1. The image ships Python 3.12; torch 1.13 has no 3.12 build.  -> micromamba
#   2. mmcv-full has no 1.6.x wheel for torch 1.13.                -> use 1.7.0
#   3. mmdet3d asserts mmcv < 1.7.0, and 1.7.0 is the oldest wheel -> lift bound
#   4. Host nvcc is 12.8, torch was built with 11.7.               -> env toolkit
#   5. Host g++ is 13.3, CUDA 11.7 caps at 11.5.                   -> apt g++-11
#   6. setup.py develop -> easy_install -> cannot build modern sdists -> pip -e
#   7. points dropped from the pipeline, forward_test still does points[0]
#
# Plus two defects in the published repos:
#   * FB-BEV's mmdet3d/models/fbbev/custom_ops ships only __init__.py; all
#     three modules it advertises are missing. They are TensorRT-only.
#   * tools/create_data_bevdet.py references an undefined name NUSCENES.
set -euo pipefail

MAMBA_ROOT=/opt/mm
ENV_NAME=occ
CKPT_URL="https://github.com/zhiqi-li/storage/releases/download/v1.0/fbocc-r50-cbgs_depth_16f_16x4_20e.pth"

say() { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------- 1. python
say "micromamba + python 3.8"
mkdir -p $MAMBA_ROOT && cd $MAMBA_ROOT
[ -x $MAMBA_ROOT/bin/micromamba ] || \
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba
export MAMBA_ROOT_PREFIX=$MAMBA_ROOT
eval "$($MAMBA_ROOT/bin/micromamba shell hook -s bash)"
micromamba create -y -n $ENV_NAME python=3.8 -c conda-forge
micromamba activate $ENV_NAME

# ---------------------------------------------------------------- 2. torch
say "torch 1.13.1+cu117"
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
  --index-url https://download.pytorch.org/whl/cu117
python -c "import torch; assert torch.cuda.is_available(); \
  print('torch', torch.__version__, torch.cuda.get_device_name(0))"

# ---------------------------------------------------------------- 3. openmm
say "mmcv / mmdet / mmseg"
# setuptools 59.5.0 : newer ones dropped the distutils hooks these builds use
# numpy 1.23.5      : 1.24 deleted np.long and np.bool, still called by mmdet3d
# yapf 0.40.1       : 0.40.2 broke mmcv's config parser
# pillow <10        : pillow 10 removed constants torchvision 0.14 expects
pip install setuptools==59.5.0 numpy==1.23.5 yapf==0.40.1 "pillow<10"
pip install mmcv-full==1.7.0 \
  -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html
pip install mmdet==2.28.2 mmsegmentation==0.30.0 numba==0.56.4

# ---------------------------------------------------------------- 4. cuda
say "CUDA 11.7 toolkit (host nvcc is too new for this torch)"
micromamba install -y -c "nvidia/label/cuda-11.7.1" cuda-toolkit
export CUDA_HOME=$MAMBA_ROOT/envs/$ENV_NAME
export PATH=$CUDA_HOME/bin:$PATH
nvcc --version | tail -1

say "g++ 11 (CUDA 11.7 refuses anything above 11.5)"
apt-get update -qq
apt-get install -y gcc-11 g++-11 ninja-build >/dev/null
export CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11
export TORCH_CUDA_ARCH_LIST="8.6"        # A6000; adjust for another GPU

# ---------------------------------------------------------------- 5. fb-bev
say "FB-BEV"
cd /opt
[ -d FB-BEV ] || git clone --depth 1 https://github.com/NVlabs/FB-BEV.git
pip install spconv-cu117 timm einops pyquaternion plyfile trimesh \
            IPython nuscenes-devkit
pip install numpy==1.23.5                # the above will try to bump it

say "patching the two published defects"
python - <<'PY'
import re, pathlib
root = pathlib.Path('/opt/FB-BEV/mmdet3d')

# custom_ops advertises grid_sampler, bev_pool_v2 and multi_scale_deformable_attn
# and ships none of them -- the package exists only for TensorRT export.
(root / 'models/fbbev/custom_ops/__init__.py').write_text('')
real = 'from mmdet3d.ops.bev_pool_v2.bev_pool import bev_pool_v2'
pat  = re.compile(r'from\s+mmdet3d\.models\.fbbev\.custom_ops\.bev_pool_v2\s+import\s+bev_pool_v2(\s+as\s+\w+)?')
dead = re.compile(r'^(\s*)from\s+mmdet3d\.models\.fbbev\.custom_ops\.(grid_sampler|multi_scale_deformable_attn)\s+import\s+.*$', re.M)
for p in root.rglob('*.py'):
    s = p.read_text()
    if 'custom_ops' not in s:
        continue
    t = dead.sub(r'\1pass  # TRT-only op, package ships empty',
                 pat.sub(lambda m: real + (m.group(1) or ''), s))
    if t != s:
        p.write_text(t)

# the TRT detector needs those missing modules; we never export to TensorRT
d = root / 'models/fbbev/detectors/__init__.py'
d.write_text(d.read_text().replace(
    'from .fbocc_trt import FBOCCTRT',
    '# from .fbocc_trt import FBOCCTRT  # TRT variant, plugin not shipped'))

# create_data_bevdet.py refers to an undefined NUSCENES instead of `dataset`
c = pathlib.Path('/opt/FB-BEV/tools/create_data_bevdet.py')
c.write_text(c.read_text().replace("f'./data/{NUSCENES}'", "f'./data/{dataset}'"))
print('patched')
PY

# WALL 6 (found 17 Sep 2026): `python setup.py develop` resolves dependencies
# through easy_install, which fetched the newest scikit-image sdist. Modern
# packages ship pyproject.toml with no setup.py, and easy_install -- deprecated
# for years -- cannot build those. Install the dep from a wheel first, then use
# pip, which does not invoke easy_install at all.
say "scikit-image from a wheel (easy_install cannot build modern sdists)"
pip install "scikit-image<0.22" "networkx<3" tensorboard
pip install numpy==1.23.5

# WALL 7 (found 17 Sep 2026): with points dropped from the pipeline,
# forward_test defaults `points` to None and then does points[0]. Guard the
# camera branch only -- the lidar-only branch below it is unreachable here and
# `points` cannot be None there by construction.
say "patching forward_test to accept points=None"
python - <<'PYP'
import pathlib
p = pathlib.Path('/opt/FB-BEV/mmdet3d/models/fbbev/detectors/fbocc.py')
s = p.read_text()
anchor = "['dist_tta']:\n                return self.simple_test(points[0],"
new = ("['dist_tta']:\n                return self.simple_test("
       "None if points is None else points[0],")
assert s.count(anchor) == 1, f'expected 1, found {s.count(anchor)}'
p.write_text(s.replace(anchor, new))
t = p.read_text()
assert new in t and anchor not in t, 'patch did not apply'
assert t.count('simple_test(points[0]') == 1, 'lidar branch should be untouched'
print('patched and verified')
PYP

say "building FB-BEV (compiles bev_pool_v2)"
cd /opt/FB-BEV && pip install -e . --no-deps --no-build-isolation 2>&1 | tail -5
cd /tmp && python -c "import mmdet3d; print('mmdet3d', mmdet3d.__version__, '->', mmdet3d.__file__)"

# ---------------------------------------------------------------- 6. weights
say "checkpoint"
mkdir -p /opt/ckpts
[ -f /opt/ckpts/fbocc-r50.pth ] || curl -L -o /opt/ckpts/fbocc-r50.pth "$CKPT_URL"
ls -lh /opt/ckpts/fbocc-r50.pth

# ---------------------------------------------------------------- 7. config
say "inference-only config"
cat > /opt/FB-BEV/occupancy_configs/fb_occ/fbocc_infer.py <<'PYCFG'
# Inference only, camera-only.
#
# LoadPointsFromFile is ABSENT BY DESIGN. The network never consumes points --
# extract_feat reads them only when a pts_voxel_encoder exists, and this config
# has none -- so loading 138 GB of LiDAR blobs to fill a tensor that is then
# discarded is pure cost. An earlier version of this script shipped a pipeline
# that DID load them, which is the superseded variant; keeping points also
# requires downloading LiDAR we deliberately never fetch, and the run then dies
# on a missing .pcd.bin.
#
# Dropping them requires the forward_test guard patched below, because
# forward_test defaults points to None and then immediately subscripts it.
_base_ = ['./fbocc-r50-cbgs_depth_16f_16x4_20e.py']
data_root = 'data/nuscenes/'
test_pipeline = [
    dict(type='CustomDistMultiScaleFlipAug3D', tta=False, transforms=[
        dict(type='PrepareImageInputs', data_config={{_base_.data_config}}),
        dict(type='LoadAnnotationsBEVDepth',
             bda_aug_conf={{_base_.bda_aug_conf}},
             classes={{_base_.class_names}}, is_train=False),
        dict(type='DefaultFormatBundle3D',
             class_names={{_base_.class_names}}, with_label=False),
        dict(type='Collect3D', keys=['img_inputs']),
    ])
]
data = dict(samples_per_gpu=1, workers_per_gpu=2,
            test=dict(pipeline=test_pipeline,
                      ann_file=data_root + 'bevdetv2-nuscenes_infos_val.pkl'))
PYCFG

cat > /opt/env.sh <<'ENVEOF'
export MAMBA_ROOT_PREFIX=/opt/mm
eval "$(/opt/mm/bin/micromamba shell hook -s bash)"
micromamba activate occ
export CUDA_HOME=/opt/mm/envs/occ
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST="8.6"
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export PYTHONPATH=/opt/FB-BEV:$PYTHONPATH
ENVEOF

say "done -- source /opt/env.sh in every new shell"
cat <<'EOF'

Next:
  source /opt/env.sh
  # put nuScenes under /opt/FB-BEV/data/nuscenes (samples/, v1.0-*/ , maps/)
  cd /opt/FB-BEV && python tools/create_data_bevdet.py    # writes the info pkl
  python /opt/run_fbocc.py /opt/preds/<name>              # ~0.23 s/keyframe

For mini instead of trainval, first:
  sed -i "s/^VERSION *= *'v1.0-trainval'/VERSION = 'v1.0-mini'/" \
    /opt/FB-BEV/tools/create_data_bevdet.py
EOF
