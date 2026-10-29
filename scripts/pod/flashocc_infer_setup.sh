#!/usr/bin/env bash
# Everything between setup_flashocc_pod.sh and the first export.
#
# Separate from setup_flashocc_pod.sh because that script is environment only
# and this one touches DATA LAYOUT and the inference pipeline. Idempotent:
# safe to re-run.
#
#   source /opt/env.sh && bash scripts/pod/flashocc_infer_setup.sh
#
# Assumes nuScenes camera keyframes are already at /workspace/nuscenes:
#   curl -sL "$BASE/v1.0-trainval${i}_keyframes.tgz" \
#     | tar -xz -C /workspace/nuscenes --wildcards 'samples/CAM_*'
# for i in 01..10, plus v1.0-trainval_meta.tgz extracted there.
# 32 GB written, ~28 min. See A65.
set -euo pipefail
say() { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }

NUSC=${NUSC:-/workspace/nuscenes}

say "data symlink"
mkdir -p /opt/BEVDet/data
ln -sfn "$NUSC" /opt/BEVDet/data/nuscenes
ls /opt/BEVDet/data/nuscenes/ >/dev/null

# ---------------------------------------------------------------- data prep
# create_data_bevdet.py only CHECKS that each LiDAR file exists; it stores the
# path and never opens it. This project is camera-only, so the check is the
# sole reason 138 GB of point clouds would have to be on disk.
say "removing the LiDAR existence check from the info builder"
python - <<'PY'
import pathlib
p = pathlib.Path('/opt/BEVDet/tools/data_converter/nuscenes_converter.py')
s = p.read_text()
old = "        mmcv.check_file_exist(lidar_path)"
if old in s:
    p.write_text(s.replace(old,
        "        # LiDAR deliberately not downloaded: camera-only project. The\n"
        "        # path is recorded in the info dict but never opened.\n"
        "        # mmcv.check_file_exist(lidar_path)"))
    print('existence check removed')
else:
    print('already removed')
PY

cat <<'EOF'

  If bevdetv3-nuscenes_infos_val.pkl does not exist yet, build it now
  (~20 min, and it WILL fail at the very end on create_groundtruth_database,
  which reads point clouds and is training-only -- that failure is expected
  and harmless, the pickles are written before it):

      cd /opt/BEVDet && python tools/create_data_bevdet.py

EOF

if [ -f /opt/BEVDet/data/nuscenes/bevdetv3-nuscenes_infos_val.pkl ]; then
  say "info pickle present; adding the bevdetv2 alias some configs expect"
  ( cd /opt/BEVDet/data/nuscenes &&
    ln -sfn bevdetv3-nuscenes_infos_val.pkl bevdetv2-nuscenes_infos_val.pkl )
  ls -lh /opt/BEVDet/data/nuscenes/*.pkl
fi

# ---------------------------------------------------------------- inference
say "camera-only inference configs for M2 and M1"
for v in "m2 flashocc-r50-4d-stereo" "m1 flashocc-r50"; do
  set -- $v
  cat > "/opt/FlashOCC/projects/configs/flashocc/flashocc_${1}_infer.py" <<PYCFG
# Inference only, camera-only. Mirrors fbocc_infer.py.
#
# LoadPointsFromFile is ABSENT BY DESIGN. This config has no pts_voxel_encoder,
# so the network never consumes points; loading LiDAR to fill a tensor that is
# then discarded would mean fetching blobs we deliberately never downloaded.
# Dropping them requires the forward_test guard applied by this same script.
_base_ = ['./${2}.py']

test_pipeline = [
    dict(type='PrepareImageInputs',
         data_config={{_base_.data_config}}, sequential=True),
    dict(type='LoadAnnotationsBEVDepth',
         bda_aug_conf={{_base_.bda_aug_conf}},
         classes={{_base_.class_names}}, is_train=False),
    dict(type='MultiScaleFlipAug3D',
         img_scale=(1333, 800), pts_scale_ratio=1, flip=False,
         transforms=[
             dict(type='DefaultFormatBundle3D',
                  class_names={{_base_.class_names}}, with_label=False),
             dict(type='Collect3D', keys=['img_inputs']),
         ])
]
data = dict(samples_per_gpu=1, workers_per_gpu=2,
            test=dict(pipeline=test_pipeline))
PYCFG
  echo "  wrote flashocc_${1}_infer.py"
done

# M1 is single-frame: sequential has no meaning there and the key is absent
# from its PrepareImageInputs signature in some revisions.
python - <<'PY'
import pathlib
p = pathlib.Path('/opt/FlashOCC/projects/configs/flashocc/flashocc_m1_infer.py')
s = p.read_text().replace(
    "data_config={{_base_.data_config}}, sequential=True),",
    "data_config={{_base_.data_config}}),")
p.write_text(s)
print('m1 config: sequential removed (single frame)')
PY

say "wall 7: forward_test defaults points to None and then subscripts it"
python - <<'PY'
import pathlib
old = "self.simple_test(points[0], img_metas[0], img_inputs[0],"
new = "self.simple_test(None if points is None else points[0], img_metas[0], img_inputs[0],"
hits = 0
for f in ['/opt/FlashOCC/projects/mmdet3d_plugin/models/detectors/bevdet.py',
          '/opt/FlashOCC/projects/mmdet3d_plugin/models/detectors/bevdepth.py',
          '/opt/BEVDet/mmdet3d/models/detectors/bevdet.py']:
    p = pathlib.Path(f)
    if not p.exists():
        continue
    s = p.read_text()
    if new in s:
        print('already guarded', f); hits += 1; continue
    if old in s:
        p.write_text(s.replace(old, new)); print('patched', f); hits += 1
print('files guarded:', hits)
assert hits, 'no forward_test call sites found -- upstream changed, investigate'
PY

cat <<'EOF'

ready. 50 frames first, ALWAYS, and run from /opt/BEVDet because the config's
data_root is relative:

  cd /opt/BEVDet
  PYTHONPATH=/opt/BEVDet:/opt/FlashOCC:$PYTHONPATH \
  python /workspace/odfm/scripts/pod/run_fbocc_voxel.py /workspace/preds_mini \
    --backbone flashocc \
    --cfg /opt/FlashOCC/projects/configs/flashocc/flashocc_m2_infer.py \
    --ckpt /opt/ckpts/flashocc-m2.pth --limit 50

  python /workspace/odfm/scripts/pod/mini_gate.py /workspace/preds_mini
EOF
