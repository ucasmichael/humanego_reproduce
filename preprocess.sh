conda activate humanego
cd /data/lyx/humanego_reproduce/WiLoR
/data/lyx/miniconda3/envs/wilor/bin/python export_realsense_humanego_handposes.py --session /data/lyx/datasets/egodata/data0617_depth_ir/20260617_155420_318122300934 --out /data/lyx/datasets/egodata/data0617_depth_ir/20260617_155420_318122300934/hand_poses_wilor.jsonl --side right --visualize
/data/lyx/miniconda3/envs/wilor/bin/python /data/lyx/humanego_reproduce/realsense_collect_data/prepare_humanego_session.py \
  --raw-session /data/lyx/datasets/egodata/data0617_depth_ir/20260617_155420_318122300934 \
  --humanego-root /data/lyx/HumanEgo \
  --task pick_banana_into_the_pot \
  --index 1 \
  --source-type teaching \
  --side right \
  --hand-poses /data/lyx/datasets/egodata/data0617_depth_ir/20260617_155420_318122300934/hand_poses_wilor.jsonl \
  --force
export HF_ENDPOINT=https://hf-mirror.com
cd /data/lyx/HumanEgo
python -m preprocess.RobotPreprocess --session_path /data/lyx/HumanEgo/data/pick_banana_into_the_pot/teaching/teaching_pick_banana_into_the_pot_001 --task pick_banana_into_the_pot --start_from auto