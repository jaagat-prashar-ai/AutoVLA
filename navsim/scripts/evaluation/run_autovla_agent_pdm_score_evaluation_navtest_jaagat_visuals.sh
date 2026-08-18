#!/bin/bash
# Small visualization pass: same setup as the smoke test but renders the
# per-scene camera+BEV+trajectory figure for a handful of scenes.
set -euo pipefail

export TOKENIZERS_PARALLELISM=false

BASE=/media/training_data/jaagat-prashar/navsim_autovla_eval
OPENSCENE_ROOT=/media/training_data/ishaan.rawal/navsim/dataset

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$OPENSCENE_ROOT/maps"
export NAVSIM_EXP_ROOT="$BASE/exp"
export NAVSIM_DEVKIT_ROOT="/home/jaagat-prashar/workspace/research-project-template-main/autovla/AutoVLA/navsim"
export OPENSCENE_DATA_ROOT="$OPENSCENE_ROOT"

export PYTHONPATH="./navsim:${PYTHONPATH:-}"

TRAIN_TEST_SPLIT=navtest
CHECKPOINT="$BASE/checkpoints/AutoVLA_PDMS_89.ckpt"
CACHE_PATH="$BASE/dataset/nuplan/navtest_metric_cache"
JSON_DATA_PATH="$BASE/dataset/nuplan/navtest_nocot"
SENSOR_DATA_PATH="$OPENSCENE_ROOT/sensor_blobs/test"
CONFIG_PATH="./config/training/qwen2.5-vl-3B-nuplan-navtest-eval.yaml"
LORA=false

CUDA_VISIBLE_DEVICES=1 python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_cot.py \
  train_test_split=$TRAIN_TEST_SPLIT \
  train_test_split.scene_filter.max_scenes=6 \
  +save_visualization=true \
  agent=autovla_agent \
  +agent.config_path="$CONFIG_PATH" \
  +agent.checkpoint_path="$CHECKPOINT" \
  +agent.sensor_data_path="$SENSOR_DATA_PATH" \
  +agent.lora_conf.use_lora=$LORA \
  metric_cache_path=$CACHE_PATH \
  json_data_path=$JSON_DATA_PATH \
  experiment_name=autovla_agent_navtest_jaagat_visuals
