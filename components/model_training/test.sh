#!/usr/bin/env bash

TMP=$(dirname $(dirname $(pwd)))/tmp/

docker run -it  \
  -v "$TMP":/mnt/kubeflow/ \
  -m 96g \
  --cpus 24 \
  "${OWNER}/kf-training:${KF_PIPELINE_VERSION}" \
  --dir_data=/mnt/kubeflow/dataset \
  --dir_checkpoints=/mnt/kubeflow/models \
  --batch_size=131 \
  --learning_rate=0.005 \
  --log_step=1 \
  --save_step=10 \
  --epochs=1 | tee $(pwd)/log.txt
