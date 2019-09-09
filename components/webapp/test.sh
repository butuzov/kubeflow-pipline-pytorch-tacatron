#!/usr/bin/env bash

TMP=$(dirname $(dirname $(pwd)))/tmp/

docker run -it  \
  -v "$TMP":/mnt/kubeflow/ \
  -p 5000:5000 \
  "${OWNER}/kf-webapp:${KF_PIPELINE_VERSION}" \
  --result=/mnt/kubeflow/results \
  --directory=/mnt/kubeflow/models \
  --model model.pth.tar | tee $(pwd)/log.txt
