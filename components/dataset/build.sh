#!/usr/bin/env bash

docker build -t "${OWNER}/kf-dataset:${KF_PIPELINE_VERSION}" .

if [[ ! -z $PUSH ]]; then
  docker push "${OWNER}/kf-dataset:${KF_PIPELINE_VERSION}"
  docker push "${OWNER}/kf-dataset:latest"
fi
