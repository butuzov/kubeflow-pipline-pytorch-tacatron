#!/usr/bin/env bash

docker build -t "${OWNER}/kf-training:${KF_PIPELINE_VERSION}" \
             -t "${OWNER}/kf-training:latest" .

if [[ ! -z $PUSH ]]; then
  docker push "${OWNER}/kf-training:${KF_PIPELINE_VERSION}"
  docker push "${OWNER}/kf-training:latest"
fi
