#!/usr/bin/env bash

docker build -t "${OWNER}/kf-ml:${KF_PIPELINE_VERSION}" \
             -t "${OWNER}/kf-ml:latest"  .

if [[ ! -z $PUSH ]]; then
  docker push "${OWNER}/kf-ml:${KF_PIPELINE_VERSION}"
  docker push "${OWNER}/kf-ml:latest"
fi
