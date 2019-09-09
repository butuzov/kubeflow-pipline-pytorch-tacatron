#!/usr/bin/env bash

docker build -t "${OWNER}/kf-webapp:${KF_PIPELINE_VERSION}" \
             -t "${OWNER}/kf-webapp:latest" .

if [[ ! -z $PUSH ]]; then
  docker push "${OWNER}/kf-webapp:${KF_PIPELINE_VERSION}"
  docker push "${OWNER}/kf-webapp:latest"
fi
