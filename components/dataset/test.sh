#!/usr/bin/env bash


TMP=$(dirname $(dirname $(pwd)))/tmp/


docker run -it  \
  -v "$TMP":/mnt/kubeflow/ \
  "${OWNER}/dataset:${KF_PIPELINE_VERSION}" \
  --url=https://sourceware.org/ftp/libffi/libffi-3.2.1.tar.gz
