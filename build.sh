#!/usr/bin/env bash


# variables

# export KF_PIPELINE_VERSION=1.1
# export OWNER="butuzov"
# export __BASE__=$(pwd)



while [ $# -gt 0 ]; do
  # printf "[%s]\n" $1
  case "$1" in
    --owner=*)
      owner="${1#*=}"
      ;;
    --push=*)
      export PUSH=1
      ;;
    --version=*)
      version="${1#*=}"
      ;;
    --withtests*)
      export TESTS=1
      ;;
    *)
      printf "Error: Invalid argument.\n"
      exit 1
  esac
  shift
done

export OWNER=${owner:-butuzov}
printf "Owner     [%s]\n" $OWNER

export KF_PIPELINE_VERSION=${version:-1.0}
printf "Version   [%s]\n" $KF_PIPELINE_VERSION

if [[ -z $TESTS ]]; then
  printf "RunTests? [%s]\n" No
else
  printf "RunTests? [%s]\n" "Yes"
fi

# virtual env
python3 -m venv .venv
. .venv/bin/activate

pip install --upgrade pip
pip install --upgrade pip kfp 1>/dev/null

export __BASE__=$(pwd)
export COMPONENTS=$__BASE__/components


for component in "${COMPONENTS}"/*/; do
  if [[ -f "$component/build.sh" ]]; then
    cd $component && ./build.sh
  fi

  if [[ ! -z $TESTS ]] && [[ ! -z $TESTS ]]; then
    echo $component
  fi
  # if [[ -f "$component/test.sh" ]]; then
  #   cd $component && ./test.sh
  # fi
done

# python3 "${__BASE__}"/pipeline.py

