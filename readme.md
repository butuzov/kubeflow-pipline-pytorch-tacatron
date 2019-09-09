# Example of [kubeflow](http://kubeflow.org) pipeline builder

## Example Project

Demo project based uppon [Tacotron-pytorch](https://github.com/soobinseo/Tacotron-pytorch) by @soobinseo

## Modifications of original project.

* ML code slitly midified, in order to make it working without issues and be more copatible with cuda (if present)
* Few values of the `hyperparametes` moved out into pipeline parameters
* Speech syntez moved out to webapp, but failing by itself (see [Issue #6](https://github.com/soobinseo/Tacotron-pytorch/issues/6))
* Sound transformation changed in regards of changed API of `librosa`

## Build

```bash
# build with all possible options
./build.sh --owner="butuzov"
           --version=1.1
           --push
           --withtests
```

* `--owner` options for docker hub user
* `--version` to set a version of pipeline and docker images
* `--push`  push to docker registry
* `--withtests` run simple tests on hosts enviropment with maped volumes and params. test for `webapp` should be terminated manually.


