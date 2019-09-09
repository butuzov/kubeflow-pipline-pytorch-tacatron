
from kfp import dsl
import os

from random import choice
from string import ascii_letters as letters


__PIPELINE_NAME__ = "example"
__PIPELINE_DESCRIPTION__ = "demo of kubeflow pipeline"


@dsl.pipeline(name=__PIPELINE_NAME__, description=__PIPELINE_DESCRIPTION__)
def pipeline(
        dataset="https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2",
        directory="/mnt/kf/",
        batch_size=16,
        learning_rate=0.001,
        log_step=1,
        save_step=1,
        epochs=1,
    ):

    OWNER   = os.environ['OWNER']
    VERSION = os.environ['KF_PIPELINE_VERSION']


    volume = dsl.VolumeOp(
        name="volume_creation",
        resource_name="share",
        size="20Gi"
    )

    Dataset_Download = dsl.ContainerOp(
        name="dataset download",
        image=f"{OWNER}/kf-dataset:{VERSION}",
        arguments=[
            f"--url={dataset}",
            f"--directory={directory}"
        ],
        pvolumes={
            f"{directory}" : volume.volume
        },
    )


    Training = dsl.ContainerOp(
        name="training model",
        image=f"{OWNER}/kf-training:{VERSION}",
        arguments=[
            f"--dir_data={directory}/dataset",
            f"--dir_checkpoints={directory}/models",
            f"--batch_size={batch_size}",
            f"--learning_rate={learning_rate}",
            f"--log_step={log_step}",
            f"--save_step={save_step}",
            f"--epochs={epochs}",
        ],
        pvolumes={
            f"{directory}" : volume.volume
        },
    )

    Training.after(Dataset_Download)

    Seving = dsl.ContainerOp(
        name="serving",
        image=f"{OWNER}/kf-webapp:{VERSION}",
        arguments=[
            f"--result={directory}/results",
            f"--directory={directory}/models",
            f"--model=model.pth.tar",
        ],
        pvolumes={
            f"{directory}" : volume.volume
        },
    )

    Seving.after(Training)



if __name__ == "__main__":
    import kfp.compiler as compiler

    compiler.Compiler().compile(pipeline,
        f"pipeline-{os.environ['KF_PIPELINE_VERSION']}.tar.gz")
