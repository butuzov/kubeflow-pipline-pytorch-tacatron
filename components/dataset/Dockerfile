FROM busybox:1.31.0

COPY src/* /

ENTRYPOINT ["./download", "--url=https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2", "--directory=/mnt/kubeflow"]
