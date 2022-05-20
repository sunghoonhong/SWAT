if [ $1 = "cpu" ]; then
    docker run -it -v `pwd`:/username/swat:rw --hostname $HOSTNAME --workdir /username/swat/modular-rl/src/scripts/ swat
else
    docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=$1 -it -v `pwd`:/username/swat:rw --hostname $HOSTNAME --workdir /username/swat/modular-rl/src/scripts/ swat
fi
