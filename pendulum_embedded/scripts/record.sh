#!/bin/bash

stty -F /dev/ttyACM0 921600
filename="log/$(date -d "today" +"%Y%m%d%H%M%S").csv"

if [ $# -gt 0 ]; then
    parameter=$1
    filename="$(date -d "today" +"%Y%m%d%H%M%S")_${parameter}.csv"
fi

echo "cart/angle,pendulum/angle,cart/velocity,pendulum/velocity,motor/voltage" > "${filename}"

cleanup(){
    sed -i '2d' "${filename}"
    exit 1
}

trap cleanup SIGINT

cat /dev/ttyACM0 >> "${filename}"