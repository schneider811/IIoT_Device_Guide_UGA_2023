#!/bin/bash
now=$(date +"%T")
SERVICE="all_together"
ADCDRIVER="/home/pi/IIoT_Device_Guide_UGA_2023/all_together/dist/"
if pgrep -f "$SERVICE" 
then
    echo "$SERVICE is running at $now"
else
    echo "$SERVICE stopped at $now"
    sudo ./$ADCDRIVER/$SERVICE 
fi