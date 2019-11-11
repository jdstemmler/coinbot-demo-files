#!/bin/bash

pausetime=5
for i in {1..50}
do
	echo "Launching sim $i pausing for $pausetime s"
	nohup python hub.py --brain $1 &
	sleep $pausetime
done 	