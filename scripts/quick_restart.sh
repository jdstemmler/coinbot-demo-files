# arg
source activate bonsai-35
bonsai train stop
bonsai push
bonsai train start
python hub.py --brain newTKHVAC --log-iterations
