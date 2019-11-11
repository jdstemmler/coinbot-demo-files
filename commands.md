# Bonsai Commands

## create the brain
`bonsai create cartpole-demo-1`

## upload files to the platform
`bonsai push --brain cartpole-demo-1`

## initialize the brain to accept training
`bonsai train start --brain cartpole-demo-1`

## run a single instance of the simulator training, with rendering
`python .\hub.py --brain cartpole-demo-1 --render`

## run a batch of training runs (30) headless
`.\scripts\launch_multiple_sims.bat 30 cartpole-demo-1`

## stop training the brain
`bonsai train stop --brain cartpole-demo-1`

## make a prediction on the brain
`python .\hub.py --brain cartpole-demo-1 --predict=latest --render`