#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=week-long-cpu
#SBATCH --job-name=erosion2
#SBATCH --output=out.log
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=ts.clarkhpc@isaackhor.com

set -euo pipefail

start_simno=$1
end_simno=$2
simulations_each=$3

cd /home/ikhor/physics-erosion
for i in $(seq $start_simno $simulations_each $end_simno); do
	echo Starting simulation $i
	matlab -nodisplay -nosplash -nodesktop -r "simulate($i,$simulations_each);exit;" > log/sim-$i.log &
	#echo matlab -nodisplay -nosplash -nodesktop -r "simulate($i,$simulations_each);exit;"
done

# Wait for everybody to finish first
wait
