# Channelisation ML project

Organisation:

- `modelpy` python code to train models
- `models` where trained models (`hdf5`) files are stored
- `simulate`, where simulation code lives
	- The `simulate.m` and `simmod.m` are the matlab scripts that actually
	  simulate something. The `simmod` is the modified script Prod Kudrolli
	  wrote after some changes
	- The seed for each simulation depends on the simulation number. This make
	  stuff reproducible, so for example simulation 250 will always look the
	  excat same since 250 is used to seed the RNG
	- `run-erosion-sim.sh` scripts need configuration, so for example we
	  configure one script to generate simulations 1-1000, another 2000-3000,
	  etc for as many as we need
	- Submit them to SLURM with `sbatch`
	- Make sure directories for each simulation already exists (`mkdir {1...10}`)
	- The simulated data is stored by default in `data/<simname>/<simnum>/`, so
	  make sure to point the matlab simulation scripts there
	- There are all just constants in the simulation matlab code
- Use the `preprocess-images.py` script to, well, preprocess images
- Check the `simple_cnn_example` notebook on how to train a model using this dataset
- All the figures were plotted with `predict.py`, but the code in there is
  was initially literally copy-pasted from a `python -i` session, so could
  use more logical reworks