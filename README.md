# WormReIDTracker

### 

1. Clone the repository:
   `git clone https://github.com/1490560350/WormReIDTracker.git`  # Clone the WormReIDTracker repository

2. Setup environments:

   `cd WormReIDTracker`  # Navigate to the WormReIDTracker directory
   
   `conda create -n wormreidtracker`  # Create a Conda environment named wormreidtracker
   
   `pip install -e .`  # Install WormReIDTracker in editable mode

4. Running the tracker after cloning the WormReIDTracker repository:
   
   `python train.py`  # Run the training script; you can configure the dataset and whether to load pre-trained weights in the train.py file.     Note: It is necessary to modify the actual path of the dataset in the `WormReIDTracker/ultralytics/cfg/datasets/data.yaml` file.
   
   `python track_BotSort.py`  # Run the tracking script

WormReIDTracker is based on the YOLO model. For more details, please visit: https://github.com/ultralytics/ultralytics. 


# FastReID
### Steps
`cd WormReIDTracker`  # Navigate to the WormReIDTracker directory
`conda create -n fastreid python=3.7`
`conda activate fastreid`
`conda install pytorch==1.6.0 torchvision tensorboard -c pytorch`
`pip install -r docs/requirements.txt`
For more details, please visit: https://github.com/JDAI-CV/fast-reid
# TrackEval
### Steps

python scripts/run_mot_challenge.py --BENCHMARK worm --TRACKERS_TO_EVAL BotSort --METRICS HOTA CLEAR Identity VACE --USE_PARALLEL False --NUM_PARALLEL_CORES 8
For the TrackEval code and usage instructions, please visit: https://github.com/JonathonLuiten/TrackEval.


