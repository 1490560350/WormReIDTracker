<h1 align="center">🎯 WormReIDTracker: A C. elegans Tracking Framework Based on Cross-Scale Feature Fusion and Identity Feature Regression</h1>

![motivation](assets/framework.png)

<p align="justify">
This repository hosts the official implementation of a novel framework dedicated to robust nematode tracking. It features two key modules, MFSM and SOEN, designed to fuse multi-scale features for improved bounding box precision. By leveraging an identity feature regression module, the framework effectively reduces identity switching for more stable tracking.  Included are training/evaluation <a href="https://github.com/1490560350/WormReIDTracker">codes</a>, and <a href="https://zenodo.org/records/19871295">results</a>.
</p>


## 📂 Data Preparation

## ⚙️ Installation
- ###  WormReIDTracker
1. Clone the repository:
   `git clone https://github.com/1490560350/WormReIDTracker.git`  # Clone the WormReIDTracker repository.

2. Setup environments:

   `cd WormReIDTracker`  # Navigate to the WormReIDTracker directory.
   
   `conda create -n wormreidtracker`  # Create a Conda environment named wormreidtracker.
   
   `pip install -e .`  # Install WormReIDTracker in editable mode.

4. Running the tracker after cloning the WormReIDTracker repository:
   
   `python train.py`  # Run the training script; you can configure the dataset and whether to load pre-trained weights in the train.py file.     Note: It is necessary to modify the actual path of the dataset in the `WormReIDTracker/ultralytics/cfg/datasets/data.yaml` file.
   
   `python track_BotSort.py`  # Run the tracking script.

WormReIDTracker is based on the YOLO model. For more details, please visit: https://github.com/ultralytics/ultralytics. 


- ###  FastReID
   `cd WormReIDTracker/FastReID`  # Navigate to the FastReID directory.

   `conda create -n fastreid python=3.7` # Create a Conda environment named fastreid.

   `conda activate fastreid` # Activate the Conda environment named fastreid.

  `pip install -r docs/requirements.txt` # Install all dependencies.

  `python tools/train_net.py --config-file ./configs/Worm/mgn_R50-ibn.yml MODEL.WEIGHTS ./weights/market_mgn_R50-ibn.pth MODEL.DEVICE "cuda:0"`

For the FastReID code and usage instructions, please visit please visit: https://github.com/JDAI-CV/fast-reid.
- ###  TrackEval
  `python scripts/run_mot_challenge.py --BENCHMARK worm --TRACKERS_TO_EVAL BotSort --METRICS HOTA CLEAR Identity VACE --USE_PARALLEL False --NUM_PARALLEL_CORES 8` #

For the TrackEval code and usage instructions, please visit: https://github.com/JonathonLuiten/TrackEval.


