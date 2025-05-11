# WormReIDTracker
 <p align="center">
  <img src="WormYOLO/examples/segmenting.gif" alt="Segmentation results" width="45%" />
</p>

### 

1. Clone the repository:
   `git clone https://github.com/1490560350/WormReIDTracker.git`  # Clone the WormReIDTracker repository

2. Setup environments:

   `cd WormReIDTracker`  # Navigate to the WormReIDTracker directory
   
   `conda create -n wormreidtracker`  # Create a Conda environment named wormreidtracker
   
   `pip install -e .`  # Install WormReIDTracker in editable mode

4. Running the tracker after cloning the WormReIDTracker repository:
   
   `python train.py`  # Run the training script; you can configure the dataset and whether to load pre-trained weights in the train.py file.     Note: It is necessary to modify the actual path of the dataset in the `WormReIDTracker/ultralytics/cfg/datasets/data.yaml` file.
   
   `python track_singleworm.py/track_multworms.py`  # Run the tracking script

WormReIDTracker is based on the YOLO model. For more details, please visit: https://github.com/ultralytics/ultralytics. For the TrackEval code and usage instructions, please visit: https://github.com/JonathonLuiten/TrackEval.

# FastReID
### Steps
1. Place the preprocessed image set into the subfolder named "samples" within the "feature points extraction" folder.

2. Double click to open ‘feature points extraction.sln’.

3. Run the program and obtain the coordinates of the pharynx, inflection point, peak point, and skeletal point. These results are saved in the files below.

# TrackEval
### Steps
1. Please place the 5 ".csv" files obtained from the feature point detection algorithm into the "Feature_point" subfolder within the "Tracker" folder.

2. Open the program using Matlab R2021a, run ‘Tracker_ Feature.p’, and obtain the experimental results.



