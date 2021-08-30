# Behavior Detection Module (BDM)

## What is it?

The Behavior Detection Module (BDM) is a library that was developed to detect fish behaviors: abnormal behaviors, interesting behaviors, feeding behaviors, and swallowing air behavior. The focus species were manta rays and sharks, so the algorithms are based on the way these species behave. It was implemented to be included in the Lisbon Oceanarium domain but the idea was to be generic enough to be integrated into other systems and species, although that the code examples use data from this oceanarium and for these species.


## Main Packages - Source Structure

- **trajectory_reader**: read trajectories from files; union trajectories from different source files; visualization base functions
- **trajectory_features**: extract features from trajectories; build datasets; data exploration
- **pre_processing**: fill gaps of trajectories; smooth trajectories; segmentation; feature vector pre-processing
- **anomaly_detection**: detect outlier trajectories (clustering, switching vector model)
- **feeding**: detect feeding periods (aggregation-based, motion/optical flow-based, CNN)
- **interesting_episodes_detection**: classify trajectories that might be interesting to analyze
- **swallow air**: detect swallow air behaviors (region-based, patterns based)
- **labeling**: utility code to help on the labeling task (define regions, label behavior episodes, specify species information)
- **resources**: all the data needed for the development/testing process


## Documentation
Work in progress :pensive:
It will be updated soon.


## Motivational videos

https://user-images.githubusercontent.com/56698352/131411540-f3d830fb-6f83-4939-adf6-738df0e569c0.mp4


https://user-images.githubusercontent.com/56698352/131412825-21aad92a-8ae5-48a9-8097-37fc4b1ac02a.mp4


https://user-images.githubusercontent.com/56698352/131411837-d642ee42-c716-4d8b-859e-6de778a2cf71.mp4


https://user-images.githubusercontent.com/56698352/131411590-8929c7d9-7ec2-4f6a-9072-3a54766e30d5.mp4


https://user-images.githubusercontent.com/56698352/131411571-36455469-1da8-46b0-9b2f-a47d731e9423.mp4
