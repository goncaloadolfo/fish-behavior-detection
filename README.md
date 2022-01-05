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
- [Master thesis](https://github.com/goncaloadolfo/fish-behavior-detection/blob/main/97090-goncalo-adolfo-dissertacao.pdf)
- [Extended Abstract](https://github.com/goncaloadolfo/fish-behavior-detection/blob/main/97090-goncalo-adolfo-resumo.pdf)
- [Feeding Paper](https://github.com/goncaloadolfo/fish-behavior-detection/blob/main/goncalo-feeding-paper.pdf)
- code snippets: work in progress :pensive:, will be updated soon
