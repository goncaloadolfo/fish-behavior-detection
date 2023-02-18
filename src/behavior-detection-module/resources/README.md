# Overview

The [resources](https://github.com/goncaloadolfo/fish-behavior-detection/tree/main/src/behavior-detection-module/resources) folder contains a set of data used to 
train and evaluate the detection models. It contains the following structure:
- `/classification` - ground truth for species and interesting moments (from `v29.m4v`)
- `/datasets` - different feeding related datasets and also a feature vector based for each trajectory
- `/detections` - fish detections for different videos
- definition of some regions (used on region feature extraction, and feeding behavior region filtering)


## Videos used to generate each data file
  
- `/classification` - `v29.m4v`
- `/datasets`:
  - bottom cannon - `feeding-v1-trim.mp4` `feeding-v1-trim2.mp4` `feeding-v2.mp4`
  - surface cannon - `feeding-v3.mp4`
  - bottom go pro - `feeding-v4.mp4`
  - v29 dataset - `v29.m4v`
- `/detections`: 
  - detections v29 sharks mantas - `v29.m4v`
  - detections v37 - `v37.m4v`
  - feeding v4 feeding period gt - `feeding-v4.mp4`
  - feeding v4 normal period gt - `feeding-v4.mp4`
  - v29-fishes - `v29.m4v`


## File Structure

### /classification/species-gt-v29.csv
fish id, species
- fish id: ID of the fish according to fishes GT (`/detections/v29-fishes.json`)
- species: name of the species


### /classification/v29-interesting-moments.csv
fish-id, description, t-initial, t-final
- fish id: ID of the fish according to fishes GT (`/detections/v29-fishes.json`)
- description: description of the moment (in this case is redundant since all of them are "interesting")
- t-initial: frame number of the initial timestamp
- t-final: frame number of the final timestamp


### /datasets/feeding-dataset and /datasets/feeding-surface-dataset and /datasets/gopro-feeding-dataset
- train-samples - frames used to train (resolution 80x50)
- test-samples - frames used to train (resolution 80x50)
- filename structure: frame-{frame-number}-{frame-gt ('normal' or 'feeding')}


### /datasets/v29-dataset1.csv

Features: 
- speed
- x-speed
- y-speed
- accelaration
- x-acceleration
- y-acceleration
- turning angle
- curvature
- centered distance
- bounding box ratio

The features mentioned above are calculated for each instant of the trajectory resulting in a time series for each.
From the time series the following metrics are calculated:
- mean
- median
- std
- min
- max
- 1º percentile
- 3º percentile
- autocorrelation

Number of features: 10 * 8 = 80

Additionally for this dataset 3 regions were considered (top, center, and bottom). The probability of staying on each region was calculated, 
and also the transition between them
- region(1)
- region(2)
- region(3)
- transition(1-2)
- transition(1-3) - redundant
- transition(2-1)
- transition(2-3)
- etc

Important: order of the different samples is the same from `species-gt-v29.csv`


###  /detections/*.txt

frame#,n[,x1,y1,x2,y2,label]

- frame# is the frame number
- n is the number of bounding boxes in that frame
- x1,y1,x2,y2 are the coordinates of a bounding box
- label is the ID of a bounding box


### /detections/v29-fishes.json

```
{
  <fish-id>: {
    "trajectory": [
      [<t>, <x>, <y>],
      ...
    ],
    "bounding-boxes": {
      <t>: (<width>, <height>),
      ...
    }
  },
  ...
}
```

- fish id: ID of the fish according to fishes GT (`/detections/v29-fishes.json`)
- t: frame number

### Region Definition Files

```
{
    <region-id>: {
        "region-id": <region-id>,
        "region-tag": "surface",  // name of the region
        "color": [  // identification color (rgb), visualization purposes
            239,
            120,
            67
        ],
        "rectangles": [  // 2D region limits
            {
                "pt1": [  // top left point (x, y)
                    2,
                    2
                ],
                "pt2": [  // bottom right point (x, y)
                    718,
                    72
                ]
            }
        ]
    },
...
}
```

