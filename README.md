# cvutilities

Miscellaneous helper functions for fetching and processing OpenPose data and camera calibration data

`cvutilities.camera_utilities` contains:

* Functions which load camera calibration data from local JSON files

* Wrappers which reshape inputs, feed them through different combinations of various OpenCV functions, and regularize the outputs

* Functions which support basic visualization of image and object points

`cvutilities.openpose_utilities` contains:

* Functions which load 2D OpenPose output data from Wildflower S3 directories

* Functions which perform very basic 3D pose reconstruction from multi-camera 2D OpenPose data

* Functions which support basic visualization of 2D and 3D pose data

`cvutilities.datetime_utilities` contains functions to convert various datetime formats (native Python, numpy, pandas; timezone-naive or timezone-aware; timezones in dateutil format or pytz format) into known timezone-naive formats in UTC

All of this functionality ultimately needs to be reorganized and grouped with similar functionality from other packages/repos.

## Installation

* Clone this repo to any convenient location on your local drive

* From within your Python development environment (base environment or a virtual environment), run `pip install -e LOCAL_PATH_TO_REPO`

* `pip` will install the `cvutilities` package in your Python development environment (along with all dependencies)

* You can then import the modules like any other Python package (e.g., `import cvutilities.openpose_utilities`)

## Testing

To run tests, change working directory to `/tests` and run one of the following.

`python 3d_reconstruction_example.py` pulls a single frame of 2D pose data from S3 and performs a simple 3D pose reconstruction

`python 3d_pose_tracking_example.py` pulls multiple frames of 2D pose data from a local file and builds a set of 3D pose tracks

These scripts pull camera calibration parameter files and sample 2D pose data from `tests/data`

To run these scripts (and to use any of the functions data from S3), you need to set the `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` environment variables to keys belonging to an account that is a member of the Wildflower `cameras` group.
