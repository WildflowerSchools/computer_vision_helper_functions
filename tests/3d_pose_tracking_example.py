import cvutilities.openpose_utilities
import cvutilities.camera_utilities
import numpy as np
import pickle
import pandas as pd
import time
import os

# Define input and output files
data_directory = './data'
input_dataframe_filename = 'example_2d_poses_dataframe.pickle.xz'
output_dataframe_filename = 'example_3d_pose_tracks_dataframe.pickle.xz'

input_dataframe_path = os.path.join(
    data_directory,
    input_dataframe_filename)

output_dataframe_path = os.path.join(
    data_directory,
    output_dataframe_filename)

# Define model parameters

room_size = np.array([7.0, 8.0, 3.0])

pose_initialization_model = cvutilities.openpose_utilities.PoseInitializationModel(
    initial_keypoint_position_means = np.tile(room_size/2, (18, 1)),
    initial_keypoint_velocity_means = np.zeros((18,3)),
    initial_keypoint_position_error = np.amax(room_size)/2.0,
    initial_keypoint_velocity_error = 2.0)

keypoint_model = cvutilities.openpose_utilities.KeypointModel(
    position_observation_error = 1.0,
    reference_delta_t = 0.1,
    reference_position_transition_error = 0.1,
    reference_velocity_transition_error = 0.1)

pose_tracking_model = cvutilities.openpose_utilities.PoseTrackingModel(
    cost_threshold = 1.0,
    num_missed_observations_threshold = 10)

# Ingest and process the data

input_dataframe = pd.read_pickle(input_dataframe_path)

num_rows = input_dataframe.shape[0]

camera_names = input_dataframe.columns.levels[0].tolist()

camera_calibration_parameters = cvutilities.camera_utilities.fetch_camera_calibration_data_from_local_drive_multiple_cameras(
    camera_names = camera_names,
    camera_calibration_data_directory = './data')

initial_dataframe_row = input_dataframe.iloc[0]

initial_poses_2d = cvutilities.openpose_utilities.Poses2D.from_dataframe_row(
    dataframe_row = initial_dataframe_row,
    camera_names = camera_names)

initial_poses_3d = cvutilities.openpose_utilities.Poses3D.from_poses_2d(
    poses_2d = initial_poses_2d,
    cameras = camera_calibration_parameters)

pose_tracks = cvutilities.openpose_utilities.Pose3DTracks.initialize(
    pose_initialization_model = pose_initialization_model,
    keypoint_model = keypoint_model,
    pose_tracking_model = pose_tracking_model,
    pose_3d_observations = initial_poses_3d)

start_time = time.time()
for row_index in range(1, num_rows):
    dataframe_row = input_dataframe.iloc[row_index]
    poses_2d = cvutilities.openpose_utilities.Poses2D.from_dataframe_row(
        dataframe_row,
        camera_names)
    poses_3d = cvutilities.openpose_utilities.Poses3D.from_poses_2d(
        poses_2d,
        camera_calibration_parameters)
    pose_tracks.update(poses_3d)
end_time = time.time()
elapsed_time = end_time - start_time

print('{} tracks produced from {} frames in {:.1f} seconds: {:.1f} milliseconds per frame'.format(
    pose_tracks.num_inactive_tracks() + pose_tracks.num_active_tracks(),
    num_rows,
    elapsed_time,
    1000*elapsed_time/num_rows))

output_dataframe = pose_tracks.dataframe()

output_dataframe.to_pickle(output_dataframe_path)

print('Output saved in {}'.format(output_dataframe_filename))
