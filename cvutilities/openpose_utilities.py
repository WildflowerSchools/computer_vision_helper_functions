import cvutilities.camera_utilities
import cvutilities.datetime_utilities
import smc_kalman
import boto3
import networkx as nx # We use NetworkX graph structures to hold the 3D pose data
import numpy as np
import pandas as pd
import scipy.optimize
import matplotlib.pyplot as plt
import json

# For now, the Wildflower-specific S3 functionality is intermingled with the more
# general S3 functionality. We should probably separate these at some point. For
# the S3 functions below to work, the environment must include AWS_ACCESS_KEY_ID and
# AWS_SECRET_ACCESS_KEY variables and that access key must have read permissions
# for the relevant buckets. You can set these environment variables manually or
# by using the AWS CLI.
classroom_data_wildflower_s3_bucket_name = 'wf-classroom-data'
pose_2d_data_wildflower_s3_directory_name = '2D-pose'

# Generate the Wildflower S3 object name for a 2D pose file from a classroom
# name, a camera name, and a Python datetime object
def generate_pose_2d_wildflower_s3_object_name(
    classroom_name,
    camera_name,
    datetime):
    date_string, time_string = generate_wildflower_s3_datetime_strings(datetime)
    pose_2d_wildflower_s3_object_name = 'camera-{}/{}/{}/{}/still_{}-{}_keypoints.json'.format(
        classroom_name,
        pose_2d_data_wildflower_s3_directory_name,
        date_string,
        camera_name,
        date_string,
        time_string)
    return pose_2d_wildflower_s3_object_name

# Generate date and time strings (as they appear in our Wildflower S3 object
# names) from a Python datetime object
def generate_wildflower_s3_datetime_strings(
    datetime):
    datetime_native_utc_naive = cvutilities.datetime_utilities.convert_to_native_utc_naive(datetime)
    date_string = datetime_native_utc_naive.strftime('%Y-%m-%d')
    time_string = datetime_native_utc_naive.strftime('%H-%M-%S')
    return date_string, time_string

# For now, the OpenPose-specific functionality is intermingled with the more
# general pose analysis functionality. We should probably separate these at some
# point. The parameters below correspond to the OpenPose output that we've been
# generating but the newest OpenPose version changes these (more body parts,
# 'pose_keypoints_2d' instead of 'pose_keypoints', etc.)
openpose_people_list_name = 'people'
openpose_keypoint_vector_name = 'pose_keypoints'
num_body_parts = 18
body_part_long_names = [
    "Nose",
    "Neck",
    "RShoulder",
    "RElbow",
    "RWrist",
    "LShoulder",
    "LElbow",
    "LWrist",
    "RHip",
    "RKnee",
    "RAnkle",
    "LHip",
    "LKnee",
    "LAnkle",
    "REye",
    "LEye",
    "REar",
    "LEar"]
body_part_connectors = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    [1, 5],
    [5, 6],
    [6, 7],
    [1, 8],
    [8, 9],
    [9, 10],
    [1, 11],
    [11, 12],
    [12, 13],
    [0, 14],
    [14, 16],
    [0, 15],
    [15, 17]]

# Define some subsets of indices that we will use when calculating anchor points
neck_index = 1
shoulder_indices = [2, 5]
head_and_torso_indices = [0, 1, 2, 5, 8, 11, 14 , 15, 16, 17]

# Specify time unit when unitless time value is needed (e.g., in Kalman filter)
time_unit = np.timedelta64(1, 's')

# Class to represent a single 2D pose
class Pose2D:
    def __init__(
        self,
        keypoints,
        confidence_scores,
        valid_keypoints,
        tag = None,
        timestamp = None):
        keypoints = np.asarray(keypoints)
        confidence_scores = np.asarray(confidence_scores)
        valid_keypoints = np.asarray(valid_keypoints, dtype = np.bool_)
        if keypoints.size != num_body_parts*2:
            raise ValueError('Keypoints array does not appear to be of size {}*2'.format(num_body_parts))
        if confidence_scores.size != num_body_parts:
            raise ValueError('Confidence scores vector does not appear to be of size {}'.format(num_body_parts))
        if valid_keypoints.size != num_body_parts:
            raise ValueError('Valid keypoints vector does not appear to be of size {}'.format(num_body_parts))
        keypoints = keypoints.reshape((num_body_parts, 2))
        confidence_scores = confidence_scores.reshape(num_body_parts)
        valid_keypoints = valid_keypoints.reshape(num_body_parts)
        self.keypoints = keypoints
        self.confidence_scores = confidence_scores
        self.valid_keypoints = valid_keypoints
        self.tag = tag
        self.timestamp = timestamp

    # Pull the pose data from a dictionary with the same structure as the
    # correponding OpenPose output JSON string for a single person
    @classmethod
    def from_openpose_person_json_data(
        cls,
        json_data,
        timestamp = None):
        keypoint_vector = np.asarray(json_data[openpose_keypoint_vector_name])
        if keypoint_vector.size != num_body_parts*3:
            raise ValueError('OpenPose keypoint vector does not appear to be of size {}*3'.format(num_body_parts))
        keypoint_array = keypoint_vector.reshape((num_body_parts, 3))
        keypoints = keypoint_array[:, :2]
        confidence_scores = keypoint_array[:, 2]
        valid_keypoints = np.not_equal(confidence_scores, 0.0)
        return cls(
            keypoints,
            confidence_scores,
            valid_keypoints,
            timestamp = timestamp)

    # Pull the pose data from an OpenPose output JSON string
    @classmethod
    def from_openpose_person_json_string(
        cls,
        json_string,
        timestamp = None):
        json_data = json.loads(json_string)
        return cls.from_openpose_person_json_data(
            json_data,
            timestamp)

    # Set tag (we provide a function for this to stay consistent with the other
    # classes and with the princple that users of these classes should never
    # have to call the base constructor)
    def set_tag(
        self,
        tag):
        self.tag = tag

    # Draw the pose onto a chart with the coordinate system of the origin image.
    # We separate this from the plotting function below because we might want to
    # draw several poses or other elements before formatting and showing the
    # chart
    def draw(self):
        all_points = self.keypoints
        valid_keypoints = self.valid_keypoints
        plottable_points = all_points[valid_keypoints]
        cvutilities.camera_utilities.draw_2d_image_points(plottable_points)
        for body_part_connector in body_part_connectors:
            body_part_from_index = body_part_connector[0]
            body_part_to_index = body_part_connector[1]
            if valid_keypoints[body_part_from_index] and valid_keypoints[body_part_to_index]:
                plt.plot(
                    [all_points[body_part_from_index,0],all_points[body_part_to_index, 0]],
                    [all_points[body_part_from_index,1],all_points[body_part_to_index, 1]],
                    'k-',
                    alpha = 0.2)

    # Plot a pose onto a chart with the coordinate system of the origin image.
    # Calls the drawing function above, adds formating, and shows the plot
    def plot(
        self,
        pose_tag = None,
        image_size=[1296, 972]):
        self.draw()
        cvutilities.camera_utilities.format_2d_image_plot(image_size)
        plt.show()

# Class to represent a collection of 2D poses, either a set of poses (as from a
# single camera) or a set of set of poses (as from multiple cameras at a single
# moment in time). Internal structure is a list of lists of Pose2D objects. When
# working with one-dimensional collections, input methods add the additional
# dimension and output methods remove the addtional dimension as necessary
class Poses2D:
    def __init__(
        self,
        pose_2d_list_list,
        source_images = None):
        self.pose_2d_list_list = pose_2d_list_list
        self.source_images = source_images

    # Pull the pose data for a single camera from a dictionary with the same
    # structure as the correponding OpenPose output JSON file for a single image
    @classmethod
    def from_openpose_output_json_data(
        cls,
        json_data,
        source_images = None,
        timestamp=None):
        people_json_data = json_data[openpose_people_list_name]
        pose_2d_list_list = [[Pose2D.from_openpose_person_json_data(person_json_data, timestamp) for person_json_data in people_json_data]]
        return cls(
            pose_2d_list_list,
            source_images)

    # Pull the pose data for a single camera from a string containing the
    # contents of an OpenPose output JSON file for a single image
    @classmethod
    def from_openpose_output_json_string(
        cls,
        json_string,
        source_images = None,
        timestamp = None):
        json_data = json.loads(json_string)
        return cls.from_openpose_output_json_data(
            json_data,
            source_images,
            timestamp)

    # Pull the pose data for a single camera from a local OpenPose output JSON
    # file
    @classmethod
    def from_openpose_output_json_file(
        cls,
        json_file_path,
        source_images = None,
        timestamp = None):
        with open(json_file_path) as json_file:
            json_data = json.load(json_file)
        return cls.from_openpose_output_json_data(
            json_data,
            source_images,
            timestamp)

    # Pull the pose data for a single camera from an OpenPose output JSON file
    # stored on S3 and specified by S3 bucket and object names
    @classmethod
    def from_openpose_output_s3_object(
        cls,
        s3_bucket_name,
        s3_object_name,
        source_images = None,
        timestamp = None):
        s3_object = boto3.resource('s3').Object(s3_bucket_name, s3_object_name)
        s3_object_content = s3_object.get()['Body'].read().decode('utf-8')
        json_data = json.loads(s3_object_content)
        return cls.from_openpose_output_json_data(
            json_data,
            source_images,
            timestamp)

    # Pull the pose data for a single camera from an OpenPose output JSON file
    # stored on S3 and specified by Wildflower classroom name, camera name, and
    # a Python datetime object
    @classmethod
    def from_openpose_output_wildflower_s3(
        cls,
        classroom_name,
        camera_name,
        datetime,
        fetch_source_image = False):
        s3_bucket_name = classroom_data_wildflower_s3_bucket_name
        s3_object_name = generate_pose_2d_wildflower_s3_object_name(
            classroom_name,
            camera_name,
            datetime)
        if fetch_source_image:
            source_images = [cvutilities.camera_utilities.fetch_image_from_wildflower_s3(
                classroom_name,
                camera_name,
                datetime)]
        else:
            source_images = None
        return cls.from_openpose_output_s3_object(
            s3_bucket_name,
            s3_object_name,
            source_images,
            timestamp = datetime)

    # Pull the pose data for multiple cameras at a single moment in time from a
    # set of OpenPose output JSON files stored on S3 and specified by Wildflower
    # classroom name, a list of camera names, and a Python datetime object
    @classmethod
    def from_openpose_timestep_wildflower_s3(
        cls,
        classroom_name,
        camera_names,
        datetime,
        fetch_source_images = False):
        s3_bucket_name = classroom_data_wildflower_s3_bucket_name
        pose_2d_list_list = []
        for camera_name in camera_names:
            s3_object_name = generate_pose_2d_wildflower_s3_object_name(
                classroom_name,
                camera_name,
                datetime)
            camera = Poses2D.from_openpose_output_s3_object(
                s3_bucket_name,
                s3_object_name,
                timestamp = datetime)
            pose_2d_list_list.append(camera.pose_2d_list_list[0])
        if fetch_source_images:
            source_images = []
            for camera_name in camera_names:
                source_image = cvutilities.camera_utilities.fetch_image_from_wildflower_s3(
                    classroom_name,
                    camera_name,
                    datetime)
                source_images.append(source_image)
        else:
            source_images = None
        return cls(
            pose_2d_list_list,
            source_images)

    # Set pose tags
    def set_tags(
        self,
        tag_lists):
        num_tag_lists = len(tag_lists)
        if num_tag_lists != self.num_cameras():
            raise ValueError('Number of tag lists does not match number of cameras')
        for tag_list_index in range(num_tag_lists):
            num_tags = len(tag_lists[tag_list_index])
            if num_tags != self.num_poses()[tag_list_index]:
                raise ValueError('Length of tag list does not match number of poses')
            for tag_index in range(num_tags):
                self.pose_2d_list_list[tag_list_index][tag_index].tag = tag_lists[tag_list_index][tag_index]

    # Return number of cameras
    def num_cameras(self):
        return len(self.pose_2d_list_list)

    # Return number of poses for each camera
    def num_poses(self):
        return [len(camera) for camera in self.pose_2d_list_list]

    # Return keypoints
    def keypoints(self):
        return [[pose.keypoints for pose in camera] for camera in self.pose_2d_list_list]

    # Return confidence_scores
    def confidence_scores(self):
        return [[pose.confidence_scores for pose in camera] for camera in self.pose_2d_list_list]

    # Return valid keypoints
    def valid_keypoints(self):
        return [[pose.valid_keypoints for pose in camera] for camera in self.pose_2d_list_list]

    # Return pose tags
    def tags(self):
        return [[pose.tag for pose in camera] for camera in self.pose_2d_list_list]

    # Return pose timestamps
    def timestamps(self):
        return [[pose.timestamp for pose in camera] for camera in self.pose_2d_list_list]

    # Plot the poses onto a set of charts, one for each source camera view.
    def plot(
        self,
        image_size = [1296, 972]):
        num_cameras = self.num_cameras()
        for camera_index in range(num_cameras):
            if self.source_images is not None:
                source_image = self.source_images[camera_index]
                cvutilities.camera_utilities.draw_background_image(source_image)
                current_image_size = np.array([
                    source_image.shape[1],
                    source_image.shape[0]])
            else:
                current_image_size = image_size
            num_poses = self.num_poses()[camera_index]
            for pose_index in range(num_poses):
                pose = self.pose_2d_list_list[camera_index][pose_index]
                pose.draw()
                if pose.tag is not None:
                    tag = pose.tag
                else:
                    tag = pose_index
                all_points = pose.keypoints
                valid_keypoints = pose.valid_keypoints
                plottable_points = all_points[valid_keypoints]
                centroid = np.mean(plottable_points, 0)
                if centroid[0] > 0 and centroid[0] < current_image_size[0] and centroid[1] > 0 and centroid[1] < current_image_size[1]:
                    plt.text(centroid[0], centroid[1], tag)
            cvutilities.camera_utilities.format_2d_image_plot(current_image_size)
            plt.show()

# Class to represent a single 3D pose
class Pose3D:
    def __init__(
        self,
        keypoints,
        valid_keypoints,
        projection_error = None,
        tag = None,
        timestamp = None,
        keypoint_std_devs = None):
        keypoints = np.asarray(keypoints)
        if keypoints.size != num_body_parts*3:
            raise ValueError('Keypoints array does not appear to be of size {}*3'.format(num_body_parts))
        keypoints = keypoints.reshape((num_body_parts, 3))
        valid_keypoints = np.asarray(valid_keypoints, dtype = np.bool_)
        if valid_keypoints.size != num_body_parts:
            raise ValueError('Valid keypoints vector does not appear to be of size {}'.format(num_body_parts))
        valid_keypoints = valid_keypoints.reshape(num_body_parts)
        if projection_error is not None:
            projection_error = np.asarray(projection_error)
            if projection_error.size != 1:
                raise ValueError('Projection error does not appear to be a scalar')
            projection_error = np.asscalar(projection_error)
        if timestamp is not None:
            timestamp = np.asarray(timestamp)
            if timestamp.size != 1:
                raise ValueError('Timestamp does not appear to be a scalar')
        if keypoint_std_devs is not None:
            keypoint_std_devs = np.asarray(keypoint_std_devs)
            if keypoint_std_devs.size != num_body_parts*3:
                raise ValueError('Keypoint standard deviations array does not appear to be of size {}*3'.format(num_body_parts))
            keypoint_std_devs = keypoint_std_devs.reshape((num_body_parts, 3))
        self.keypoints = keypoints
        self.valid_keypoints = valid_keypoints
        self.projection_error = projection_error
        self.tag = tag
        self.timestamp = timestamp
        self.keypoint_std_devs = keypoint_std_devs

    # Build a 3D pose object from a simple array of keypoints. Assume that
    # non-valid keypoints are indicated by numpy NAN values
    @classmethod
    def from_keypoints(
        cls,
        keypoints,
        tag = None,
        timestamp = None):
        keypoints = np.asarray(keypoints)
        if keypoints.size > 0:
            valid_keypoints = np.logical_not(np.isnan(keypoints[:, 0]))
        else:
            if keypoints.size != num_body_parts*3:
                raise ValueError('Keypoints array does not appear to be of size {}*3'.format(num_body_parts))
            keypoints = keypoints.reshape((num_body_parts, 3))
            valid_keypoints = np.array([])
        return cls(
            keypoints,
            valid_keypoints,
            tag = tag,
            timestamp = timestamp)

    # Calculate a 3D pose by triangulating between two 2D poses from two
    # different cameras
    @classmethod
    def from_poses_2d(
        cls,
        pose_2d_a,
        pose_2d_b,
        rotation_vector_a,
        translation_vector_a,
        rotation_vector_b,
        translation_vector_b,
        camera_matrix,
        distortion_coefficients = np.array([])):
        rotation_vector_a = np.asarray(rotation_vector_a).reshape(3)
        translation_vector_a = np.asarray(translation_vector_a).reshape(3)
        rotation_vector_b = np.asarray(rotation_vector_b).reshape(3)
        translation_vector_b = np.asarray(translation_vector_b).reshape(3)
        camera_matrix  = np.asarray(camera_matrix).reshape((3,3))
        distortion_coefficients = np.asarray(distortion_coefficients)
        image_points_a, image_points_b, common_keypoint_positions_mask = extract_common_keypoints(
            pose_2d_a,
            pose_2d_b)
        image_points_a_distortion_removed = cvutilities.camera_utilities.undistort_points(
            image_points_a,
            camera_matrix,
            distortion_coefficients)
        image_points_b_distortion_removed = cvutilities.camera_utilities.undistort_points(
            image_points_b,
            camera_matrix,
            distortion_coefficients)
        object_points = cvutilities.camera_utilities.reconstruct_object_points_from_camera_poses(
            image_points_a_distortion_removed,
            image_points_b_distortion_removed,
            camera_matrix,
            rotation_vector_a,
            translation_vector_a,
            rotation_vector_b,
            translation_vector_b)
        image_points_a_reconstructed = cvutilities.camera_utilities.project_points(
            object_points,
            rotation_vector_a,
            translation_vector_a,
            camera_matrix,
            distortion_coefficients)
        image_points_b_reconstructed = cvutilities.camera_utilities.project_points(
            object_points,
            rotation_vector_b,
            translation_vector_b,
            camera_matrix,
            distortion_coefficients)
        projection_error_a = rms_projection_error(
            image_points_a,
            image_points_a_reconstructed)
        projection_error_b = rms_projection_error(
            image_points_b,
            image_points_b_reconstructed)
        object_points = object_points.reshape((-1, 3))
        keypoints = restore_all_keypoints(
            object_points,
            common_keypoint_positions_mask)
        if np.isnan(projection_error_a) or np.isnan(projection_error_b):
            projection_error = np.nan
        else:
            projection_error = max(
                projection_error_a,
                projection_error_b)
        if pose_2d_a.tag is None and pose_2d_b.tag is None:
            tag = None
        elif pose_2d_a.tag == pose_2d_b.tag:
            tag = pose_2d_a.tag
        else:
            tag = '{}/{}'.format(pose_2d_a.tag, pose_2d_b.tag)
        timestamp = None
        if pose_2d_a.timestamp is not None and pose_2d_b.timestamp is not None:
            if pose_2d_a.timestamp != pose_2d_b.timestamp:
                raise ValueError('2D poses have different timstamps')
            timestamp = pose_2d_a.timestamp
        return cls(
            keypoints,
            common_keypoint_positions_mask,
            projection_error,
            tag,
            timestamp)

    # Calculate an anchor point that we can use as the position of the person as
    # a whole (e.g., for pose tracking). Ideally, this would be a point that
    # moves only when the whole person moves (e.g., not just when they move
    # their limbs)
    def anchor_point(self):
        if self.valid_keypoints[neck_index]:
            return self.keypoints[neck_index]
        if np.all(self.valid_keypoints[shoulder_indices]):
            return np.mean(self.keypoints[shoulder_indices], axis = 0)
        if np.any(self.valid_keypoints[head_and_torso_indices]):
            head_and_torso_keypoints = np.full(18, False)
            head_and_torso_keypoints[head_and_torso_indices] = True
            valid_head_and_torso_keypoints = np.logical_and(
                head_and_torso_keypoints,
                self.valid_keypoints)
            return np.mean(self.keypoints[valid_head_and_torso_keypoints], axis = 0)
        return np.mean(self.keypoints[self.valid_keypoints], axis = 0)

    # Given a set of camera calibration parameters, project this 3D pose into
    # the camera coordinate system to produce a 2D pose
    def to_pose_2d(
        self,
        camera):
        keypoints_2d = cvutilities.camera_utilities.project_points(
            self.keypoints,
            camera['rotation_vector'],
            camera['translation_vector'],
            camera['camera_matrix'],
            camera['distortion_coefficients'])
        valid_keypoints_2d =  self.valid_keypoints
        confidence_scores_2d = np.repeat(np.nan, num_body_parts)
        tag_2d = self.tag
        timestamp_2d = self.timestamp
        pose_2d = Pose2D(
            keypoints_2d,
            confidence_scores_2d,
            valid_keypoints_2d,
            tag_2d,
            timestamp_2d)
        return pose_2d

    # Draw the 3D pose onto a chart representing a top-down view of the room. We
    # separate this from the plotting function below because we might want to
    # draw several poses or other elements before formatting and showing the
    # chart
    def draw_topdown(self):
        plottable_points = self.keypoints[self.valid_keypoints]
        cvutilities.camera_utilities.draw_3d_object_points_topdown(plottable_points)

    # Plot a pose onto a chart representing a top-down view of the room. Calls
    # the drawing function above, adds formating, and shows the plot
    def plot_topdown(
        self,
        room_corners = None):
        self.draw_topdown()
        cvutilities.camera_utilities.format_3d_topdown_plot(room_corners)
        plt.show()

# Class to represent a collection of 3D poses, either a set of poses over time
# (as from a pose track) or a set of poses at a single point in time (as from a
# set of observations or the last poses in a set of pose tracks or the output of
# a 3D reconstruction from 2D poses) or set of sets of poses over time (as from
# a collection of pose tracks). Internal structure is a list of lists of Pose3D
# objects. When working with one-dimensional collections, input methods add the
# additional dimension and output methods remove the addtional dimension as
# necessary
class Poses3D:
    def __init__(
        self,
        pose_3d_list_list,
        num_cameras_source_images = None,
        num_2d_poses_source_images = None,
        source_cameras = None,
        source_images = None):
        self.pose_3d_list_list = pose_3d_list_list
        self.num_cameras_source_images = num_cameras_source_images
        self.num_2d_poses_source_images = num_2d_poses_source_images
        self.source_cameras = source_cameras
        self.source_images = source_images

    # Build a 3D poses object from a simple array of keypoints. Assume that
    # non-valid keypoints are indicated by numpy NAN values
    @classmethod
    def from_keypoints(
        cls,
        keypoints,
        timestamp = None):
        keypoints = np.asarray(keypoints)
        if keypoints.size == 0:
            return cls([[]])
        if keypoints.ndim > 4 or keypoints.ndim < 2:
            print(keypoints.ndim)
            print(keypoints)
            raise ValueError('Keypoints array needs to be of dimensions 4, 3, 2, or 0 (empty)')
        if keypoints.ndim == 3:
            keypoints = np.expand_dims(keypoints, 0)
        if keypoints.ndim == 2:
            keypoints = np.expand_dims(np.expand_dims(keypoints, 0), 0)
        num_pose_lists = keypoints.shape[0]
        num_poses = keypoints.shape[1]
        pose_3d_list_list = []
        for pose_list_index in range(num_pose_lists):
            pose_3d_list = []
            for pose_index in range(num_poses):
                pose_3d = Pose3D.from_keypoints(
                    keypoints[pose_list_index, pose_index],
                    timestamp = timestamp)
                pose_3d_list.append(pose_3d)
            pose_3d_list_list.append(pose_3d_list)
        return cls(pose_3d_list_list)

    # Return the number of 3D poses in each list
    def num_3d_poses(self):
        num_3d_poses_list = [len(pose_3d_list) for pose_3d_list in self.pose_3d_list_list]
        if len(num_3d_poses_list) == 1:
            return num_3d_poses_list[0]
        else:
            return num_3d_poses_list

    # Return the number of 3D poses in the lists
    def total_num_3d_poses(self):
        return np.sum(np.array(self.num_3d_poses()))

    # Return the keypoints for all of the 3D poses in the lists
    def keypoints(self):
        keypoints_list = [np.array([pose_3d.keypoints for pose_3d in pose_3d_list]) for pose_3d_list in self.pose_3d_list_list]
        if len(keypoints_list) == 1:
            return keypoints_list[0]
        else:
            return keypoints_list

    # Return the keypoint standard deviations for all of the 3D poses in the
    # lists
    def keypoint_std_devs(self):
        keypoint_std_devs_list = [np.array([pose_3d.keypoint_std_devs for pose_3d in pose_3d_list]) for pose_3d_list in self.pose_3d_list_list]
        if len(keypoint_std_devs_list) == 1:
            return keypoint_std_devs_list[0]
        else:
            return keypoint_std_devs_list

    # Return the valid keypoints Boolean vector for all of the 3D poses in the
    # lists
    def valid_keypoints(self):
        valid_keypoints_list = [np.array([pose_3d.valid_keypoints for pose_3d in pose_3d_list]) for pose_3d_list in self.pose_3d_list_list]
        if len(valid_keypoints_list) == 1:
            return valid_keypoints_list[0]
        else:
            return valid_keypoints_list

    # Return the projection errors for all of the 3D poses in the lists
    def projection_errors(self):
        projection_errors_list = [np.array([pose_3d.projection_error for pose_3d in pose_3d_list]) for pose_3d_list in self.pose_3d_list_list]
        if len(projection_errors_list) == 1:
            return projection_errors_list[0]
        else:
            return projection_errors_list

    # Return the tags for all of the 3D poses in the lists
    def tags(self):
        tags_list = [[pose_3d.tag for pose_3d in pose_3d_list] for pose_3d_list in self.pose_3d_list_list]
        if len(tags_list) == 1:
            return tags_list[0]
        else:
            return tags_list

    # Return the timestamps for all of the 3D poses in the lists
    def timestamps(self):
        timestamps_list = [np.array([pose_3d.timestamp for pose_3d in pose_3d_list]) for pose_3d_list in self.pose_3d_list_list]
        if len(timestamps_list) == 1:
            return timestamps_list[0]
        else:
            return timestamps_list

    # Return a dataframe with the data from the 3D poses
    def dataframe(self):
        flattened_pose_3d_list = [pose_3d for pose_3d_list in self.pose_3d_list_list for pose_3d in pose_3d_list]
        dataframe = pd.DataFrame({
            'timestamp': [pose_3d.timestamp for pose_3d in flattened_pose_3d_list],
            'keypoints': [pose_3d.keypoints for pose_3d in flattened_pose_3d_list],
            'valid_keypoints': [pose_3d.valid_keypoints for pose_3d in flattened_pose_3d_list],
            'keypoint_std_devs': [pose_3d.keypoint_std_devs for pose_3d in flattened_pose_3d_list],
            'tag': [pose_3d.tag for pose_3d in flattened_pose_3d_list]})
        return(dataframe)

    # Using the camera calibration parameters from the original source images,
    # project this collection of 3D poses back into the coordinate system for
    # each camera to produce a collection of 2D poses
    def to_poses_2d(self):
        if self.source_cameras is None:
            raise ValueError('Source camera calibration data not specified')
        cameras=[]
        for camera_index in range(self.num_cameras_source_images):
            poses = []
            if self.total_num_3d_poses() > 0:
                for pose_3d_list_index in range(len(self.pose_3d_list_list)):
                    for pose_3d_index in range(len(self.pose_3d_list_list[pose_3d_list_index])):
                        pose_2d = self.pose_3d_list_list[pose_3d_list_index][pose_3d_index].to_pose_2d(self.source_cameras[camera_index])
                        if pose_2d.tag is None:
                            pose_2d.set_tag(pose_index_3d)
                        poses.append(pose_2d)
            cameras.append(poses)
        return Poses2D(cameras, self.source_images)

    # Plot the lists of poses onto charts representing a top-down view of the
    # room, one chart per list
    def plot_topdown(
        self,
        room_corners = None):
        for pose_3d_list in self.pose_3d_list_list:
            for pose_3d in pose_3d_list:
                pose_3d.draw_topdown()
                if pose_3d.tag is not None:
                    plottable_points = pose_3d.keypoints[pose_3d.valid_keypoints]
                    centroid = np.mean(plottable_points[:, :2], 0)
                    plt.text(centroid[0], centroid[1], pose_3d.tag)
            cvutilities.camera_utilities.format_3d_topdown_plot(room_corners)
            plt.show()

# Class to represent a collection of 3D poses reconstructed from 2D poses across
# multiple cameras at a single moment in time. This data has a natural graph
# structure, with 2D poses as the nodes and 3D poses as the edges
class Pose3DGraph:
    def __init__(
        self,
        pose_graph,
        num_cameras_source_images,
        num_2d_poses_source_images,
        source_cameras = None,
        source_images = None):
        self.pose_graph = pose_graph
        self.num_cameras_source_images = num_cameras_source_images
        self.num_2d_poses_source_images = num_2d_poses_source_images
        self.source_cameras = source_cameras
        self.source_images = source_images

    # Calculate all possible 3D poses at a single moment in time (from every
    # pair of 2D poses across every pair of cameras)
    @classmethod
    def from_poses_2d_timestep(
        cls,
        poses_2d,
        cameras):
        pose_graph = nx.Graph()
        num_cameras_source_images = poses_2d.num_cameras()
        num_2d_poses_source_images = poses_2d.num_poses()
        source_images = poses_2d.source_images
        for camera_index_a in range(num_cameras_source_images - 1):
            for camera_index_b in range(camera_index_a + 1, num_cameras_source_images):
                num_poses_a = poses_2d.num_poses()[camera_index_a]
                num_poses_b = poses_2d.num_poses()[camera_index_b]
                for pose_index_a in range(num_poses_a):
                    for pose_index_b in range(num_poses_b):
                        pose_3d = Pose3D.from_poses_2d(
                            poses_2d.pose_2d_list_list[camera_index_a][pose_index_a],
                            poses_2d.pose_2d_list_list[camera_index_b][pose_index_b],
                            cameras[camera_index_a]['rotation_vector'],
                            cameras[camera_index_a]['translation_vector'],
                            cameras[camera_index_b]['rotation_vector'],
                            cameras[camera_index_b]['translation_vector'],
                            cameras[camera_index_a]['camera_matrix'],
                            cameras[camera_index_a]['distortion_coefficients'])
                        pose_graph.add_edge(
                            (camera_index_a, pose_index_a),
                            (camera_index_b, pose_index_b),
                            pose=pose_3d)
        return cls(
            pose_graph,
            num_cameras_source_images,
            num_2d_poses_source_images,
            cameras,
            source_images)

    # Return the number of 3D poses (edges) in the collection
    def total_num_3d_poses(self):
        return self.pose_graph.number_of_edges()

    # Return the number of 2D poses (nodes) in the collection
    def total_num_2d_poses(self):
        return self.pose_graph.number_of_nodes()

    # Return the camera and pose indices for the source 2D poses correponding to
    # each 3D pose in the collection
    def pose_indices(self):
        return np.asarray(list(self.pose_graph.edges))

    # Return the 3D pose objects themselves (instances of the 3DPose class
    # above)
    def poses_3d(self):
        return Poses3D(
            pose_3d_list_list = [[edge[2]['pose'] for edge in list(self.pose_graph.edges.data())]],
            num_cameras_source_images = self.num_cameras_source_images,
            num_2d_poses_source_images = self.num_2d_poses_source_images,
            source_cameras = self.source_cameras,
            source_images = self.source_images)

    # Return the keypoints for all of the 3D poses in the collection
    def keypoints(self):
        return self.poses_3d().keypoints()

    # Return the valid keypoints Boolean vector for all of the 3D poses in the
    # collection.
    def valid_keypoints(self):
        return self.poses_3d().valid_keypoints()

    # Return the projection errors for all of the 3D poses in the collection
    def projection_errors(self):
        return self.poses_3d().projection_errors()

    # Return the tags for all of the 3D poses in the collection.
    def tags(self):
        return self.poses_3d().tags()

    # Return the timestamps for all of the 3D poses in the collection.
    def timestamps(self):
        return self.poses_3d().timestamps()

    # Using the camera calibration parameters from the original source images,
    # project this collection of 3D poses back into the coordinate system for
    # each camera to produce a collection of 2D poses
    def to_poses_2d(self):
        return self.poses_3d().to_poses_2d()

    # Draw the graph representing all of the 3D poses in the collection (2D
    # poses as nodes, 3D poses as edges)
    def draw_graph(self):
        nx.draw(self.pose_graph, with_labels=True, font_weight='bold')

    # Starting with a collection of 3D poses representing all possible matches,
    # for each camera pair, pull out the best match for each 2D pose across each
    # pair, with the contraint that (1) If pose A is the best match for pose B
    # then pose B must be the best match for pose A, and (2) The reprojection
    # error has to be below a threshold
    def extract_likely_matches_from_all_matches(
        self,
        projection_error_threshold = 15.0):
        # For now, we initialize a new empty graph and copy selected edges into
        # it. We should really do this either by creating a copy of the original
        # graph and deleting the edges we don't want or by tracking pointers
        # back to the original graph
        likely_matches_graph = nx.Graph()
        for camera_index_a in range(self.num_cameras_source_images - 1):
            for camera_index_b in range(camera_index_a + 1, self.num_cameras_source_images):
                num_poses_a = self.num_2d_poses_source_images[camera_index_a]
                num_poses_b = self.num_2d_poses_source_images[camera_index_b]
                # For each pair of cameras, we build an array of projection errors
                # because it's easier to express our matching rule as a rule on an array
                # rather than a rule on the graph
                projection_errors = np.full((num_poses_a, num_poses_b), np.nan)
                for pose_index_a in range(num_poses_a):
                    for pose_index_b in range(num_poses_b):
                        if self.pose_graph.has_edge(
                            (camera_index_a, pose_index_a),
                            (camera_index_b, pose_index_b)):
                            projection_errors[pose_index_a, pose_index_b] = self.pose_graph[(camera_index_a, pose_index_a)][(camera_index_b, pose_index_b)]['pose'].projection_error
                # Apply our matching rule to the array of projection errors.
                for pose_index_a in range(num_poses_a):
                    for pose_index_b in range(num_poses_b):
                        if (
                            not np.all(np.isnan(projection_errors[pose_index_a, :])) and
                            np.nanargmin(projection_errors[pose_index_a, :]) == pose_index_b and
                            not np.all(np.isnan(projection_errors[:, pose_index_b])) and
                            np.nanargmin(projection_errors[:, pose_index_b]) == pose_index_a and
                            projection_errors[pose_index_a, pose_index_b] < projection_error_threshold):
                            likely_matches_graph.add_edge(
                                (camera_index_a, pose_index_a),
                                (camera_index_b, pose_index_b),
                                pose=self.pose_graph[(camera_index_a, pose_index_a)][(camera_index_b, pose_index_b)]['pose'])
        likely_matches = self.__class__(
            likely_matches_graph,
            self.num_cameras_source_images,
            self.num_2d_poses_source_images,
            self.source_cameras,
            self.source_images)
        return likely_matches

    # Starting with a collection of 3D poses representing likely matches, for
    # each connected subgraph (which now represents a set of poses connected
    # across camera pairs that ought to be the same person), we extract the
    # match with the lowest reprojection error (we could average instead)
    def extract_best_matches_from_likely_matches(
        self):
        # For now, we make a copy of each subgraph of the likely matches, select
        # the best edge from each subgraph, and copy that best edge into a new
        # graph. We should really do this either by deleting all edges from the
        # pruned graph other than the best one for each subgraph or by tracking
        # pointers back to the original graph
        best_matches_graph = nx.Graph()
        subgraphs_list = [self.pose_graph.subgraph(component).copy() for component in nx.connected_components(self.pose_graph)]
        for subgraph_index in range(len(subgraphs_list)):
            if nx.number_of_edges(subgraphs_list[subgraph_index]) > 0:
                best_edge = sorted(subgraphs_list[subgraph_index].edges.data(), key = lambda x: x[2]['pose'].projection_error)[0]
                best_matches_graph.add_edge(best_edge[0], best_edge[1], pose = best_edge[2]['pose'])
        best_matches = self.__class__(
            best_matches_graph,
            self.num_cameras_source_images,
            self.num_2d_poses_source_images,
            self.source_cameras,
            self.source_images)
        return best_matches

    # Starting with a collection of 3D poses representing all possible matches,
    # scan through all of the 3D poses and pull out a set of best matches, one
    # for each person in the room (combines the two methods above)
    def extract_matched_poses(
        self,
        projection_error_threshold = 15.0):
        likely_matches = self.extract_likely_matches_from_all_matches(projection_error_threshold)
        best_matches = likely_matches.extract_best_matches_from_likely_matches()
        return best_matches

    # Plot the poses onto a chart representing a top-down view of the room.
    # Calls the drawing function above, adds formating, and shows the plot
    def plot_topdown(
        self,
        room_corners = None):
        self.poses_3d().plot_topdown(room_corners)

# Class to define a model for each keypoint. We use a simple constant-velocity
# model in which only position is observed. We specify transition error (drift)
# for a reference time interval, so we can scale this error for longer and
# shorter time intervals. If no observation error is specified, the model can
# only be used for prediction. If no transition error is specified, the model
# can only be used for observation.
class KeypointModel:
    def __init__(
        self,
        reference_delta_t = None,
        reference_position_transition_error = None,
        reference_velocity_transition_error = None,
        position_observation_error = None):
        self.reference_delta_t = reference_delta_t
        self.reference_position_transition_error = reference_position_transition_error
        self.reference_velocity_transition_error = reference_velocity_transition_error
        self.position_observation_error = position_observation_error

    # Based on the keypoint model and a specified time interval, generate a
    # complete linear Gaussian model (in the format expected by the smc_kalman
    # package). We allow for the possibility that no time interval is specified,
    # in which case the resulting linear Gaussian model can only be used for
    # observation, not prediction
    def linear_gaussian_model(
        self,
        delta_t = None):
        if delta_t is not None:
            if self.reference_delta_t is None:
                raise ValueError('Transition model not specified.')
            position_transition_error = self.reference_position_transition_error*np.sqrt(
                np.divide(
                    delta_t,
                    self.reference_delta_t))
            velocity_transition_error = self.reference_velocity_transition_error*np.sqrt(
                np.divide(
                    delta_t,
                    self.reference_delta_t))
            keypoint_transition_model = np.concatenate((
                np.concatenate(
                    (np.identity(3), delta_t*np.identity(3)),
                    axis=1),
                np.concatenate(
                    (np.zeros((3,3)), np.identity(3)),
                    axis=1)),
                axis=0)
            keypoint_transition_noise_covariance = np.diagflat(
                np.concatenate((
                    np.repeat(position_transition_error**2, 3),
                    np.repeat(velocity_transition_error**2, 3))))
        else:
            keypoint_transition_model = None
            keypoint_transition_noise_covariance = None
        keypoint_control_model = None
        if self.position_observation_error is not None:
            keypoint_observation_model = np.concatenate((
                np.identity(3),
                np.zeros((3,3))),
                axis = 1)
            keypoint_observation_noise_covariance = (self.position_observation_error**2)*np.identity(3)
        else:
            keypoint_observation_model = None
            keypoint_observation_noise_covariance = None
        keypoint_linear_gaussian_model = smc_kalman.LinearGaussianModel(
            keypoint_transition_model,
            keypoint_transition_noise_covariance,
            keypoint_observation_model,
            keypoint_observation_noise_covariance,
            keypoint_control_model)
        return keypoint_linear_gaussian_model

# Class to define the initial parameters of a 3D pose distribution
class PoseInitializationModel:
    def __init__(
        self,
        initial_keypoint_position_means,
        initial_keypoint_velocity_means,
        initial_keypoint_position_error,
        initial_keypoint_velocity_error):
        initial_keypoint_position_means = np.asarray(initial_keypoint_position_means)
        initial_keypoint_velocity_means = np.asarray(initial_keypoint_velocity_means)
        initial_keypoint_position_error = np.asarray(initial_keypoint_position_error)
        initial_keypoint_velocity_error = np.asarray(initial_keypoint_velocity_error)
        if initial_keypoint_position_means.shape != (num_body_parts, 3):
            raise ValueError('Initial position means array does not appear to be of shape ({}, 3)'.format(num_body_parts))
        if initial_keypoint_velocity_means.shape != (num_body_parts, 3):
            raise ValueError('Initial velocity means array does not appear to be of shape ({}, 3)'.format(num_body_parts))
        if initial_keypoint_position_error.size != 1:
            raise ValueError('Initial position error does not appear to be a scalar'.format(num_body_parts))
        if initial_keypoint_velocity_error.size != 1:
            raise ValueError('Initial velocity error does not appear to be a scalar'.format(num_body_parts))
        initial_keypoint_position_error = np.asscalar(initial_keypoint_position_error)
        initial_keypoint_velocity_error = np.asscalar(initial_keypoint_velocity_error)
        self.initial_keypoint_position_means = initial_keypoint_position_means
        self.initial_keypoint_velocity_means = initial_keypoint_velocity_means
        self.initial_keypoint_position_error = initial_keypoint_position_error
        self.initial_keypoint_velocity_error = initial_keypoint_velocity_error

# Class to hold the data for a set of Gaussian distributions describing a 3D
# pose (one three-dimensional Gaussian distribution describing the position and
# velocity of each body part). Internal structure is a list of
# smc_kalman.GaussianDistribution objects
class Pose3DDistribution:
    def __init__(
        self,
        keypoint_distributions,
        tag = None,
        timestamp = None):
        if len(keypoint_distributions) != num_body_parts:
            raise ValueError('List of keypoint distributions is not of length {}'.format(num_body_parts))
        self.keypoint_distributions = keypoint_distributions
        self.tag = tag
        self.timestamp = timestamp

    # Initialize the 3D pose distribution
    @classmethod
    def initialize(
        cls,
        pose_initialization_model,
        tag = None,
        timestamp = None,
        keypoint_model = None,
        pose_3d_observation = None):
        keypoint_covariance = np.diagflat(
            np.concatenate((
                np.repeat(pose_initialization_model.initial_keypoint_position_error**2,3),
                np.repeat(pose_initialization_model.initial_keypoint_velocity_error**2, 3))))
        keypoint_distributions=[]
        for body_part_index in range(num_body_parts):
            keypoint_mean = np.concatenate((
                pose_initialization_model.initial_keypoint_position_means[body_part_index],
                pose_initialization_model.initial_keypoint_velocity_means[body_part_index]))
            keypoint_distribution = smc_kalman.GaussianDistribution(
                keypoint_mean,
                keypoint_covariance)
            keypoint_distributions.append(keypoint_distribution)
        initial_distribution = cls(
            keypoint_distributions,
            tag,
            timestamp)
        if pose_3d_observation is not None:
            if keypoint_model is None:
                raise ValueError('Observation provided but no keypoint model is specified')
            initial_distribution = initial_distribution.incorporate_observation(
                keypoint_model,
                pose_3d_observation)
        return initial_distribution

    # Return keypoint means
    def keypoint_means(self):
        return np.array([keypoint_distribution.mean for keypoint_distribution in self.keypoint_distributions])

    # Return keypoint covariances
    def keypoint_covariances(self):
        return np.array([keypoint_distribution.covariance for keypoint_distribution in self.keypoint_distributions])

    # Return keypoint position means
    def keypoint_position_means(self):
        return self.keypoint_means()[:, :3]

    # Return keypoint velocity means
    def keypoint_velocity_means(self):
        return self.keypoint_means()[:, 3:]

    # Return keypoint standard deviations
    def keypoint_std_devs(self):
        return np.sqrt(np.array([np.diag(keypoint_distribution.covariance) for keypoint_distribution in self.keypoint_distributions]))

    # Return keypoint position standard deviations
    def keypoint_position_std_devs(self):
        return self.keypoint_std_devs()[:, :3]

    # Return keypoint velocity standard deviations
    def keypoint_velocity_std_devs(self):
        return self.keypoint_std_devs()[:, 3:]

    # Construct and return a 3D pose with the keypoints equal to the mean
    # position keypoints of the distribution and position standard deviations
    # equal to the position standard deviations of the distribution
    def to_pose_3d(self):
        pose_3d = Pose3D(
            keypoints = self.keypoint_position_means(),
            valid_keypoints = np.full(18, True),
            projection_error = None,
            tag = self.tag,
            timestamp = self.timestamp,
            keypoint_std_devs = self.keypoint_position_std_devs())
        return pose_3d

    # Given a keypoint model and a time interval, apply the model to calculate
    # the next 3D pose distribution. User can specify time interval explicitly
    # or method will attempt to infer from ending timestamp. Time unit is
    # specified above. In the future, we may want to make underlying functions
    # be able to handle time intervals with units
    def predict(
        self,
        keypoint_model,
        delta_t = None,
        next_timestamp = None):
        current_keypoint_distributions = self.keypoint_distributions
        current_tag = self.tag
        current_timestamp = self.timestamp
        if delta_t is not None and next_timestamp is not None:
            raise ValueError('Specify either time interval or ending timestamp but not both')
        if delta_t is None:
            if current_timestamp is None:
                raise ValueError('Time interval not specified and cannot be inferred')
            delta_t = (next_timestamp - current_timestamp)/time_unit
        else:
            if current_timestamp is not None:
                next_timestamp = current_timestamp + delta_t*time_unit
        keypoint_linear_gaussian_model = keypoint_model.linear_gaussian_model(delta_t)
        next_keypoint_distributions = []
        for body_part_index in range(num_body_parts):
            next_keypoint_distribution = current_keypoint_distributions[body_part_index].predict(
                keypoint_linear_gaussian_model)
            next_keypoint_distributions.append(next_keypoint_distribution)
        next_tag = current_tag
        next_pose_3d_distribution = Pose3DDistribution(
            next_keypoint_distributions,
            next_tag,
            next_timestamp)
        return next_pose_3d_distribution

    # Given a keypoint model and an observation of a 3D pose (specified as a
    # Pose3D object), apply the model to calculate the posterior 3D pose
    # distribution which incorporates the information from this observation. For
    # any keypoints we don't observe, the keypoint distribution remains
    # unchanged
    def incorporate_observation(
        self,
        keypoint_model,
        pose_3d_observation):
        prior_keypoint_distributions = self.keypoint_distributions
        prior_tag = self.tag
        prior_timestamp = self.timestamp
        observation_timestamp = pose_3d_observation.timestamp
        if prior_timestamp is not None and observation_timestamp is not None and prior_timestamp != observation_timestamp:
            raise ValueError('Timestamp on observation does not match timestamp on observed state')
        keypoint_linear_gaussian_model = keypoint_model.linear_gaussian_model()
        posterior_keypoint_distributions = []
        for body_part_index in range(num_body_parts):
            if pose_3d_observation.valid_keypoints[body_part_index]:
                posterior_keypoint_distribution = prior_keypoint_distributions[body_part_index].incorporate_observation(
                    keypoint_linear_gaussian_model,
                    pose_3d_observation.keypoints[body_part_index])
            else:
                posterior_keypoint_distribution = prior_keypoint_distributions[body_part_index]
            posterior_keypoint_distributions.append(posterior_keypoint_distribution)
        if prior_tag is None:
            posterior_tag = pose_3d_observation.tag
        else:
            posterior_tag = prior_tag
        if observation_timestamp is not None:
            posterior_timestamp = observation_timestamp
        else:
            posterior_timestamp = prior_timestamp
        posterior_pose_3d_distribution =  Pose3DDistribution(
            posterior_keypoint_distributions,
            posterior_tag,
            posterior_timestamp)
        return posterior_pose_3d_distribution

    # Given a keypoint model and an observation of a 3D pose (specified as a
    # Pose3D object), apply the model to predict the distribution at the time of
    # the observation and then calculate the posterior 3D pose distribution
    # which incorporates the information from the observation (combines the
    # methods above)
    def predict_and_incorporate_observation(
        self,
        keypoint_model,
        pose_3d_observation):
        if pose_3d_observation.timestamp is None:
            raise ValueError('Observation must have timestamp in order to drive prediction step')
        observation_timestamp = pose_3d_observation.timestamp
        prior_distribution = self.predict(
            keypoint_model,
            next_timestamp = observation_timestamp)
        posterior_distribution = self.incorporate_observation(
            keypoint_model,
            pose_3d_observation)
        return posterior_distribution

    # Given a keypoint model and an observation of a 3D pose (specified as a
    # Pose3D object), calculate the Mahalanobis distance between the anchor
    # point of the pose and the anchor point of the observation
    def observation_mahalanobis_distance(
        self,
        keypoint_model,
        pose_3d_observation):
        anchor_point_state_distribution = self.keypoint_distributions[neck_index]
        anchor_point_observation = pose_3d_observation.anchor_point()
        keypoint_linear_gaussian_model = keypoint_model.linear_gaussian_model()
        observation_mahalanobis_distance = anchor_point_state_distribution.observation_mahalanobis_distance(
            keypoint_linear_gaussian_model,
            anchor_point_observation)
        observation_mahalanobis_distance = np.asscalar(observation_mahalanobis_distance)
        return observation_mahalanobis_distance

# Class to represent a 3D pose track: a collection of 3D pose distributions
# describing the path of a person over time. Internal structure is a list of
# Pose3DDistribution objects, one for each moment in time
class Pose3DTrack:
    def __init__(
        self,
        keypoint_model,
        pose_3d_distributions,
        track_id = None,
        num_missed_observations = 0):
        self.keypoint_model = keypoint_model
        self.pose_3d_distributions = pose_3d_distributions
        self.track_id = track_id
        self.num_missed_observations = num_missed_observations

    # Initialize the track
    @classmethod
    def initialize(
        cls,
        pose_initialization_model,
        keypoint_model,
        tag = None,
        timestamp = None,
        track_id = None,
        pose_3d_observation = None):
        pose_3d_distributions = [Pose3DDistribution.initialize(
            pose_initialization_model,
            tag,
            timestamp)]
        initial_track = cls(
            keypoint_model,
            pose_3d_distributions,
            track_id = track_id)
        if pose_3d_observation is not None:
            initial_track.incorporate_observation(pose_3d_observation)
        return initial_track

    # Return timestamps
    def timestamps(self):
        return np.array([pose_3d_distribution.timestamp for pose_3d_distribution in self.pose_3d_distributions])

    # Return keypoint means
    def keypoint_means(self):
        return np.array([[keypoint_distribution.mean for keypoint_distribution in pose_3d_distribution.keypoint_distributions] for pose_3d_distribution in self.pose_3d_distributions])

    # Return keypoint covariances
    def keypoint_covariances(self):
        return np.array([[keypoint_distribution.covariance for keypoint_distribution in pose_3d_distribution.keypoint_distributions] for pose_3d_distribution in self.pose_3d_distributions])

    # Return keypoint position means
    def keypoint_position_means(self):
        return self.keypoint_means()[:, :, :3]

    # Return keypoint velocity means
    def keypoint_velocity_means(self):
        return self.keypoint_means()[:, :, 3:]

    # Return keypoint standard deviations
    def keypoint_std_devs(self):
        return np.array([np.sqrt(np.array([np.diag(keypoint_distribution.covariance) for keypoint_distribution in pose_3d_distribution.keypoint_distributions])) for pose_3d_distribution in self.pose_3d_distributions])

    # Return keypoint position standard deviations
    def keypoint_position_std_devs(self):
        return self.keypoint_std_devs()[:, :, :3]

    # Return keypoint velocity standard deviations
    def keypoint_velocity_std_devs(self):
        return self.keypoint_std_devs()[:, :, 3:]

    # Return the last 3D pose distribution in the list
    def last(self):
        return self.pose_3d_distributions[-1]

    # Construct and return a list of 3D poses with keypoints equal to the the
    # mean position keypoints of the distributions in the track
    def to_poses_3d(self):
        poses_3d = Poses3D([[pose_3d_distribution.to_pose_3d() for pose_3d_distribution in self.pose_3d_distributions]])
        return poses_3d

    # Construct and return a dataframe with the 3D pose data from the pose track
    def dataframe(self):
        dataframe = self.to_poses_3d().dataframe()
        return dataframe

    # Construct and return the 3D pose with the mean position keypoints
    # of the last distribution in the track
    def last_to_pose_3d(self):
        pose_3d = self.last().to_pose_3d()
        return pose_3d

    # Append a 3D pose distribution to the track
    def append(
        self,
        pose_3d_distribution):
        self.pose_3d_distributions.append(pose_3d_distribution)

    # Given a keypoint model and a time interval, apply the model to the last
    # pose distribution in the track to calculate the next pose distribution and
    # add this new pose distribution to the track. User can specify time
    # interval explicitly or method will attempt to infer from ending timestamp.
    # Time unit is specified above. In the future, we may want to make
    # underlying functions be able to handle time intervals with units.
    def predict(
        self,
        delta_t = None,
        next_timestamp = None):
        current_pose_3d_distribution = self.last()
        next_pose_3d_distribution = current_pose_3d_distribution.predict(
            self.keypoint_model,
            delta_t,
            next_timestamp)
        self.append(next_pose_3d_distribution)

    # Given a keypoint model and an observation of the 3D pose (specified as a
    # Pose3D object), apply the model to the last pose distribution in the track
    # to calculate the posterior 3D pose distribution which incorporates the
    # information from this observation. Replace the last pose distribution in
    # the track with this posterior distribution
    def incorporate_observation(
        self,
        pose_3d_observation):
        prior_pose_3d_distribution = self.last()
        posterior_pose_3d_distribution = prior_pose_3d_distribution.incorporate_observation(
            self.keypoint_model,
            pose_3d_observation)
        self.pose_3d_distributions[-1] = posterior_pose_3d_distribution

    # Given a keypoint model and an observation of the 3D pose (specified as a
    # Pose3D object), apply the model to the last pose distribution in the track
    # to predict the distribution at the time of the observation and then
    # calculate the posterior 3D pose distribution which incorporates the
    # information from this observation. Add this new pose distribution to the
    # track (combines the two methods above)
    def predict_and_incorporate_observation(
        self,
        pose_3d_observation):
        if pose_3d_observation.timestamp is None:
            raise ValueError('Observation must have timestamp in order to drive prediction step')
        observation_timestamp = pose_3d_observation.timestamp
        self.predict(next_timestamp = observation_timestamp)
        self.incorporate_observation(pose_3d_observation)

    # Given a keypoint model and an observation of a 3D pose (specified as a
    # Pose3D object), calculate the Mahalanobis distance between the anchor
    # point of the last pose distribution in the track and the anchor point of
    # the observation. Keypoint model is an instance of the KeypointModel class
    def observation_mahalanobis_distance(
        self,
        pose_3d_observation):
        observation_mahalanobis_distance = self.last().observation_mahalanobis_distance(
            self.keypoint_model,
            pose_3d_observation)
        return observation_mahalanobis_distance

# Class to define the parameters of the pose tracking model
class PoseTrackingModel:
    def __init__(
        self,
        cost_threshold = 1.0,
        num_missed_observations_threshold = 10):
        self.cost_threshold = cost_threshold
        self.num_missed_observations_threshold = num_missed_observations_threshold

# Class to represent a collection of 3D pose tracks. We allow this object to
# contain both active tracks and inactive tracks. The active tracks are assumed
# to be synchronized in the sense that all of the last distributions in the
# active tracks are assumed to describe the same moment in time
class Pose3DTracks:
    def __init__(
        self,
        pose_initialization_model,
        keypoint_model,
        pose_tracking_model,
        active_tracks = None,
        inactive_tracks = None,
        next_track_id = 0):
        if active_tracks is not None:
            check_last_timestamps(active_tracks)
        if active_tracks is None:
            active_tracks = []
        if inactive_tracks is None:
            inactive_tracks = []
        self.pose_initialization_model = pose_initialization_model
        self.keypoint_model = keypoint_model
        self.pose_tracking_model = pose_tracking_model
        self.active_tracks = active_tracks
        self.inactive_tracks = inactive_tracks
        self.next_track_id = next_track_id

    # Initialize the tracks
    @classmethod
    def initialize(
        cls,
        pose_initialization_model,
        keypoint_model,
        pose_tracking_model,
        tag = None,
        timestamp = None,
        num_tracks = None,
        pose_3d_observations = None):
        if num_tracks is None and pose_3d_observations is None:
            raise ValueError('Must specify either the number of tracks or a set of initial observations')
        if num_tracks is not None and pose_3d_observations is not None:
            raise ValueError('Must specify either the number of tracks or a set of initial observations but not both')
        if pose_3d_observations is not None:
            if len(pose_3d_observations.pose_3d_list_list) != 1:
                raise ValueError('Observations must be a one-dimensional object')
            num_tracks = len(pose_3d_observations.pose_3d_list_list[0])
        active_tracks=[]
        for track_index in range(num_tracks):
            track = Pose3DTrack.initialize(
                pose_initialization_model,
                keypoint_model,
                tag = None,
                timestamp = None,
                track_id = track_index)
            active_tracks.append(track)
        initial_tracks = cls(
            pose_initialization_model = pose_initialization_model,
            keypoint_model = keypoint_model,
            pose_tracking_model = pose_tracking_model,
            active_tracks = active_tracks,
            inactive_tracks = None,
            next_track_id = num_tracks)
        if pose_3d_observations is not None:
            initial_tracks.incorporate_observations(
                pose_3d_observations,
                selected_track_indices = range(num_tracks),
                selected_observation_indices = range(num_tracks))
        return initial_tracks

    # Return number of active tracks
    def num_active_tracks(self):
        return len(self.active_tracks)

    # Return number of inactive tracks
    def num_inactive_tracks(self):
        return len(self.inactive_tracks)

    # Return timestamps for active tracks
    def active_timestamps(self):
        return [np.array([pose_3d_distribution.timestamp for pose_3d_distribution in active_track.pose_3d_distributions]) for active_track in self.active_tracks]

    # Return track IDs for active tracks
    def active_track_ids(self):
        return np.array([active_track.track_id for active_track in self.active_tracks])

    # Return number of missed observations for active tracks
    def active_num_missed_observations(self):
        return np.array([active_track.num_missed_observations for active_track in self.active_tracks])

    # Return keypoint means for active tracks
    def active_keypoint_means(self):
        return [np.array([[keypoint_distribution.mean for keypoint_distribution in pose_3d_distribution.keypoint_distributions] for pose_3d_distribution in active_track.pose_3d_distributions]) for active_track in self.active_tracks]

    # Return keypoint covariances for active tracks
    def active_keypoint_covariances(self):
        return [np.array([[keypoint_distribution.covariance for keypoint_distribution in pose_3d_distribution.keypoint_distributions] for pose_3d_distribution in active_track.pose_3d_distributions]) for active_track in self.active_tracks]

    # Return keypoint position means for active tracks
    def active_keypoint_position_means(self):
        return [np.array([[keypoint_distribution.mean[:3] for keypoint_distribution in pose_3d_distribution.keypoint_distributions] for pose_3d_distribution in active_track.pose_3d_distributions]) for active_track in self.active_tracks]

    # Return keypoint velocity means for active tracks
    def active_keypoint_velocity_means(self):
        return [np.array([[keypoint_distribution.mean[3:] for keypoint_distribution in pose_3d_distribution.keypoint_distributions] for pose_3d_distribution in active_track.pose_3d_distributions]) for active_track in self.active_tracks]

    # Return keypoint standard deviations for active tracks
    def active_keypoint_std_devs(self):
        return [np.array([np.sqrt(np.array([np.diag(keypoint_distribution.covariance) for keypoint_distribution in pose_3d_distribution.keypoint_distributions])) for pose_3d_distribution in active_track.pose_3d_distributions]) for active_track in self.active_tracks]

    # Return keypoint position standard deviations for active tracks
    def active_keypoint_position_std_devs(self):
        return [np.array([np.sqrt(np.array([np.diag(keypoint_distribution.covariance)[:3] for keypoint_distribution in pose_3d_distribution.keypoint_distributions])) for pose_3d_distribution in active_track.pose_3d_distributions]) for active_track in self.active_tracks]

    # Return keypoint velocity standard deviations for active tracks
    def active_keypoint_velocity_std_devs(self):
        return [np.array([np.sqrt(np.array([np.diag(keypoint_distribution.covariance)[3:] for keypoint_distribution in pose_3d_distribution.keypoint_distributions])) for pose_3d_distribution in active_track.pose_3d_distributions]) for active_track in self.active_tracks]

    # Construct and return a list of lists of 3D poses with keypoints equal to
    # the mean position keypoints of the distributions in the active tracks
    def active_to_poses_3d(self):
        poses_3d = Poses3D([[pose_3d_distribution.to_pose_3d() for pose_3d_distribution in active_track.pose_3d_distributions] for active_track in self.active_tracks])
        return poses_3d

    # Construct and return a dataframe with the data from the active
    # tracks
    def active_dataframe(self):
        num_active_tracks = len(self.active_tracks)
        dataframe_list = []
        for track_index in range(num_active_tracks):
            dataframe = self.active_tracks[track_index].dataframe()
            dataframe.insert(0, 'track_id', self.active_tracks[track_index].track_id)
            dataframe_list.append(dataframe)
        combined_dataframe = pd.concat(dataframe_list, ignore_index = True)
        combined_dataframe.insert(1, 'track_status', 'active')
        return combined_dataframe

    # Return timestamps for inactive tracks
    def inactive_timestamps(self):
        return [np.array([pose_3d_distribution.timestamp for pose_3d_distribution in inactive_track.pose_3d_distributions]) for inactive_track in self.inactive_tracks]

    # Return track IDs for inactive tracks
    def inactive_track_ids(self):
        return np.array([inactive_track.track_id for inactive_track in self.inactive_tracks])

    # Return number of missed observations for inactive tracks
    def inactive_num_missed_observations(self):
        return np.array([inactive_track.num_missed_observations for inactive_track in self.inactive_tracks])

    # Return keypoint means for inactive tracks
    def inactive_keypoint_means(self):
        return [np.array([[keypoint_distribution.mean for keypoint_distribution in pose_3d_distribution.keypoint_distributions] for pose_3d_distribution in inactive_track.pose_3d_distributions]) for inactive_track in self.inactive_tracks]

    # Return keypoint covariances for inactive tracks
    def inactive_keypoint_covariances(self):
        return [np.array([[keypoint_distribution.covariance for keypoint_distribution in pose_3d_distribution.keypoint_distributions] for pose_3d_distribution in inactive_track.pose_3d_distributions]) for inactive_track in self.inactive_tracks]

    # Return keypoint position means for inactive tracks
    def inactive_keypoint_position_means(self):
        return [np.array([[keypoint_distribution.mean[:3] for keypoint_distribution in pose_3d_distribution.keypoint_distributions] for pose_3d_distribution in inactive_track.pose_3d_distributions]) for inactive_track in self.inactive_tracks]

    # Return keypoint velocity means for inactive tracks
    def inactive_keypoint_velocity_means(self):
        return [np.array([[keypoint_distribution.mean[3:] for keypoint_distribution in pose_3d_distribution.keypoint_distributions] for pose_3d_distribution in inactive_track.pose_3d_distributions]) for inactive_track in self.inactive_tracks]

    # Return keypoint standard deviations for inactive tracks
    def inactive_keypoint_std_devs(self):
        return [np.array([np.sqrt(np.array([np.diag(keypoint_distribution.covariance) for keypoint_distribution in pose_3d_distribution.keypoint_distributions])) for pose_3d_distribution in inactive_track.pose_3d_distributions]) for inactive_track in self.inactive_tracks]

    # Return keypoint position standard deviations for inactive tracks
    def inactive_keypoint_position_std_devs(self):
        return [np.array([np.sqrt(np.array([np.diag(keypoint_distribution.covariance)[:3] for keypoint_distribution in pose_3d_distribution.keypoint_distributions])) for pose_3d_distribution in inactive_track.pose_3d_distributions]) for inactive_track in self.inactive_tracks]

    # Return keypoint velocity standard deviations for inactive tracks
    def inactive_keypoint_velocity_std_devs(self):
        return [np.array([np.sqrt(np.array([np.diag(keypoint_distribution.covariance)[3:] for keypoint_distribution in pose_3d_distribution.keypoint_distributions])) for pose_3d_distribution in inactive_track.pose_3d_distributions]) for inactive_track in self.inactive_tracks]

    # Construct and return a list of lists of 3D poses with the mean position
    # keypoints of the distributions in the inactive tracks
    def inactive_to_poses_3d(self):
        poses_3d = Poses3D([[pose_3d_distribution.to_pose_3d() for pose_3d_distribution in inactive_track.pose_3d_distributions] for inactive_track in self.inactive_tracks])
        return poses_3d

    # Construct and return a dataframe with the data from the inactive
    # tracks
    def inactive_dataframe(self):
        num_inactive_tracks = len(self.inactive_tracks)
        dataframe_list = []
        for track_index in range(num_inactive_tracks):
            dataframe = self.inactive_tracks[track_index].dataframe()
            dataframe.insert(0, 'track_id', self.inactive_tracks[track_index].track_id)
            dataframe_list.append(dataframe)
        combined_dataframe = pd.concat(dataframe_list, ignore_index = True)
        combined_dataframe.insert(1, 'track_status', 'inactive')
        return combined_dataframe

    # Construct and return a dataframe with the data from all tracks (active and
    # inactive)
    def dataframe(self):
        active_dataframe = self.active_dataframe()
        inactive_dataframe = self.inactive_dataframe()
        combined_dataframe = pd.concat(
            (inactive_dataframe, active_dataframe),
            ignore_index = True)
        return combined_dataframe

    # Return timestamps for last distributions in each active track
    def last_timestamps(self):
        return np.array([active_track.last().timestamp for active_track in self.active_tracks])

    # Return keypoint means for last distributions in each active track
    def last_keypoint_means(self):
        return np.array([[keypoint_distribution.mean for keypoint_distribution in active_track.last().keypoint_distributions] for active_track in self.active_tracks])

    # Return keypoint covariances for last distributions in each active track
    def last_keypoint_covariances(self):
        return np.array([[keypoint_distribution.covariance for keypoint_distribution in active_track.last().keypoint_distributions] for active_track in self.active_tracks])

    # Return keypoint position means for last distributions in each active track
    def last_keypoint_position_means(self):
        return np.array([[keypoint_distribution.mean[:3] for keypoint_distribution in active_track.last().keypoint_distributions] for active_track in self.active_tracks])

    # Return keypoint velocity means for last distributions in each active track
    def last_keypoint_velocity_means(self):
        return np.array([[keypoint_distribution.mean[3:] for keypoint_distribution in active_track.last().keypoint_distributions] for active_track in self.active_tracks])

    # Return keypoint standard deviations for last distributions in each active
    # track
    def last_keypoint_std_devs(self):
        return np.sqrt(np.array([[np.diag(keypoint_distribution.covariance) for keypoint_distribution in active_track.last().keypoint_distributions] for active_track in self.active_tracks]))

    # Return keypoint position standard deviations for last distributions in
    # each active track
    def last_keypoint_position_std_devs(self):
        return np.sqrt(np.array([[np.diag(keypoint_distribution.covariance)[:3] for keypoint_distribution in active_track.last().keypoint_distributions] for active_track in self.active_tracks]))

    # Return keypoint velocity standard deviations for last distributions in
    # each active track
    def last_keypoint_velocity_std_devs(self):
        return np.sqrt(np.array([[np.diag(keypoint_distribution.covariance)[3:] for keypoint_distribution in active_track.last().keypoint_distributions] for active_track in self.active_tracks]))

    # Construct and return a list of 3D poses with keypoints equal to the mean
    # position keypoints of the last distributions in the active tracks
    def last_to_poses_3d(self):
        poses_3d = [[active_track.last().to_pose_3d() for active_track in self.active_tracks]]
        return Poses3D(poses_3d)

    # Move a track from active to inactive
    def deactivate_track(
        self,
        track_index):
        self.inactive_tracks.append(self.active_tracks.pop(track_index))

    # Move one or more tracks to inactive
    def deactivate_tracks(
        self,
        track_indices):
        # We need to reverse-sort the track indices first so the first pop()
        # operations don't throw off the indices for the subsequent pop()
        # operations
        reverse_sorted_track_indices = sorted(track_indices, reverse = True)
        for track_index in reverse_sorted_track_indices:
            self.deactivate_track(track_index)

    # Add new tracks, using the initialization parameters stored in the class
    def add_new_tracks(
        self,
        num_new_tracks = 1):
        last_timestamps = self.last_timestamps()
        if last_timestamps.size > 0:
            if np.any(last_timestamps != None) and np.any(last_timestamps != last_timestamps[0]):
                raise ValueError('The last timestamps of the specified tracks are not all equal')
            timestamp = last_timestamps[0]
        else:
            timestamp = None
        for new_track_index in range(num_new_tracks):
            new_track = Pose3DTrack.initialize(
                self.pose_initialization_model,
                self.keypoint_model,
                timestamp = timestamp,
                track_id = self.next_track_id)
            self.next_track_id += 1
            self.active_tracks.append(new_track)


    # Given a keypoint model and a time interval, apply the model to all tracks.
    # User can specify time interval explicitly or method will attempt to infer
    # from ending timestamp. Time unit is specified above. In the future, we may
    # want to make underlying functions be able to handle time intervals with
    # units.
    def predict(
        self,
        delta_t = None,
        next_timestamp = None):
        for active_track in self.active_tracks:
            active_track.predict(
                delta_t,
                next_timestamp)

    # Given a keypoint model and an observation of a 3D pose (specified as a
    # Pose3D object), apply the model to selected tracks to calculate the
    # posterior 3D pose distribution for those tracks. Leave other tracks
    # unchanged
    def incorporate_observations(
        self,
        pose_3d_observations,
        selected_track_indices,
        selected_observation_indices):
        if pose_3d_observations.total_num_3d_poses() > 0:
            selected_track_indices = np.asarray(selected_track_indices)
            selected_observation_indices = np.asarray(selected_observation_indices)
            if selected_track_indices.ndim != 1:
                raise ValueError('Track indices must be a one-dimensional array-like object')
            if selected_observation_indices.ndim != 1:
                raise ValueError('Observation indices must be a one-dimensional array-like object')
            num_selected_tracks = selected_track_indices.shape[0]
            num_selected_observations = selected_observation_indices.shape[0]
            if num_selected_tracks != num_selected_observations:
                raise ValueError('Number of selected observations does not match number of selected tracks')
            if len(pose_3d_observations.pose_3d_list_list) != 1:
                raise ValueError('Observations must be a one-dimensional object')
            pose_3d_observations_list = pose_3d_observations.pose_3d_list_list[0]
            for index in range(num_selected_tracks):
                track_index = selected_track_indices[index]
                observation_index = selected_observation_indices[index]
                self.active_tracks[track_index].incorporate_observation(
                    pose_3d_observations_list[observation_index])

    # Given a keypoint model and an observation of a 3D pose (specified as a
    # Pose3D object), apply the model to predict the tracks at the time of the
    # observations and then incorporate observations for selected tracks to
    # calculate the posterior 3D pose distribution for those tracks. Leave other
    # tracks unchanged
    def predict_and_incorporate_observations(
        self,
        pose_3d_observations,
        selected_track_indices,
        selected_observation_indices):
        if pose_3d_observations.total_num_3d_poses() > 0:
            selected_track_indices = np.asarray(selected_track_indices)
            selected_observation_indices = np.asarray(selected_observation_indices)
            if len(pose_3d_observations.pose_3d_list_list) != 1:
                raise ValueError('Observations must be a one-dimensional object')
            observation_timestamps = np.array([pose_3d_observation.timestamp for pose_3d_observation in pose_3d_observations.pose_3d_list_list[0]])
            if np.any(observation_timestamps != observation_timestamps[0]):
                raise ValueError('All observations must have timestamps and all timestamps must be equal')
            observation_timestamp = observation_timestamps[0]
            self.predict(
                next_timestamp = observation_timestamp)
            self.incorporate_observations(
                pose_3d_observations,
                selected_track_indices,
                selected_observation_indices)

    # Given a set of observations, predict the 3D pose distributions of the
    # active tracks at the time of the observations, compare the observations to
    # these predictions to calculate a best-guess association between
    # observations and tracks, and update the tracks accordingly
    def update(
        self,
        pose_3d_observations):
        if pose_3d_observations.total_num_3d_poses() > 0:
            if len(pose_3d_observations.pose_3d_list_list) != 1:
                raise ValueError('Observations must be a one-dimensional object')
            pose_3d_observations_list = pose_3d_observations.pose_3d_list_list[0]
            timestamps = np.array([pose_3d_observation.timestamp for pose_3d_observation in pose_3d_observations_list])
            if np.any(timestamps != timestamps[0]):
                raise ValueError('Timestamps on observations are missing or not equal to each other')
            timestamp = timestamps[0]
            self.predict(next_timestamp = timestamp)
            cost_matrix = self.cost_matrix(pose_3d_observations)
            matched_track_indices, matched_observation_indices = scipy.optimize.linear_sum_assignment(cost_matrix)
            matched_track_indices = matched_track_indices.tolist()
            matched_observation_indices = matched_observation_indices.tolist()
            num_matched_indices = len(matched_track_indices)
            for index in sorted(range(num_matched_indices), reverse = True):
                if cost_matrix[matched_track_indices[index], matched_observation_indices[index]] > self.pose_tracking_model.cost_threshold:
                    del matched_track_indices[index]
                    del matched_observation_indices[index]
            unmatched_track_indices = list(set(range(self.num_active_tracks())) - set(matched_track_indices))
            unmatched_observation_indices = list(set(range(len(pose_3d_observations_list))) - set(matched_observation_indices))
            self.incorporate_observations(
                pose_3d_observations,
                matched_track_indices,
                matched_observation_indices)
            for unmatched_track_index in unmatched_track_indices:
                self.active_tracks[unmatched_track_index].num_missed_observations += 1
            reverse_sorted_active_track_indices = sorted(range(len(self.active_tracks)), reverse = True)
            for active_track_index in reverse_sorted_active_track_indices:
                if self.active_tracks[active_track_index].num_missed_observations >= self.pose_tracking_model.num_missed_observations_threshold:
                    self.deactivate_track(active_track_index)
            for unmatched_observation_index in unmatched_observation_indices:
                self.add_new_tracks(num_new_tracks = 1)
                self.active_tracks[-1].incorporate_observation(
                    pose_3d_observations_list[unmatched_observation_index])

    # Given a keypoint model and a set of 3D pose observations (specified as a
    # list of Pose3D objects), calculate the cost matrix between the last 3D
    # pose distributions in the active tracks and the observations
    def cost_matrix(
        self,
        pose_3d_observations):
        if len(pose_3d_observations.pose_3d_list_list) != 1:
            raise ValueError('Observations must be a one-dimensional object')
        pose_3d_observations_list = pose_3d_observations.pose_3d_list_list[0]
        num_active_tracks = self.num_active_tracks()
        num_observations = len(pose_3d_observations_list)
        cost_matrix = np.zeros((num_active_tracks, num_observations))
        for active_track_index in range(num_active_tracks):
            active_track = self.active_tracks[active_track_index]
            for observation_index in range(num_observations):
                observation = pose_3d_observations_list[observation_index]
                cost_matrix[active_track_index, observation_index] = active_track.observation_mahalanobis_distance(observation)
        return cost_matrix

# Check that all of the last timestamps in a list of 3D pose tracks are equal
def check_last_timestamps(pose_3d_tracks):
    last_timestamps = np.array([pose_3d_track.last().timestamp for pose_3d_track in pose_3d_tracks])
    if np.any(last_timestamps != None) and np.any(last_timestamps != last_timestamps[0]):
        raise ValueError('The last timestamps of the specified tracks are not all equal')


# Calculate the reprojection error between two sets of corresponding 2D points.
# Used above in evaluating potential 3D poses
def rms_projection_error(
    image_points,
    image_points_reconstructed):
    image_points = np.asarray(image_points)
    image_points_reconstructed = np.asarray(image_points_reconstructed)
    if image_points.size == 0 or image_points_reconstructed.size == 0:
        return np.nan
    image_points = image_points.reshape((-1,2))
    image_points_reconstructed = image_points_reconstructed.reshape((-1,2))
    if image_points.shape != image_points_reconstructed.shape:
        raise ValueError('Sets of image points do not appear to be the same shape')
    rms_error = np.sqrt(np.sum(np.square(image_points_reconstructed - image_points))/image_points.shape[0])
    return rms_error

# For two sets of pose keypoints, extract the intersection of their valid
# keypoints and returns a mask which encodes where these keypoints belong in the
# total set
def extract_common_keypoints(
    pose_a,
    pose_b):
    common_keypoint_positions_mask = np.logical_and(
        pose_a.valid_keypoints,
        pose_b.valid_keypoints)
    common_keypoints_a = pose_a.keypoints[common_keypoint_positions_mask]
    common_keypoints_b = pose_b.keypoints[common_keypoint_positions_mask]
    return common_keypoints_a, common_keypoints_b, common_keypoint_positions_mask

# Inverse of the above. For a set of valid keypoints and a mask, repopulates the
# points back into the total set of keypoints
def restore_all_keypoints(
    common_keypoints,
    common_keypoint_positions_mask):
    common_keypoints = np.asarray(common_keypoints)
    common_keypoint_positions_mask = np.asarray(common_keypoint_positions_mask)
    all_keypoints_dims = [len(common_keypoint_positions_mask)] + list(common_keypoints.shape[1:])
    all_keypoints = np.full(all_keypoints_dims, np.nan)
    all_keypoints[common_keypoint_positions_mask] = common_keypoints
    return all_keypoints
