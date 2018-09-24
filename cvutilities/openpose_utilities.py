import cvutilities.camera_utilities
import cvutilities.datetime_utilities
import boto3
import networkx as nx # We use graph structures to hold the 3D pose data
import numpy as np
import matplotlib.pyplot as plt
import json

# For now, the Wildflower-specific functionality is intermingled with the more
# general S3 functionality. We should probably separate these at some point. For
# the S3 functions to work, the environment must include AWS_ACCESS_KEY_ID and
# AWS_SECRET_ACCESS_KEY variables and that access key must have read permissions
# for the relevant buckets. You can set these environment variables manually or
# by using the AWS CLI.
classroom_data_wildflower_s3_bucket_name = 'wf-classroom-data'
pose_2d_data_wildflower_s3_directory_name = '2D-pose'

# Generate the S3 object name for a 2D pose file from a classroom name, a camera
# name, and a date-time (according to our current naming conventions)
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

# Generate date and time strings as they appear in our S3 object names from a
# date-time (according to our current conventions)
def generate_wildflower_s3_datetime_strings(
    datetime):
    datetime_native_utc_naive = cvutilities.datetime_utilities.convert_to_native_utc_naive(datetime)
    date_string = datetime_native_utc_naive.strftime('%Y-%m-%d')
    time_string = datetime_native_utc_naive.strftime('%H-%M-%S')
    return date_string, time_string

# For now, the OpenPose-specific functionality is intermingled with the more
# general pose analysis functionality. We should probably separate these at some
# point. The parameters below correspond to the OpenPose output we've been
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

# Class to hold the data for a single 2D pose
class Pose2D:
    def __init__(self, keypoints, confidence_scores, valid_keypoints):
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

    # Pull the pose data from a dictionary with the same structure as the
    # correponding OpenPose output JSON string
    @classmethod
    def from_openpose_person_json_data(cls, json_data):
        keypoint_vector = np.asarray(json_data[openpose_keypoint_vector_name])
        if keypoint_vector.size != num_body_parts*3:
            raise ValueError('OpenPose keypoint vector does not appear to be of size {}*3'.format(num_body_parts))
        keypoint_array = keypoint_vector.reshape((num_body_parts, 3))
        keypoints = keypoint_array[:, :2]
        confidence_scores = keypoint_array[:, 2]
        valid_keypoints = np.not_equal(confidence_scores, 0.0)
        return cls(keypoints, confidence_scores, valid_keypoints)

    # Pull the pose data from an OpenPose output JSON string
    @classmethod
    def from_openpose_person_json_string(cls, json_string):
        json_data = json.loads(json_string)
        return cls.from_openpose_person_json_data(json_data)

    # Draw the pose onto a chart with the dimensions of the origin image. We
    # separate this from the plotting function below because we might want to
    # draw several poses or other elements before formatting and showing the
    # chart.
    def draw(
        self,
        pose_tag = None):
        all_points = self.keypoints
        valid_keypoints = self.valid_keypoints
        plottable_points = all_points[valid_keypoints]
        centroid = np.mean(plottable_points, 0)
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
        plt.text(centroid[0], centroid[1], pose_tag)

    # Plot a pose onto a chart with the dimensions of the origin image. Calls
    # the drawing function above, adds formating, and shows the plot.
    def plot(
        self,
        pose_tag = None,
        image_size=[1296, 972]):
        self.draw_pose_2d(pose_tag)
        cvutilities.camera_utilities.format_2d_image_plot(image_size)
        plt.show()

# Class to hold the data from a collection of 2D poses corresponding to a single
# camera image
class Poses2DCamera:
    def __init__(self, poses):
        self.poses = poses
        self.num_poses = len(poses)

    # Pull the pose data from a dictionary with the same structure as the
    # correponding OpenPose output JSON file
    @classmethod
    def from_openpose_output_json_data(cls, json_data):
        people_json_data = json_data[openpose_people_list_name]
        poses = [Pose2D.from_openpose_person_json_data(person_json_data) for person_json_data in people_json_data]
        return cls(poses)

    # Pull the pose data from a string containing the contents of an OpenPose
    # output JSON file
    @classmethod
    def from_openpose_output_json_string(cls, json_string):
        json_data = json.loads(json_string)
        return cls.from_openpose_output_json_data(json_data)

    # Pull the pose data from a local OpenPose output JSON file
    @classmethod
    def from_openpose_output_json_file(cls, json_file_path):
        with open(json_file_path) as json_file:
            json_data = json.load(json_file)
        return cls.from_openpose_output_json_data(json_data)

    # Pull the pose data from an OpenPose output JSON file stored in a specified
    # location on S3
    @classmethod
    def from_openpose_output_s3_object(cls, s3_bucket_name, s3_object_name):
        s3_object = boto3.resource('s3').Object(s3_bucket_name, s3_object_name)
        s3_object_content = s3_object.get()['Body'].read().decode('utf-8')
        json_data = json.loads(s3_object_content)
        return cls.from_openpose_output_json_data(json_data)

    # Pull the pose data from an OpenPose output JSON file stored on S3 and
    # specified by classroom name, camera name, and date-time
    @classmethod
    def from_openpose_output_wildflower_s3(
        cls,
        classroom_name,
        camera_name,
        datetime):
        s3_bucket_name = classroom_data_wildflower_s3_bucket_name
        s3_object_name = generate_pose_2d_wildflower_s3_object_name(
            classroom_name,
            camera_name,
            datetime)
        return cls.from_openpose_output_s3_object(s3_bucket_name, s3_object_name)

    # Draw the poses onto a chart with the dimensions of the origin image. We
    # separate this from the plotting function below because we might want to
    # draw other elements before formatting and showing the chart
    def draw(
        self,
        pose_tags = None):
        num_poses = self.num_poses
        if pose_tags is None:
            pose_tags = range(num_poses)
        for pose_index in range(num_poses):
            self.poses[pose_index].draw(pose_tags[pose_index])

    # Plot the poses onto a chart with the dimensions of the origin image. Calls
    # the drawing function above, adds formating, and shows the plot
    def plot(
        self,
        pose_tags = None,
        image_size=[1296, 972]):
        self.draw(pose_tags)
        cvutilities.camera_utilities.format_2d_image_plot(image_size)
        plt.show()

# Class to hold the data from a collection of 2D poses from multiple cameras at
# a single timestep
class Poses2DTimestep:
    def __init__(self, cameras):
        self.cameras = cameras
        self.num_cameras = len(cameras)

    # Pull the pose data from a set of OpenPose output JSON files stored on S3
    # and specified by classroom name, a list of camera names, and date-time
    @classmethod
    def from_openpose_timestep_wildflower_s3(
        cls,
        classroom_name,
        camera_names,
        datetime):
        s3_bucket_name = classroom_data_wildflower_s3_bucket_name
        cameras = []
        for camera_name in camera_names:
            s3_object_name = generate_pose_2d_wildflower_s3_object_name(
                classroom_name,
                camera_name,
                datetime)
            cameras.append(Poses2DCamera.from_openpose_output_s3_object(s3_bucket_name, s3_object_name))
        return cls(cameras)

    # Plot the poses onto a set of charts, one for each source camera view.
    def plot(
        self,
        pose_tags = None,
        image_size=[1296, 972]):
        num_cameras = self.num_cameras
        for camera_index in range(num_cameras):
            if pose_tags is None:
                pose_tags_single_camera = None
            else:
                pose_tags_single_camera = pose_tags[camera_index]
            self.cameras[camera_index].plot(
                pose_tags_single_camera,
                image_size)

# Class to hold the data for a single 3D pose
class Pose3D:
    def __init__(self, keypoints, valid_keypoints, projection_error=None):
        keypoints = np.asarray(keypoints)
        valid_keypoints = np.asarray(valid_keypoints, dtype = np.bool_)
        projection_error = np.asarray(projection_error)
        if keypoints.size != num_body_parts*3:
            raise ValueError('Keypoints array does not appear to be of size {}*3'.format(num_body_parts))
        if valid_keypoints.size != num_body_parts:
            raise ValueError('Valid keypoints vector does not appear to be of size {}'.format(num_body_parts))
        if projection_error.size != 1:
            raise ValueError('Projection error does not appear to be a scalar'.format(num_body_parts))
        keypoints = keypoints.reshape((num_body_parts, 3))
        valid_keypoints = valid_keypoints.reshape(num_body_parts)
        projection_error = np.asscalar(projection_error)
        self.keypoints = keypoints
        self.valid_keypoints = valid_keypoints
        self.projection_error = projection_error

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
        return cls(keypoints, common_keypoint_positions_mask, projection_error)

    # Draw the pose onto a chart representing a top-down view of the room. We
    # separate this from the plotting function below because we might want to
    # draw several poses or other elements before formatting and showing the
    # chart
    def draw_topdown(
        self,
        pose_tag = None):
        plottable_points = self.keypoints[self.valid_keypoints]
        centroid = np.mean(plottable_points[:, :2], 0)
        cvutilities.camera_utilities.draw_3d_object_points_topdown(plottable_points)
        if pose_tag is not None:
            plt.text(centroid[0], centroid[1], pose_tag)

    # Plot a pose onto a chart representing a top-down view of the room. Calls
    # the drawing function above, adds formating, and shows the plot
    def plot_topdown(
        self,
        pose_tag = None,
        room_corners = None):
        self.draw_pose_3d_topdown(pose_tag)
        cvutilities.camera_utilities.format_3d_topdown_plot(room_corners)
        plt.show()

# Class to hold the data for a colection of 3D poses reconstructed from 2D poses
# across multiple cameras at a single timestep
class Poses3D:
    def __init__(
        self,
        pose_graph,
        num_cameras_source_images,
        num_2d_poses_source_images):
        self.pose_graph = pose_graph
        self.num_cameras_source_images = num_cameras_source_images
        self.num_2d_poses_source_images = num_2d_poses_source_images

    # Calculate all possible 3D poses at a single time step (from every pair of
    # 2D poses across every pair of cameras)
    @classmethod
    def from_poses_2d_timestep(
        cls,
        poses_2d,
        cameras):
        pose_graph = nx.Graph()
        num_cameras_source_images = poses_2d.num_cameras
        num_2d_poses_source_images = np.zeros(num_cameras_source_images, dtype=int)
        for camera_index_a in range(num_cameras_source_images - 1):
            for camera_index_b in range(camera_index_a + 1, num_cameras_source_images):
                num_poses_a = poses_2d.cameras[camera_index_a].num_poses
                num_poses_b = poses_2d.cameras[camera_index_b].num_poses
                num_2d_poses_source_images[camera_index_a] = num_poses_a
                num_2d_poses_source_images[camera_index_b] = num_poses_b
                for pose_index_a in range(num_poses_a):
                    for pose_index_b in range(num_poses_b):
                        pose_3d = Pose3D.from_poses_2d(
                            poses_2d.cameras[camera_index_a].poses[pose_index_a],
                            poses_2d.cameras[camera_index_b].poses[pose_index_b],
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
        return cls(pose_graph, num_cameras_source_images, num_2d_poses_source_images)

    # Return the number of 3D poses (edges) in the collection
    def num_3d_poses(self):
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
    def poses(self):
        return [edge[2]['pose'] for edge in list(self.pose_graph.edges.data())]

    # Return the keypoints for all of the 3D poses in the collection
    def keypoints(self):
        return np.array([edge[2]['pose'].keypoints for edge in list(self.pose_graph.edges.data())])

    # Return the valid keypoints Boolean vector for all of the 3D poses in the
    # collection.
    def valid_keypoints(self):
        return np.array([edge[2]['pose'].valid_keypoints for edge in list(self.pose_graph.edges.data())])

    # Return the projection errors for all of the 3D poses in the collection.
    def projection_errors(self):
        return np.array([edge[2]['pose'].projection_error for edge in list(self.pose_graph.edges.data())])

    # Draw the graph representing all of the 3D poses in the collection (2D
    # poses as nodes, 3D poses as edges)
    def draw_graph(self):
        nx.draw(self.pose_graph, with_labels=True, font_weight='bold')

    # Scan through all of the 3D poses in a collection and pull out a set of
    # best matches, one for each person in the room
    def extract_matched_poses(
        self,
        projection_error_threshold = 15.0):
        # First, for each camera pair, we pull out the best match for each 2D
        # pose across that pair, with the contraint that (1) If pose A is the
        # best match for pose B then pose B must be the best match for pose A,
        # and (2) The reprojection error has to be below a threshold. For now,
        # we initialize a new empty graph and copy selected edges into it. We
        # should really do this either by creating a copy of the original graph
        # and deleting the edges we don't want or by tracking pointers back to
        # the original graph
        pruned_graph = nx.Graph()
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
                            pruned_graph.add_edge(
                                (camera_index_a, pose_index_a),
                                (camera_index_b, pose_index_b),
                                pose=self.pose_graph[(camera_index_a, pose_index_a)][(camera_index_b, pose_index_b)]['pose'])
        # Second, for each connected subgraph (which now represents a set of
        # poses connected across camera pairs that ought to be the same person),
        # we extract the match with the lowest reprojection error (we could
        # average instead). For now, we make a copy of each subgraph of the
        # pruned graph, select the best edge from each subgraph, and copy that
        # best edge into a new graph. We should really do this either by
        # deleting all edges from the pruned graph other than the best one for
        # each subgraph or by tracking pointers back to the pruned graph
        matched_poses_graph = nx.Graph()
        subgraphs_list = [pruned_graph.subgraph(component).copy() for component in nx.connected_components(pruned_graph)]
        for subgraph_index in range(len(subgraphs_list)):
            if nx.number_of_edges(subgraphs_list[subgraph_index]) > 0:
                best_edge = sorted(subgraphs_list[subgraph_index].edges.data(), key = lambda x: x[2]['pose'].projection_error)[0]
                matched_poses_graph.add_edge(best_edge[0], best_edge[1], pose = best_edge[2]['pose'])
        matched_poses = self.__class__(
            matched_poses_graph,
            self.num_cameras_source_images,
            self.num_2d_poses_source_images)
        return matched_poses

    # Draw the poses onto a chart representing a top-down view of the room. We
    # separate this from the plotting function below because we might want to
    # draw other elements before formatting and showing the chart
    def draw_topdown(
        self,
        pose_tags_2d = None):
        num_poses = len(self.poses())
        pose_indices = self.pose_indices()
        if pose_tags_2d is None:
            pose_tags_3d = range(num_poses)
        else:
            pose_tags_3d = []
            for match_index in range(pose_indices.shape[0]):
                pose_tags_3d.append('{},{}'.format(
                    pose_tags_2d[pose_indices[match_index, 0, 0]][pose_indices[match_index, 0, 1]],
                    pose_tags_2d[pose_indices[match_index, 1, 0]][pose_indices[match_index, 1, 1]]))
        for pose_index in range(num_poses):
            self.poses()[pose_index].draw_topdown(pose_tags_3d[pose_index])

    # Plot the poses onto a chart representing a top-down view of the room.
    # Calls the drawing function above, adds formating, and shows the plot
    def plot_topdown(
        self,
        pose_tags_2d = None,
        room_corners = None):
        self.draw_topdown(pose_tags_2d)
        cvutilities.camera_utilities.format_3d_topdown_plot(room_corners)
        plt.show()

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
