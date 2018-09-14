import cvutilities.camera_utilities
import cvutilities.datetime_utilities
import boto3
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import json

# For now the Wildflower-specific functionality is intermingled with the more
# general S3 functionality. We should probably separate these at some point.
classroom_data_wildflower_s3_bucket_name = 'wf-classroom-data'
pose_2d_data_wildflower_s3_directory_name = '2D-pose'

def generate_pose_2d_frame_wildflower_s3_object_name(
    classroom_name,
    camera_name,
    datetime):
    date_string, time_string = generate_wildflower_s3_datetime_strings(datetime)
    pose_2d_frame_wildflower_s3_object_name = 'camera-{}/{}/{}/{}/still_{}-{}_keypoints.json'.format(
        classroom_name,
        pose_2d_data_wildflower_s3_directory_name,
        date_string,
        camera_name,
        date_string,
        time_string)
    return pose_2d_frame_wildflower_s3_object_name

def generate_wildflower_s3_datetime_strings(
    datetime):
    datetime_native_utc_naive = cvutilities.datetime_utilities.convert_to_native_utc_naive(datetime)
    date_string = datetime_native_utc_naive.strftime('%Y-%m-%d')
    time_string = datetime_native_utc_naive.strftime('%H-%M-%S')
    return date_string, time_string

# For now, the OpenPose-specific functionality is intermingled with the more
# general pose analysis functionality. We should probably separate these at some
# point. The parameters below correspond to the OpenPose output we've been
# generating (I think) but the newest OpenPose version changes these (more body
# parts, 'pose_keypoints_2d' instead of 'pose_keypoints', etc.)
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

class Pose2DPerson:
    def __init__(self, pose_keypoints, pose_confidence_scores, valid_keypoints):
        pose_keypoints = np.asarray(pose_keypoints)
        pose_confidence_scores = np.asarray(pose_confidence_scores)
        valid_keypoints = np.asarray(valid_keypoints, dtype = np.bool_)
        if pose_keypoints.size != num_body_parts*2:
            raise ValueError('Pose keypoints array does not appear to be of size {}*2'.format(num_body_parts))
        if pose_confidence_scores.size != num_body_parts:
            raise ValueError('Pose confidence scores vector does not appear to be of size {}'.format(num_body_parts))
        if valid_keypoints.size != num_body_parts:
            raise ValueError('Valid keypoints vector does not appear to be of size {}'.format(num_body_parts))
        pose_keypoints = pose_keypoints.reshape((num_body_parts, 2))
        pose_confidence_scores = pose_confidence_scores.reshape(num_body_parts)
        valid_keypoints = valid_keypoints.reshape(num_body_parts)
        self.pose_keypoints = pose_keypoints
        self.pose_confidence_scores = pose_confidence_scores
        self.valid_keypoints = valid_keypoints

    @classmethod
    def from_openpose_person_json_data(cls, json_data):
        keypoint_vector = np.asarray(json_data[openpose_keypoint_vector_name])
        if keypoint_vector.size != num_body_parts*3:
            raise ValueError('OpenPose keypoint vector does not appear to be of size {}*3'.format(num_body_parts))
        keypoint_array = keypoint_vector.reshape((num_body_parts, 3))
        pose_keypoints = keypoint_array[:, :2]
        pose_confidence_scores = keypoint_array[:, 2]
        valid_keypoints = np.not_equal(pose_confidence_scores, 0.0)
        return cls(pose_keypoints, pose_confidence_scores, valid_keypoints)

    @classmethod
    def from_openpose_person_json_string(cls, json_string):
        json_data = json.loads(json_string)
        return cls.from_openpose_person_json_data(json_data)

class Pose2DFrame:
    def __init__(self, poses):
        self.poses = poses
        self.num_poses = len(poses)

    @classmethod
    def from_openpose_frame_json_data(cls, json_data):
        people_json_data = json_data[openpose_people_list_name]
        poses = [Pose2DPerson.from_openpose_person_json_data(person_json_data) for person_json_data in people_json_data]
        return cls(poses)

    @classmethod
    def from_openpose_frame_json_string(cls, json_string):
        json_data = json.loads(json_string)
        return cls.from_openpose_frame_json_data(json_data)

    @classmethod
    def from_openpose_frame_json_file(cls, json_file_path):
        with open(json_file_path) as json_file:
            json_data = json.load(json_file)
        return cls.from_openpose_frame_json_data(json_data)

    @classmethod
    def from_openpose_frame_s3_object(cls, s3_bucket_name, s3_object_name):
        s3_object = boto3.resource('s3').Object(s3_bucket_name, s3_object_name)
        s3_object_content = s3_object.get()['Body'].read().decode('utf-8')
        json_data = json.loads(s3_object_content)
        return cls.from_openpose_frame_json_data(json_data)

    @classmethod
    def from_openpose_frame_wildflower_s3(
        cls,
        classroom_name,
        camera_name,
        datetime):
        s3_bucket_name = classroom_data_wildflower_s3_bucket_name
        s3_object_name = generate_pose_2d_frame_wildflower_s3_object_name(
            classroom_name,
            camera_name,
            datetime)
        return cls.from_openpose_frame_s3_object(s3_bucket_name, s3_object_name)

class Pose2DRoom:
    def __init__(self, frames):
        self.frames = frames
        self.num_cameras = len(frames)

    @classmethod
    def from_openpose_room_wildflower_s3(
        cls,
        classroom_name,
        camera_names,
        datetime):
        s3_bucket_name = classroom_data_wildflower_s3_bucket_name
        frames = []
        for camera_name in camera_names:
            s3_object_name = generate_pose_2d_frame_wildflower_s3_object_name(
                classroom_name,
                camera_name,
                datetime)
            frames.append(Pose2DFrame.from_openpose_frame_s3_object(s3_bucket_name, s3_object_name))
        return cls(frames)

### OLDER CODE ###

def extract_keypoint_positions_UNUSED(openpose_json_data_single_person, openpose_keypoint_vector_name='pose_keypoints'):
    keypoint_list = openpose_json_data_single_person[openpose_keypoint_vector_name]
    keypoint_positions = np.array(keypoint_list).reshape((-1, 3))[:,:2]
    return keypoint_positions

def extract_keypoint_confidence_scores_UNUSED(openpose_json_data_single_person, openpose_keypoint_vector_name='pose_keypoints'):
    keypoint_list = openpose_json_data_single_person[openpose_keypoint_vector_name]
    keypoint_confidence_scores = np.array(keypoint_list).reshape((-1, 3))[:,2]
    return keypoint_confidence_scores

def extract_keypoints_UNUSED(openpose_json_data_single_person, openpose_keypoint_vector_name='pose_keypoints'):
    keypoint_positions = extract_keypoint_positions(openpose_json_data_single_person, openpose_keypoint_vector_name)
    keypoint_confidence_scores = extract_keypoint_confidence_scores(openpose_json_data_single_person, openpose_keypoint_vector_name)
    return keypoint_positions, keypoint_confidence_scores

def fetch_openpose_data_from_s3_single_camera_UNUSED(
    classroom_name,
    datetime,
    camera_name,
    s3_bucket_name = 'wf-classroom-data',
    pose_directory_name = '2D-pose'):
    date_string, time_string = generate_wildflower_s3_datetime_strings(datetime)
    keypoints_filename = '%s/%s/%s/%s/still_%s-%s_keypoints.json' % (
        classroom_name,
        pose_directory_name,
        date_string,
        camera_name,
        date_string,
        time_string)
    content_object = boto3.resource('s3').Object(s3_bucket_name, keypoints_filename)
    file_content = content_object.get()['Body'].read().decode('utf-8')
    json_content = json.loads(file_content)
    openpose_data_single_camera = []
    for openpose_json_data_single_person in json_content['people']:
        openpose_data_single_camera.append({
            'keypoint_positions': extract_keypoint_positions(openpose_json_data_single_person),
            'keypoint_confidence_scores': extract_keypoint_confidence_scores(openpose_json_data_single_person)})
    return openpose_data_single_camera

def fetch_openpose_data_from_s3_multiple_cameras_UNUSED(
    classroom_name,
    datetime,
    camera_names,
    s3_bucket_name = 'wf-classroom-data',
    pose_directory_name = '2D-pose'):
    openpose_data_multiple_cameras = []
    for camera_name in camera_names:
        openpose_data_multiple_cameras.append(
            fetch_openpose_data_from_s3_single_camera(
                classroom_name,
                datetime,
                camera_name,
                s3_bucket_name,
                pose_directory_name))
    return openpose_data_multiple_cameras

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

def extract_common_keypoints(
    pose_a,
    pose_b):
    common_keypoint_positions_mask = np.logical_and(
        pose_a.valid_keypoints,
        pose_b.valid_keypoints)
    image_points_a = pose_a.pose_keypoints[common_keypoint_positions_mask]
    image_points_b = pose_b.pose_keypoints[common_keypoint_positions_mask]
    return image_points_a, image_points_b, common_keypoint_positions_mask

def populate_array(
    partial_array,
    mask):
    partial_array = np.asarray(partial_array)
    mask = np.asarray(mask)
    array_dims = [len(mask)] + list(partial_array.shape[1:])
    array = np.full(array_dims, np.nan)
    array[mask] = partial_array
    return array

def calculate_pose_3d(
    pose_a,
    pose_b,
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
        pose_a,
        pose_b)
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
    rms_projection_error_a = rms_projection_error(
        image_points_a,
        image_points_a_reconstructed)
    rms_projection_error_b = rms_projection_error(
        image_points_b,
        image_points_b_reconstructed)
    object_points = object_points.reshape((-1, 3))
    pose_3d = populate_array(
        object_points,
        common_keypoint_positions_mask)
    return pose_3d, rms_projection_error_a, rms_projection_error_b

def calculate_poses_3d_camera_pair(
    poses_camera_a,
    poses_camera_b,
    rotation_vector_a,
    translation_vector_a,
    rotation_vector_b,
    translation_vector_b,
    camera_matrix,
    distortion_coefficients = 0):
    num_people_a = poses_camera_a.num_poses
    num_people_b = poses_camera_b.num_poses
    poses_3d = np.full((num_people_a, num_people_b, num_body_parts, 3), np.nan)
    projection_errors = np.full((num_people_a, num_people_b), np.nan)
    # match_mask = np.full((num_people_a, num_people_b), False)
    for person_index_a in range(num_people_a):
        for person_index_b in range(num_people_b):
            pose, projection_error_a, projection_error_b = calculate_pose_3d(
                poses_camera_a.poses[person_index_a],
                poses_camera_b.poses[person_index_b],
                rotation_vector_a,
                translation_vector_a,
                rotation_vector_b,
                translation_vector_b,
                camera_matrix,
                distortion_coefficients)[:3]
            poses_3d[person_index_a, person_index_b] = pose
            if np.isnan(projection_error_a) or np.isnan(projection_error_b):
                projection_errors[person_index_a, person_index_b] = np.nan
            else:
                projection_errors[person_index_a, person_index_b] = max(
                    projection_error_a,
                    projection_error_b)
    return poses_3d, projection_errors

def extract_matched_poses_3d_camera_pair(
    poses_3d,
    projection_errors,
    projection_error_threshold = 15.0):
    matches = np.full_like(projection_errors, False, dtype='bool_')
    for person_index_a in range(projection_errors.shape[0]):
        for person_index_b in range(projection_errors.shape[1]):
            matches[person_index_a, person_index_b] = (
                not np.all(np.isnan(projection_errors[person_index_a, :])) and
                np.nanargmin(projection_errors[person_index_a, :]) == person_index_b and
                not np.all(np.isnan(projection_errors[:, person_index_b])) and
                np.nanargmin(projection_errors[:, person_index_b]) == person_index_a and
                projection_errors[person_index_a, person_index_b] < projection_error_threshold)
    matched_poses_3d = poses_3d[matches]
    matched_projection_errors = projection_errors[matches]
    match_indices = np.vstack(np.where(matches)).T
    return matched_poses_3d, matched_projection_errors, match_indices

def calculate_poses_3d_multiple_cameras(
    openpose_data_multiple_cameras,
    camera_calibration_data_multiple_cameras):
    num_cameras = openpose_data_multiple_cameras.num_cameras
    poses_3d_multiple_cameras = [[None]*num_cameras for _ in range(num_cameras)]
    projection_errors_multiple_cameras = [[None]*num_cameras for _ in range(num_cameras)]
    for camera_index_a in range(num_cameras):
        for camera_index_b in range(num_cameras):
            if camera_index_b > camera_index_a:
                poses_3d_multiple_cameras[camera_index_a][camera_index_b], projection_errors_multiple_cameras[camera_index_a][camera_index_b] = calculate_poses_3d_camera_pair(
                    openpose_data_multiple_cameras.frames[camera_index_a],
                    openpose_data_multiple_cameras.frames[camera_index_b],
                    camera_calibration_data_multiple_cameras[camera_index_a]['rotation_vector'],
                    camera_calibration_data_multiple_cameras[camera_index_a]['translation_vector'],
                    camera_calibration_data_multiple_cameras[camera_index_b]['rotation_vector'],
                    camera_calibration_data_multiple_cameras[camera_index_b]['translation_vector'],
                    camera_calibration_data_multiple_cameras[camera_index_a]['camera_matrix'],
                    camera_calibration_data_multiple_cameras[camera_index_a]['distortion_coefficients'])
    return poses_3d_multiple_cameras, projection_errors_multiple_cameras

def extract_matched_poses_3d_multiple_cameras(
    poses_3d_multiple_cameras,
    projection_errors_multiple_cameras,
    projection_error_threshold = 15.0):
    person_list=[]
    num_cameras = len(projection_errors_multiple_cameras)
    for camera_index in range(num_cameras):
        if camera_index == 0:
            num_people = projection_errors_multiple_cameras[0][1].shape[0]
        else:
            num_people = projection_errors_multiple_cameras[0][camera_index].shape[1]
        for person_index in range(num_people):
            person_list.append((camera_index, person_index))
    person_graph = nx.Graph()
    person_graph.add_nodes_from(person_list)
    for camera_index_a in range(num_cameras):
        for camera_index_b in range(camera_index_a + 1, num_cameras):
            matched_poses_3d, matched_projection_errors, match_indices = extract_matched_poses_3d_camera_pair(
                poses_3d_multiple_cameras[camera_index_a][camera_index_b],
                projection_errors_multiple_cameras[camera_index_a][camera_index_b],
                projection_error_threshold)
            for match_index in range(match_indices.shape[0]):
                person_index_a = match_indices[match_index, 0]
                person_index_b = match_indices[match_index, 1]
                person_graph.add_edge(
                    (camera_index_a, person_index_a),
                    (camera_index_b, person_index_b),
                    pose_3d = poses_3d_multiple_cameras[camera_index_a][camera_index_b][person_index_a, person_index_b],
                    projection_error=projection_errors_multiple_cameras[camera_index_a][camera_index_b][person_index_a, person_index_b])
    subgraphs_list = [person_graph.subgraph(component).copy() for component in nx.connected_components(person_graph)]
    matched_poses_3d_multiple_cameras=[]
    matched_projection_errors_multiple_cameras=[]
    match_indices_list = []
    for subgraph_index in range(len(subgraphs_list)):
        if nx.number_of_edges(subgraphs_list[subgraph_index]) > 0:
            best_edge = sorted(subgraphs_list[subgraph_index].edges.data(), key = lambda x: x[2]['projection_error'])[0]
            matched_poses_3d_multiple_cameras.append(best_edge[2]['pose_3d'])
            matched_projection_errors_multiple_cameras.append(best_edge[2]['projection_error'])
            match_indices_list.append(np.vstack((best_edge[0], best_edge[1])))
    matched_poses_3d = np.asarray(matched_poses_3d_multiple_cameras)
    matched_projection_errors = np.asarray(matched_projection_errors_multiple_cameras)
    match_indices = np.asarray(match_indices_list)
    return matched_poses_3d, matched_projection_errors, match_indices, subgraphs_list, person_graph

def calculate_matched_poses_3d_multiple_cameras(
    openpose_data_multiple_cameras,
    camera_calibration_data_multiple_cameras,
    projection_error_threshold = 15.0):
    poses_3d_multiple_cameras, projection_errors_multiple_cameras = calculate_poses_3d_multiple_cameras(
        openpose_data_multiple_cameras,
        camera_calibration_data_multiple_cameras)
    matched_poses_3d, matched_projection_errors, match_indices, subgraphs_list, person_graph = extract_matched_poses_3d_multiple_cameras(
        poses_3d_multiple_cameras,
        projection_errors_multiple_cameras,
        projection_error_threshold)
    return matched_poses_3d, matched_projection_errors, match_indices

def draw_2d_pose_data_one_person(
    pose,
    pose_tag = None):
    all_points = pose.pose_keypoints
    valid_points = all_points[pose.valid_keypoints]
    centroid = np.mean(valid_points, 0)
    cvutilities.camera_utilities.draw_2d_image_points(valid_points)
    plt.text(centroid[0], centroid[1], pose_tag)

def plot_2d_pose_data_one_person(
    pose,
    pose_tag = None,
    image_size=[1296, 972]):
    draw_2d_pose_data_one_person(
        pose,
        pose_tag)
    cvutilities.camera_utilities.format_2d_image_plot(image_size)
    plt.show()

def draw_2d_pose_data_one_camera(
    poses,
    pose_tags = None):
    num_people = poses.num_poses
    if pose_tags is None:
        pose_tags = range(num_people)
    for person_index in range(num_people):
        draw_2d_pose_data_one_person(
            poses.poses[person_index],
            pose_tags[person_index])

def plot_2d_pose_data_one_camera(
    poses,
    pose_tags = None,
    image_size=[1296, 972]):
    draw_2d_pose_data_one_camera(
        poses,
        pose_tags)
    cvutilities.camera_utilities.format_2d_image_plot(image_size)
    plt.show()

def plot_2d_pose_data_multiple_cameras(
    poses_multiple_cameras,
    pose_tags_multiple_cameras = None,
    image_size=[1296, 972]):
    num_cameras = poses_multiple_cameras.num_cameras
    for camera_index in range(num_cameras):
        if pose_tags_multiple_cameras is None:
            pose_tags_single_camera = None
        else:
            pose_tags_single_camera = pose_tags_multiple_cameras[camera_index]
        plot_2d_pose_data_one_camera(
            poses_multiple_cameras.frames[camera_index],
            pose_tags_single_camera,
            image_size)

def draw_3d_pose_data_topdown_single_person(
    pose_data_single_person,
    pose_tag = None):
    valid_points = pose_data_single_person[np.isfinite(pose_data_single_person[:, 0])]
    centroid = np.mean(valid_points[:, :2], 0)
    cvutilities.camera_utilities.draw_3d_object_points_topdown(valid_points)
    if pose_tag is not None:
        plt.text(centroid[0], centroid[1], pose_tag)

def plot_3d_pose_data_topdown_single_person(
    pose_data_single_person,
    pose_tag = None,
    room_corners = None):
    draw_3d_pose_data_topdown_single_person(
        pose_data_single_person,
        pose_tag)
    cvutilities.camera_utilities.format_3d_topdown_plot(room_corners)
    plt.show()

def draw_3d_pose_data_topdown_multiple_people(
    pose_data_multiple_people,
    pose_tags = None):
    num_people = pose_data_multiple_people.shape[0]
    if pose_tags is None:
        pose_tags = range(num_people)
    for person_index in range(num_people):
        draw_3d_pose_data_topdown_single_person(
            pose_data_multiple_people[person_index],
            pose_tags[person_index])

def plot_3d_pose_data_topdown_multiple_people(
    pose_data_multiple_people,
    pose_tags = None,
    room_corners= None):
    draw_3d_pose_data_topdown_multiple_people(
        pose_data_multiple_people,
        pose_tags)
    cvutilities.camera_utilities.format_3d_topdown_plot(room_corners)
    plt.show()

def generate_match_pose_tags(
    match_indices,
    pose_tags_multiple_cameras):
    match_pose_tags = []
    for match_index in range(match_indices.shape[0]):
        match_pose_tags.append('{},{}'.format(
            pose_tags_multiple_cameras[match_indices[match_index, 0, 0]][match_indices[match_index, 0, 1]],
            pose_tags_multiple_cameras[match_indices[match_index, 1, 0]][match_indices[match_index, 1, 1]]))
    return match_pose_tags

def plot_matched_3d_pose_data_topdown(
    matched_3d_pose_data,
    match_indices,
    pose_tags_multiple_cameras = None,
    room_corners = None):
    if pose_tags_multiple_cameras is None:
        match_pose_tags = range(matched_3d_pose_data.shape[0])
    else:
        match_pose_tags = generate_match_pose_tags(
            match_indices,
            pose_tags_multiple_cameras)
    plot_3d_pose_data_topdown_multiple_people(
        matched_3d_pose_data,
        match_pose_tags,
        room_corners)
