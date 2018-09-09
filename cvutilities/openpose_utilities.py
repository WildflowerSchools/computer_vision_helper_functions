import cvutilities.camera_utilities
import cvutilities.datetime_utilities
import boto3
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import json

def generate_s3_datetime_strings(
    datetime):
    datetime_native_utc_naive = cvutilities.datetime_utilities.convert_to_native_utc_naive(datetime)
    date_string = datetime_native_utc_naive.strftime('%Y-%m-%d')
    time_string = datetime_native_utc_naive.strftime('%H-%M-%S')
    return date_string, time_string

def extract_keypoint_positions(openpose_json_data_single_person, keypoint_list_name='pose_keypoints'):
    keypoint_list = openpose_json_data_single_person[keypoint_list_name]
    keypoint_positions = np.array(keypoint_list).reshape((-1, 3))[:,:2]
    return keypoint_positions

def extract_keypoint_confidence_scores(openpose_json_data_single_person, keypoint_list_name='pose_keypoints'):
    keypoint_list = openpose_json_data_single_person[keypoint_list_name]
    keypoint_confidence_scores = np.array(keypoint_list).reshape((-1, 3))[:,2]
    return keypoint_confidence_scores

def extract_keypoints(openpose_json_data_single_person, keypoint_list_name='pose_keypoints'):
    keypoint_positions = extract_keypoint_positions(openpose_json_data_single_person, keypoint_list_name)
    keypoint_confidence_scores = extract_keypoint_confidence_scores(openpose_json_data_single_person, keypoint_list_name)
    return keypoint_positions, keypoint_confidence_scores

def fetch_openpose_data_from_s3_single_camera(
    classroom_name,
    datetime,
    camera_name,
    s3_bucket_name = 'wf-classroom-data',
    pose_directory_name = '2D-pose'):
    date_string, time_string = generate_s3_datetime_strings(datetime)
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

def fetch_openpose_data_from_s3_multiple_cameras(
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
    keypoint_positions_a,
    keypoint_confidence_scores_a,
    keypoint_positions_b,
    keypoint_confidence_scores_b):
    keypoint_positions_a = np.asarray(keypoint_positions_a)
    keypoint_confidence_scores_a = np.asarray(keypoint_confidence_scores_a)
    keypoint_positions_b = np.asarray(keypoint_positions_b)
    keypoint_confidence_scores_b = np.asarray(keypoint_confidence_scores_b)
    if (
        keypoint_positions_a.size == 0 or
        keypoint_confidence_scores_a.size == 0 or
        keypoint_positions_b.size == 0 or
        keypoint_confidence_scores_b.size == 0):
        raise ValueError('One or both sets of keypoints or confidence scores appear to be empty')
    keypoint_positions_a = keypoint_positions_a.reshape((-1, 2))
    keypoint_confidence_scores_a = keypoint_confidence_scores_a.flatten()
    keypoint_positions_b = keypoint_positions_b.reshape((-1, 2))
    keypoint_confidence_scores_b = keypoint_confidence_scores_b.flatten()
    if (keypoint_positions_a.shape[0] != keypoint_confidence_scores_a.shape[0] or
        keypoint_positions_a.shape[0] != keypoint_positions_b.shape[0] or
        keypoint_positions_a.shape[0] != keypoint_confidence_scores_b.shape[0]):
        raise ValueError('Keypoint arrays do not appear to be the same shape')
    common_keypoint_positions_mask = np.logical_and(
        keypoint_confidence_scores_a > 0.0,
        keypoint_confidence_scores_b > 0.0)
    image_points_a = keypoint_positions_a[common_keypoint_positions_mask]
    image_points_b = keypoint_positions_b[common_keypoint_positions_mask]
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
    openpose_data_single_person_a,
    openpose_data_single_person_b,
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
        openpose_data_single_person_a['keypoint_positions'],
        openpose_data_single_person_a['keypoint_confidence_scores'],
        openpose_data_single_person_b['keypoint_positions'],
        openpose_data_single_person_b['keypoint_confidence_scores'])
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
    openpose_data_single_camera_a,
    openpose_data_single_camera_b,
    rotation_vector_a,
    translation_vector_a,
    rotation_vector_b,
    translation_vector_b,
    camera_matrix,
    distortion_coefficients = 0,
    num_joints = 18):
    num_people_a = len(openpose_data_single_camera_a)
    num_people_b = len(openpose_data_single_camera_b)
    poses_3d = np.full((num_people_a, num_people_b, num_joints, 3), np.nan)
    projection_errors = np.full((num_people_a, num_people_b), np.nan)
    # match_mask = np.full((num_people_a, num_people_b), False)
    for person_index_a in range(num_people_a):
        for person_index_b in range(num_people_b):
            pose, projection_error_a, projection_error_b = calculate_pose_3d(
                openpose_data_single_camera_a[person_index_a],
                openpose_data_single_camera_b[person_index_b],
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
    camera_calibration_data_multiple_cameras,
    num_joints = 18):
    num_cameras = len(openpose_data_multiple_cameras)
    poses_3d_multiple_cameras = [[None]*num_cameras for _ in range(num_cameras)]
    projection_errors_multiple_cameras = [[None]*num_cameras for _ in range(num_cameras)]
    for camera_index_a in range(num_cameras):
        for camera_index_b in range(num_cameras):
            if camera_index_b > camera_index_a:
                poses_3d_multiple_cameras[camera_index_a][camera_index_b], projection_errors_multiple_cameras[camera_index_a][camera_index_b] = calculate_poses_3d_camera_pair(
                    openpose_data_multiple_cameras[camera_index_a],
                    openpose_data_multiple_cameras[camera_index_b],
                    camera_calibration_data_multiple_cameras[camera_index_a]['rotation_vector'],
                    camera_calibration_data_multiple_cameras[camera_index_a]['translation_vector'],
                    camera_calibration_data_multiple_cameras[camera_index_b]['rotation_vector'],
                    camera_calibration_data_multiple_cameras[camera_index_b]['translation_vector'],
                    camera_calibration_data_multiple_cameras[camera_index_a]['camera_matrix'],
                    camera_calibration_data_multiple_cameras[camera_index_a]['distortion_coefficients'],
                    num_joints)
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
    projection_error_threshold = 15.0,
    num_joints = 18):
    poses_3d_multiple_cameras, projection_errors_multiple_cameras = calculate_poses_3d_multiple_cameras(
        openpose_data_multiple_cameras,
        camera_calibration_data_multiple_cameras,
        num_joints)
    matched_poses_3d, matched_projection_errors, match_indices, subgraphs_list, person_graph = extract_matched_poses_3d_multiple_cameras(
        poses_3d_multiple_cameras,
        projection_errors_multiple_cameras,
        projection_error_threshold)
    return matched_poses_3d, matched_projection_errors, match_indices

def draw_2d_pose_data_one_person(
    openpose_data_single_person,
    pose_tag = None):
    all_points = openpose_data_single_person['keypoint_positions']
    confidence_scores = openpose_data_single_person['keypoint_confidence_scores']
    valid_points = all_points[confidence_scores > 0.0]
    centroid = np.mean(valid_points, 0)
    cvutilities.camera_utilities.draw_2d_image_points(valid_points)
    plt.text(centroid[0], centroid[1], pose_tag)

def plot_2d_pose_data_one_person(
    openpose_data_single_person,
    pose_tag = None,
    image_size=[1296, 972]):
    draw_2d_pose_data_one_person(
        openpose_data_single_person,
        pose_tag)
    cvutilities.camera_utilities.format_2d_image_plot(image_size)
    plt.show()

def draw_2d_pose_data_one_camera(
    openpose_data_single_camera,
    pose_tags_single_camera = None):
    num_people = len(openpose_data_single_camera)
    if pose_tags_single_camera is None:
        pose_tags_single_camera = range(num_people)
    for person_index in range(num_people):
        draw_2d_pose_data_one_person(
            openpose_data_single_camera[person_index],
            pose_tags_single_camera[person_index])

def plot_2d_pose_data_one_camera(
    openpose_data_single_camera,
    pose_tags_single_camera = None,
    image_size=[1296, 972]):
    draw_2d_pose_data_one_camera(
        openpose_data_single_camera,
        pose_tags_single_camera)
    cvutilities.camera_utilities.format_2d_image_plot(image_size)
    plt.show()

def plot_2d_pose_data_multiple_cameras(
    openpose_data_multiple_cameras,
    pose_tags_multiple_cameras = None,
    image_size=[1296, 972]):
    num_cameras = len(openpose_data_multiple_cameras)
    for camera_index in range(num_cameras):
        if pose_tags_multiple_cameras is None:
            pose_tags_single_camera = None
        else:
            pose_tags_single_camera = pose_tags_multiple_cameras[camera_index]
        plot_2d_pose_data_one_camera(
            openpose_data_multiple_cameras[camera_index],
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
