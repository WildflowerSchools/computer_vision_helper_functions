import cvutilities.datetime_utilities
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import boto3
import json
import os

# For now, the Wildflower-specific S3 functionality is intermingled with the more
# general S3 functionality. We should probably separate these at some point. For
# the S3 functions below to work, the environment must include AWS_ACCESS_KEY_ID and
# AWS_SECRET_ACCESS_KEY variables and that access key must have read permissions
# for the relevant buckets. You can set these environment variables manually or
# by using the AWS CLI.
classroom_data_wildflower_s3_bucket_name = 'wf-classroom-data'
camera_image_wildflower_s3_directory_name = 'camera'

# Generate the Wildflower S3 object name for a camera image from a classroom
# name, a camera name, and a Python datetime object
def generate_camera_image_wildflower_s3_object_name(
    classroom_name,
    camera_name,
    datetime):
    date_string, time_string = generate_wildflower_s3_datetime_strings(datetime)
    camera_image_wildflower_s3_object_name = 'camera-{}/{}/{}/{}/still_{}-{}.jpg'.format(
        classroom_name,
        camera_image_wildflower_s3_directory_name,
        date_string,
        camera_name,
        date_string,
        time_string)
    return camera_image_wildflower_s3_object_name

# Generate date and time strings (as they appear in our Wildflower S3 object
# names) from a Python datetime object
def generate_wildflower_s3_datetime_strings(
    datetime):
    datetime_native_utc_naive = cvutilities.datetime_utilities.convert_to_native_utc_naive(datetime)
    date_string = datetime_native_utc_naive.strftime('%Y-%m-%d')
    time_string = datetime_native_utc_naive.strftime('%H-%M-%S')
    return date_string, time_string

def fetch_camera_calibration_data_from_local_drive_single_camera(
    camera_name,
    camera_calibration_data_directory = '.'):
    camera_calibration_data_filename = camera_name + '_cal.json'
    camera_calibration_data_path = os.path.join(
        camera_calibration_data_directory,
        camera_calibration_data_filename)
    with open(camera_calibration_data_path) as json_file:
        camera_calibration_json_data_single_camera = json.load(json_file)
    camera_calibration_data_single_camera = {
        'camera_matrix': np.asarray(camera_calibration_json_data_single_camera['cameraMatrix']),
        'distortion_coefficients': np.asarray(camera_calibration_json_data_single_camera['distortionCoefficients']),
        'rotation_vector': np.asarray(camera_calibration_json_data_single_camera['rotationVector']),
        'translation_vector': np.asarray(camera_calibration_json_data_single_camera['translationVector'])}
    return camera_calibration_data_single_camera

def fetch_camera_calibration_data_from_local_drive_multiple_cameras(
    camera_names,
    camera_calibration_data_directory = '.'):
    camera_calibration_data_multiple_cameras = []
    for camera_name in camera_names:
        camera_calibration_data_multiple_cameras.append(
            fetch_camera_calibration_data_from_local_drive_single_camera(
                camera_name,
                camera_calibration_data_directory))
    return camera_calibration_data_multiple_cameras

# Fetch an image from a local image file and return it in OpenCV format
def fetch_image_from_local_drive(image_path):
    image = cv.imread(image_path)
    return image

# Fetch an image stored on S3 and specified by S3 bucket and object names and
# return it in OpenCV format
def fetch_image_from_s3_object(s3_bucket_name, s3_object_name):
    s3_object = boto3.resource('s3').Object(s3_bucket_name, s3_object_name)
    s3_object_content = s3_object.get()['Body'].read()
    s3_object_content_array = np.frombuffer(s3_object_content, dtype=np.uint8)
    image = cv.imdecode(s3_object_content_array, flags = cv.IMREAD_UNCHANGED)
    return image

# Fetch a camera image stored on S3 and specified by classroom name, camera
# name, and Python datetime object and return it in OpenCV format
def fetch_image_from_wildflower_s3(
    classroom_name,
    camera_name,
    datetime):
    s3_bucket_name = classroom_data_wildflower_s3_bucket_name
    s3_object_name = generate_camera_image_wildflower_s3_object_name(
        classroom_name,
        camera_name,
        datetime)
    image = fetch_image_from_s3_object(s3_bucket_name, s3_object_name)
    return image

# Take an image in OpenCV format and draw it as a background for a Matplotlib
# plot. We separate this from the plotting function below because we might want
# to draw other elements before formatting and showing the chart.
def draw_background_image(
    image,
    alpha = 0.4):
    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB), alpha = alpha)

# Take an image in OpenCV format and plot it as a Matplotlib plot. Calls the
# drawing function above, adds formating, and shows the plot.
def plot_background_image(
    image,
    alpha = 0.4):
    image_size=np.array([
        image.shape[1],
        image.shape[0]])
    draw_background_image(image, alpha)
    format_2d_image_plot(image_size)
    plt.show()

def compose_transformations(
    rotation_vector_1,
    translation_vector_1,
    rotation_vector_2,
    translation_vector_2):
    rotation_vector_1= np.asarray(rotation_vector_1).reshape(3)
    translation_vector_1 = np.asarray(translation_vector_1).reshape(3)
    rotation_vector_2 = np.asarray(rotation_vector_2).reshape(3)
    translation_vector_2 = np.asarray(translation_vector_2).reshape(3)
    rotation_vector_composed, translation_vector_composed = cv.composeRT(
        rotation_vector_1,
        translation_vector_1,
        rotation_vector_2,
        translation_vector_2)[:2]
    rotation_vector_composed = np.squeeze(rotation_vector_composed)
    translation_vector_composed = np.squeeze(translation_vector_composed)
    return rotation_vector_composed, translation_vector_composed

def invert_transformation(
    rotation_vector,
    translation_vector):
    rotation_vector = np.asarray(rotation_vector).reshape(3)
    translation_vector = np.asarray(translation_vector).reshape(3)
    new_rotation_vector, new_translation_vector = compose_transformations(
        np.array([0.0, 0.0, 0.0]),
        -translation_vector,
        -rotation_vector,
        np.array([0.0, 0.0, 0.0]))
    new_rotation_vector = np.squeeze(new_rotation_vector)
    new_translation_vector = np.squeeze(new_translation_vector)
    return new_rotation_vector, new_translation_vector

def transform_object_points(
    object_points,
    rotation_vector = np.array([0.0, 0.0, 0.0]),
    translation_vector = np.array([0.0, 0.0, 0.0])):
    object_points = np.asarray(object_points)
    rotation_vector = np.asarray(rotation_vector)
    translation_vector = np.asarray(translation_vector)
    if object_points.size == 0:
        return object_points
    object_points = object_points.reshape((-1, 3))
    rotation_vector = rotation_vector.reshape(3)
    translation_vector = translation_vector.reshape(3)
    transformed_points = np.add(
        np.matmul(
            cv.Rodrigues(rotation_vector)[0],
            object_points.T).T,
        translation_vector.reshape((1, 3)))
    transformed_points = np.squeeze(transformed_points)
    return transformed_points

def generate_camera_pose(
    camera_position = np.array([0.0, 0.0, 0.0]),
    yaw = 0.0,
    pitch = 0.0,
    roll = 0.0):
    # yaw: 0.0 points north (along the positive y-axis), positive angles rotate counter-clockwise
    # pitch: 0.0 is level with the ground, positive angles rotate upward
    # roll: 0.0 is level with the ground, positive angles rotate clockwise
    # All angles in radians
    camera_position = np.asarray(camera_position).reshape(3)
    # First: Move the camera to the specified position
    rotation_vector_1 = np.array([0.0, 0.0, 0.0])
    translation_vector_1 = -camera_position
    # Second: Rotate the camera so when we lower to the specified inclination, it will point in the specified compass direction
    rotation_vector_2 = np.array([0.0, 0.0, -(yaw - np.pi/2)])
    translation_vector_2 = np.array([0.0, 0.0, 0.0])
    # Third: Lower to the specified inclination
    rotation_vector_2_3 = np.array([(np.pi/2 - pitch), 0.0, 0.0])
    translation_vector_2_3 = np.array([0.0, 0.0, 0.0])
    # Fourth: Roll the camera by the specified angle
    rotation_vector_2_3_4 = np.array([0.0, 0.0, -roll])
    translation_vector_2_3_4 = np.array([0.0, 0.0, 0.0])
    # Combine these four moves
    rotation_vector_1_2, translation_vector_1_2 = compose_transformations(
        rotation_vector_1,
        translation_vector_1,
        rotation_vector_2,
        translation_vector_2)
    rotation_vector_1_2_3, translation_vector_1_2_3 = compose_transformations(
        rotation_vector_1_2,
        translation_vector_1_2,
        rotation_vector_2_3,
        translation_vector_2_3)
    rotation_vector, translation_vector = compose_transformations(
        rotation_vector_1_2_3,
        translation_vector_1_2_3,
        rotation_vector_2_3_4,
        translation_vector_2_3_4)
    rotation_vector = np.squeeze(rotation_vector)
    translation_vector = np.squeeze(translation_vector)
    return rotation_vector, translation_vector

def extract_camera_position(
    rotation_vector,
    translation_vector):
    rotation_vector = np.asarray(rotation_vector).reshape(3)
    translation_vector = np.asarray(translation_vector).reshape(3)
    new_rotation_vector, new_translation_vector = compose_transformations(
        rotation_vector,
        translation_vector,
        -rotation_vector,
        np.array([0.0, 0.0, 0.0]))
    camera_position = -np.squeeze(new_translation_vector)
    return camera_position

def extract_camera_direction(
    rotation_vector,
    translation_vector):
    rotation_vector = np.asarray(rotation_vector).reshape(3)
    translation_vector = np.asarray(translation_vector).reshape(3)
    camera_direction = np.matmul(
        cv.Rodrigues(-rotation_vector)[0],
        np.array([[0.0], [0.0], [1.0]]))
    camera_direction = np.squeeze(camera_direction)
    return camera_direction

def reconstruct_z_rotation(x, y):
    if x>=0.0 and y>=0.0: return np.arctan(y/x)
    if x>=0.0 and y<0.0: return np.arctan(y/x) + 2*np.pi
    return np.arctan(y/x) + np.pi

# Currently unused; needs to be fixed up for cases in which x and/or y are close
# to zero
def extract_yaw_from_camera_direction(
    camera_direction):
    camera_direction = np.asarray(camera_direction).reshape(3)
    yaw = reconstruct_z_rotation(
        camera_direction[0],
        camera_direction[1])
    return yaw

def generate_camera_matrix(
    focal_length,
    principal_point):
    focal_length = np.asarray(focal_length).reshape(2)
    principal_point = np.asarray(principal_point).reshape(2)
    camera_matrix = np.array([
        [focal_length[0], 0, principal_point[0]],
        [0, focal_length[1], principal_point[1]],
        [0, 0, 1.0]])
    return camera_matrix

def generate_projection_matrix(
    camera_matrix,
    rotation_vector,
    translation_vector):
    camera_matrix = np.asarray(camera_matrix).reshape((3,3))
    rotation_vector = np.asarray(rotation_vector).reshape(3)
    translation_vector = np.asarray(translation_vector).reshape(3)
    projection_matrix = np.matmul(
        camera_matrix,
        np.concatenate((
            cv.Rodrigues(rotation_vector)[0],
            translation_vector.reshape((3,1))),
            axis=1))
    return(projection_matrix)

def project_points(
        object_points,
        rotation_vector,
        translation_vector,
        camera_matrix,
        distortion_coefficients):
    object_points = np.asarray(object_points)
    rotation_vector = np.asarray(rotation_vector)
    translation_vector = np.asarray(translation_vector)
    camera_matrix = np.asarray(camera_matrix)
    distortion_coefficients = np.asarray(distortion_coefficients)
    if object_points.size == 0:
        return np.zeros((0, 2))
    object_points = object_points.reshape((-1, 3))
    rotation_vector = rotation_vector.reshape(3)
    translation_vector = translation_vector.reshape(3)
    camera_matrix = camera_matrix.reshape((3,3))
    image_points = cv.projectPoints(
        object_points,
        rotation_vector,
        translation_vector,
        camera_matrix,
        distortion_coefficients)[0]
    image_points = np.squeeze(image_points)
    return image_points

def undistort_points(
    image_points,
    camera_matrix,
    distortion_coefficients):
    image_points = np.asarray(image_points)
    camera_matrix = np.asarray(camera_matrix)
    distortion_coefficients = np.asarray(distortion_coefficients)
    if image_points.size == 0:
        return image_points
    image_points = image_points.reshape((-1, 1, 2))
    camera_matrix = camera_matrix.reshape((3,3))
    undistorted_points = cv.undistortPoints(
        image_points,
        camera_matrix,
        distortion_coefficients,
        P=camera_matrix)
    undistorted_points = np.squeeze(undistorted_points)
    return undistorted_points

def estimate_camera_pose_from_image_points(
    image_points_1,
    image_points_2,
    camera_matrix,
    rotation_vector_1 = np.array([0.0, 0.0, 0.0]),
    translation_vector_1 = np.array([0.0, 0.0, 0.0]),
    distance_between_cameras = 1.0):
    image_points_1 = np.asarray(image_points_1)
    image_points_2 = np.asarray(image_points_2)
    camera_matrix = np.asarray(camera_matrix)
    rotation_vector_1 = np.asarray(rotation_vector_1)
    translation_vector_1 = np.asarray(translation_vector_1)
    if image_points_1.size == 0 or image_points_2.size == 0:
        raise ValueError('One or both sets of image points appear to be empty')
    image_points_1 = image_points_1.reshape((-1, 2))
    image_points_2 = image_points_2.reshape((-1, 2))
    if image_points_1.shape != image_points_2.shape:
        raise ValueError('Sets of image points do not appear to be the same shape')
    camera_matrix = camera_matrix.reshape((3,3))
    rotation_vector_1 = rotation_vector_1.reshape(3)
    translation_vector_1 = translation_vector_1.reshape(3)
    essential_matrix, mask = cv.findEssentialMat(
        image_points_1,
        image_points_2,
        camera_matrix)
    relative_rotation_matrix, relative_translation_vector = cv.recoverPose(
        essential_matrix,
        image_points_1,
        image_points_2,
        camera_matrix,
        mask=mask)[1:3]
    relative_rotation_vector = cv.Rodrigues(relative_rotation_matrix)[0]
    relative_translation_vector = relative_translation_vector * distance_between_cameras
    rotation_vector_2, translation_vector_2 = compose_transformations(
        rotation_vector_1,
        translation_vector_1,
        relative_rotation_vector,
        relative_translation_vector)
    rotation_vector_2 = np.squeeze(rotation_vector_2)
    translation_vector_2 = np.squeeze(translation_vector_2)
    return rotation_vector_2, translation_vector_2

def reconstruct_object_points_from_camera_poses(
    image_points_1,
    image_points_2,
    camera_matrix,
    rotation_vector_1,
    translation_vector_1,
    rotation_vector_2,
    translation_vector_2):
    image_points_1 = np.asarray(image_points_1)
    image_points_2 = np.asarray(image_points_2)
    camera_matrix = np.asarray(camera_matrix)
    rotation_vector_1 = np.asarray(rotation_vector_1)
    translation_vector_1 = np.asarray(translation_vector_1)
    rotation_vector_2 = np.asarray(rotation_vector_2)
    translation_vector_2 = np.asarray(translation_vector_2)
    if image_points_1.size == 0 or image_points_2.size == 0:
        return np.zeros((0,3))
    image_points_1 = image_points_1.reshape((-1, 2))
    image_points_2 = image_points_2.reshape((-1, 2))
    if image_points_1.shape != image_points_2.shape:
        raise ValueError('Sets of image points do not appear to be the same shape')
    camera_matrix = camera_matrix.reshape((3,3))
    rotation_vector_1 = rotation_vector_1.reshape(3)
    translation_vector_1 = translation_vector_1.reshape(3)
    rotation_vector_2 = rotation_vector_2.reshape(3)
    translation_vector_2 = translation_vector_2.reshape(3)
    projection_matrix_1 = generate_projection_matrix(
        camera_matrix,
        rotation_vector_1,
        translation_vector_1)
    projection_matrix_2 = generate_projection_matrix(
        camera_matrix,
        rotation_vector_2,
        translation_vector_2)
    object_points_homogeneous = cv.triangulatePoints(
        projection_matrix_1,
        projection_matrix_2,
        image_points_1.T,
        image_points_2.T)
    object_points = cv.convertPointsFromHomogeneous(
        object_points_homogeneous.T)
    object_points = np.squeeze(object_points)
    return object_points

def reconstruct_object_points_from_relative_camera_pose(
    image_points_1,
    image_points_2,
    camera_matrix,
    relative_rotation_vector,
    relative_translation_vector,
    rotation_vector_1 = np.array([[0.0], [0.0], [0.0]]),
    translation_vector_1 = np.array([[0.0], [0.0], [0.0]]),
    distance_between_cameras = 1.0):
    image_points_1 = np.asarray(image_points_1)
    image_points_2 = np.asarray(image_points_2)
    camera_matrix = np.asarray(camera_matrix)
    relative_rotation_vector = np.asarray(relative_rotation_vector)
    relative_translation_vector = np.asarray(relative_translation_vector)
    rotation_vector_1 = np.asarray(rotation_vector_1)
    translation_vector_1 = np.asarray(translation_vector_1)
    if image_points_1.size == 0 or image_points_2.size == 0:
        return np.zeros((0,3))
    image_points_1 = image_points_1.reshape((-1, 2))
    image_points_2 = image_points_2.reshape((-1, 2))
    if image_points_1.shape != image_points_2.shape:
        raise ValueError('Sets of image points do not appear to be the same shape')
    camera_matrix = camera_matrix.reshape((3,3))
    relative_rotation_vector = relative_rotation_vector.reshape(3)
    relative_translation_vector = relative_translation_vector.reshape(3)
    rotation_vector_1 = rotation_vector_1.reshape(3)
    translation_vector_1 = translation_vector_1.reshape(3)
    rotation_vector_2, translation_vector_2 = cv.composeRT(
        rotation_vector_1,
        translation_vector_1,
        relative_rotation_vector,
        relative_translation_vector*distance_between_cameras)[:2]
    object_points = reconstruct_object_points_from_camera_poses(
        image_points_1,
        image_points_2,
        camera_matrix,
        rotation_vector_1,
        translation_vector_1,
        rotation_vector_2,
        translation_vector_2)
    return object_points

def reconstruct_object_points_from_image_points(
    image_points_1,
    image_points_2,
    camera_matrix,
    rotation_vector_1 = np.array([[0.0], [0.0], [0.0]]),
    translation_vector_1 = np.array([[0.0], [0.0], [0.0]]),
    distance_between_cameras = 1.0):
    image_points_1 = np.asarray(image_points_1)
    image_points_2 = np.asarray(image_points_2)
    camera_matrix = np.asarray(camera_matrix)
    rotation_vector_1 = np.asarray(rotation_vector_1)
    translation_vector_1 = np.asarray(translation_vector_1)
    if image_points_1.size == 0 or image_points_2.size == 0:
        return np.zeros((0,3))
    image_points_1 = image_points_1.reshape((-1, 2))
    image_points_2 = image_points_2.reshape((-1, 2))
    if image_points_1.shape != image_points_2.shape:
        raise ValueError('Sets of image points do not appear to be the same shape')
    camera_matrix = camera_matrix.reshape((3,3))
    rotation_vector_1 = rotation_vector_1.reshape(3)
    translation_vector_1 = translation_vector_1.reshape(3)
    rotation_vector_2, translation_vector_2 = estimate_camera_pose_from_image_points(
        image_points_1,
        image_points_2,
        camera_matrix,
        rotation_vector_1,
        translation_vector_1,
        distance_between_cameras)
    object_points = reconstruct_object_points_from_camera_poses(
        image_points_1,
        image_points_2,
        camera_matrix,
        rotation_vector_1,
        translation_vector_1,
        rotation_vector_2,
        translation_vector_2)
    return object_points

def estimate_camera_pose_from_plane_object_points(
    input_object_points,
    height,
    origin_index,
    x_axis_index,
    y_reference_point,
    y_reference_point_sign,
    distance_calibration_indices,
    calibration_distance):
    input_object_points = np.asarray(input_object_points)
    if input_object_points.size == 0:
        raise ValueError('Obect point array appears to be empty')
    input_object_points = input_object_points.reshape((-1,3))

    scale_factor = np.divide(
        calibration_distance,
        np.linalg.norm(
            np.subtract(
                input_object_points[distance_calibration_indices[0]],
                input_object_points[distance_calibration_indices[1]])))

    object_points_1 = np.multiply(
        input_object_points,
        scale_factor)

    def objective_function(parameters):
        rotation_x = parameters[0]
        rotation_y = parameters[1]
        translation_z = parameters[2]
        object_points_transformed = transform_object_points(
            object_points_1,
            np.array([rotation_x, rotation_y, 0.0]),
            np.array([0.0, 0.0, translation_z]))
        return np.sum(np.square(object_points_transformed[:, 2] - height))

    optimization_solution = scipy.optimize.minimize(
        objective_function,
        np.array([0.0, 0.0, 0.0]))

    rotation_x_a = optimization_solution['x'][0]
    rotation_y_a = optimization_solution['x'][1]
    translation_z_a = optimization_solution['x'][2]

    rotation_x_rotation_y_a_norm = np.linalg.norm([rotation_x_a, rotation_y_a])

    rotation_x_b = rotation_x_a * ((rotation_x_rotation_y_a_norm + np.pi)/rotation_x_rotation_y_a_norm)
    rotation_y_b = rotation_y_a * ((rotation_x_rotation_y_a_norm + np.pi)/rotation_x_rotation_y_a_norm)
    translation_z_b = - translation_z_a

    rotation_vector_2_a = np.array([rotation_x_a, rotation_y_a, 0.0])
    translation_vector_2_a = np.array([0.0, 0.0, translation_z_a])
    object_points_2_a = transform_object_points(
        object_points_1,
        rotation_vector_2_a,
        translation_vector_2_a)

    rotation_vector_2_b = np.array([rotation_x_b, rotation_y_b, 0.0])
    translation_vector_2_b = np.array([0.0, 0.0, translation_z_b])
    object_points_2_b = transform_object_points(
        object_points_1,
        rotation_vector_2_b,
        translation_vector_2_b)

    sign_a = np.sign(
        np.cross(
            np.subtract(
                object_points_2_a[x_axis_index],
                object_points_2_a[origin_index]),
            np.subtract(
                object_points_2_a[y_reference_point],
                object_points_2_a[origin_index]))[2])

    sign_b = np.sign(
        np.cross(
            np.subtract(
                object_points_2_b[x_axis_index],
                object_points_2_b[origin_index]),
            np.subtract(
                object_points_2_b[y_reference_point],
                object_points_2_b[origin_index]))[2])

    if sign_a == y_reference_point_sign:
        rotation_vector_2 = rotation_vector_2_a
        translation_vector_2 = translation_vector_2_a
        object_points_2 = object_points_2_a
    else:
        rotation_vector_2 = rotation_vector_2_b
        translation_vector_2 = translation_vector_2_b
        object_points_2 = object_points_2_b

    xy_shift = - object_points_2[origin_index, :2]

    rotation_vector_3 = np.array([0.0, 0.0, 0.0])
    translation_vector_3 = np.array([xy_shift[0], xy_shift[1], 0.0])
    object_points_3 = transform_object_points(
        object_points_2,
        rotation_vector_3,
        translation_vector_3)

    final_z_rotation = - reconstruct_z_rotation(
        object_points_3[x_axis_index, 0],
        object_points_3[x_axis_index, 1])

    rotation_vector_4 = np.array([0.0, 0.0, final_z_rotation])
    translation_vector_4 = np.array([0.0, 0.0, 0.0])
    object_points_4 = transform_object_points(
        object_points_3,
        rotation_vector_4,
        translation_vector_4)

    rotation_vector_2_3, translation_vector_2_3 = compose_transformations(
        rotation_vector_2,
        translation_vector_2,
        rotation_vector_3,
        translation_vector_3)

    rotation_vector_2_3_4, translation_vector_2_3_4 = compose_transformations(
        rotation_vector_2_3,
        translation_vector_2_3,
        rotation_vector_4,
        translation_vector_4)

    camera_rotation_vector, camera_translation_vector = invert_transformation(
        rotation_vector_2_3_4,
        translation_vector_2_3_4)

    return camera_rotation_vector, camera_translation_vector, scale_factor, object_points_4

def estimate_camera_poses_from_plane_image_points(
    image_points_1,
    image_points_2,
    camera_matrix,
    height,
    origin_index,
    x_axis_index,
    y_reference_point,
    y_reference_point_sign,
    distance_calibration_indices,
    calibration_distance):
    image_points_1 = np.asarray(image_points_1)
    image_points_2 = np.asarray(image_points_2)
    camera_matrix = np.asarray(camera_matrix)
    if image_points_1.size == 0 or image_points_2.size == 0:
        raise ValueError('One or both sets of image points appear to be empty')
    image_points_1 = image_points_1.reshape((-1, 2))
    image_points_2 = image_points_2.reshape((-1, 2))
    if image_points_1.shape != image_points_2.shape:
        raise ValueError('Sets of image points do not appear to be the same shape')
    camera_matrix = camera_matrix.reshape((3,3))
    relative_rotation_vector, relative_translation_vector = estimate_camera_pose_from_image_points(
        image_points_1,
        image_points_2,
        camera_matrix)
    input_object_points = reconstruct_object_points_from_image_points(
        image_points_1,
        image_points_2,
        camera_matrix)
    rotation_vector_1, translation_vector_1, scale_factor = estimate_camera_pose_from_plane_object_points(
        input_object_points,
        height,
        origin_index,
        x_axis_index,
        y_reference_point,
        y_reference_point_sign,
        distance_calibration_indices,
        calibration_distance)[:3]
    rotation_vector_2, translation_vector_2 = compose_transformations(
        rotation_vector_1,
        translation_vector_1,
        relative_rotation_vector,
        relative_translation_vector*scale_factor)
    return rotation_vector_1, translation_vector_1, rotation_vector_2, translation_vector_2

def draw_2d_image_points(
    image_points,
    point_labels=[]):
    image_points = np.asarray(image_points).reshape((-1, 2))
    points_image_u = image_points[:, 0]
    points_image_v = image_points[:, 1]
    plt.plot(
        points_image_u,
        points_image_v,
        '.')
    if len(point_labels) > 0:
        for i in range(len(point_labels)):
            plt.text(points_image_u[i], points_image_v[i], point_labels[i])

def format_2d_image_plot(
    image_size=[1296, 972]):
    plt.xlim(0, image_size[0])
    plt.ylim(0, image_size[1])
    plt.xlabel(r'$u$')
    plt.ylabel(r'$v$')
    plt.gca().invert_yaxis()
    plt.gca().xaxis.set_ticks_position('top')
    plt.gca().xaxis.set_label_position('top')
    plt.gca().set_aspect('equal')

def plot_2d_image_points(
    image_points,
    image_size=[1296, 972],
    point_labels=[]):
    image_points = np.asarray(image_points).reshape((-1, 2))
    draw_2d_image_points(
        image_points,
        point_labels)
    format_2d_image_plot(image_size)
    plt.show()

def draw_3d_object_points_topdown(
    object_points,
    point_labels=[]):
    object_points = np.asarray(object_points).reshape((-1, 3))
    points_x = object_points[:, 0]
    points_y = object_points[:, 1]
    plt.plot(
        points_x,
        points_y,
        '.')
    if len(point_labels) > 0:
        for i in range(len(point_labels)):
            plt.text(points_x[i], points_y[i], point_labels[i])

def format_3d_topdown_plot(
    room_corners = None):
    if room_corners is not None:
        plt.xlim(room_corners[0][0], room_corners[1][0])
        plt.ylim(room_corners[0][1], room_corners[1][1])
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.gca().set_aspect('equal')

def plot_3d_object_points_topdown(
    object_points,
    room_corners = None,
    point_labels=[]):
    object_points = np.asarray(object_points).reshape((-1, 3))
    draw_3d_object_points_topdown(
        object_points,
        point_labels)
    format_3d_topdown_plot(
        room_corners)
    plt.show()
