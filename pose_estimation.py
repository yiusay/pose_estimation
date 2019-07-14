import sys

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from pyimagesearch.minivggnet import MiniVGGNet
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import backend as K
from imutils import build_montages
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import glob
import pywavefront

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def read_point_file(file_path):
	points = []
	with open(file_path) as f:
		content = f.readlines()

		for line in content:
			elements = line.split()
			points.append([float(x) for x in elements])

	return points

def get_3d_control_points_coord_for_all_parts(data_dir):
	annotation_3d_points = np.array(read_point_file(os.path.join(data_dir, "3D Rigid Tracking from RGB Images Dataset/annotations2D/BOX_Test_Annotations3DPoints.txt")))

	# extract only 4 representative parts
	annotation_3d_points = np.delete(annotation_3d_points, [1, 3, 5, 7], 0)
	assert annotation_3d_points.shape == (4, 3)

	control_point_offset = np.array([[0.05, 0, 0], [-0.05, 0, 0], [0, 0.05, 0], [0, -0.05, 0], [0, 0, 0.05], [0, 0, -0.05]])

	control_points_for_all_parts = []
	for annotation_pt in annotation_3d_points:
		control_pts_for_part = []
		for offset in control_point_offset:
			control_pts_for_part.append(annotation_pt + offset)

		control_points_for_all_parts.append(control_pts_for_part)

	control_points_for_all_parts = np.array(control_points_for_all_parts)
	assert control_points_for_all_parts.shape == (4, 6, 3)

	return control_points_for_all_parts

def difference_of_Gaussians(img):
	kernel1 = cv2.getGaussianKernel(10, 1)
	kernel2 = cv2.getGaussianKernel(10, 3)

	dog_img = cv2.sepFilter2D(img, -1, kernel1 - kernel2, kernel1 - kernel2)
	dog_img = cv2.normalize(dog_img, dog_img, alpha = 0, beta = 255, norm_type=cv2.NORM_MINMAX)

	return dog_img

def get_cropped_image(x, y, img):
	x_min = x - 16
	x_max = x + 16
	y_min = y - 16
	y_max = y + 16
	if x_min < 0 or x_max >= np.size(img,1) or y_min < 0 or y_max >= np.size(img,0):
		return np.empty(0)

	# crop = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[y_min:y_max, x_min:x_max]
	crop = difference_of_Gaussians(img[y_min:y_max, x_min:x_max]) # this was changed! check the latest extract_parts.py

	return crop

labelNames = ["0", "1", "2", "3", "background"]

data_dir = "/extract_parts_output/"

mesh = pywavefront.Wavefront(os.path.join(data_dir, "3D Rigid Tracking from RGB Images Dataset/BOX-TestAndInfo/openbox.obj"))

for name, material in mesh.materials.items():
	triangles = material.vertices

triangles = np.array(triangles)
triangles = triangles.reshape(-1, 3)

control_points_for_all_parts = get_3d_control_points_coord_for_all_parts(data_dir)

#path_to_data = os.path.join(data_dir, "3D Rigid Tracking from RGB Images Dataset/BOX-TestAndInfo/Test/video1")

#test from training data
path_to_data = "3D Rigid Tracking from RGB Images Dataset/video1"

num_input_files = len(glob.glob(os.path.join(path_to_data, "*.png")))

camera_matrix = np.float32([[2666.67, 0, 960], [0, 2666.67, 540], [0, 0, 1.]])

model = tf.keras.models.load_model('model_part_detection.h5')

model_for_control_points = []

for part_id in range(4):
	model_for_control_points.append(tf.keras.models.load_model("model_control_points_" + str(part_id) + ".h5"))

n = 0

while True:
	print("Pose estimation:" + str(n))
	testX = []
	test_locations = []

	file_name = "frame" + str(n).zfill(5) + ".png"
	img_path = os.path.join(path_to_data, file_name)
	img = cv2.imread(img_path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# scale image by the same ratio when generating training data
	scale_ratio = 0.25
	inv_scale_ratio = 1.0 / scale_ratio

	img = cv2.resize(img, None, fx=scale_ratio, fy=scale_ratio, interpolation=cv2.INTER_AREA)

	normalize = 1.0 / 255
	step = 2

	for y in range(16, img.shape[0]-16, step):
		for x in range(16, img.shape[1]-16, step):
			cropped = get_cropped_image(x, y, img)
			if cropped.shape == 0:
				continue;
				
			# scale data to the range of [0, 1]
			cropped = cropped.astype("float32") * normalize # normalize outside of loop!

			testX.append(cropped)
			test_locations.append((x, y))

	print("data ready")

	test_locations_width = int((img.shape[1] - 32) / step)
	test_locations_height = int((img.shape[0] - 32) / step)

	# convert from python list to numpy array
	testX = np.array(testX)

	if K.image_data_format() == "channels_first":
		testX = testX.reshape((testX.shape[0], 1, 32, 32))
	else:
		testX = testX.reshape((testX.shape[0], 32, 32, 1))

	print("testX.shape", testX.shape)

	probs = model.predict(testX)

	test_locations_probs = probs.reshape(test_locations_height, test_locations_width, probs.shape[1])

	to_test_location = lambda coord : coord * step + np.array([16, 16])
                    
	label_color = [ (0,0,255), (0,255,0), (255,0,0), (0,255,255), ]
	color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

	part_probability = []
	object_points_for_pose_estimation = []
	image_points_for_pose_estimation = []

	for part_id in range(4):
		test_locations_probs_for_part = test_locations_probs[:,:,part_id]

		retval, high_probs_locations = cv2.threshold(test_locations_probs_for_part, 0.99, 255, cv2.THRESH_BINARY)
		high_probs_locations = high_probs_locations.astype("ubyte")

		retval, labels, stats, centroids = cv2.connectedComponentsWithStats(high_probs_locations, connectivity=8)

		kMinRequiredArea = 3
		part_location_candidates = []

		for label in range(1, stats.shape[0]):
			area = stats[label, cv2.CC_STAT_AREA]
			centorid = np.array([int(centroids[label, 0]), int(centroids[label, 1])])

			part_location_candidates.append((to_test_location(centorid), area))
		
		part_location_candidates.sort(key=lambda candidate: candidate[1], reverse = True)

		if part_location_candidates and part_location_candidates[0][1] >= kMinRequiredArea:
			location = part_location_candidates[0][0]
			cv2.circle(color_img, tuple(location), 10, label_color[part_id], 1)

			x_min = location[0] - 32
			x_max = location[0] + 32
			y_min = location[1] - 32
			y_max = location[1] + 32
			if x_min < 0 or x_max >= np.size(img,1) or y_min < 0 or y_max >= np.size(img,0):
				continue

			crop_for_control_points = img[y_min:y_max, x_min:x_max]

			# scale data to the range of [0, 1]
			crop_for_control_points = crop_for_control_points.astype("float32") * normalize # normalize outside of loop!

			# convert from python list to numpy array
			testX = np.array([crop_for_control_points])

			if K.image_data_format() == "channels_first":
				testX = testX.reshape((testX.shape[0], 1, 64, 64))
			else:
				testX = testX.reshape((testX.shape[0], 64, 64, 1))

			print("testX.shape", testX.shape)

			control_points_offset = model_for_control_points[part_id].predict(testX)
			control_points_offset = control_points_offset.reshape(6, 1, 2)

			ctrl_pt_color = [ (0,0,255), (0,0,255), (0,255,0), (0,255,0), (255,0,0), (255,0,0) ]

			if False: # visualize control points
				for idx, offset in enumerate(control_points_offset):
					projected_control_points = location + offset * 0.25 #32 + offset * 0.25
					cv2.circle(color_img, (int(projected_control_points[0][0]), int(projected_control_points[0][1])), 3, ctrl_pt_color[idx], -1)

			object_points_for_pose_estimation.append(control_points_for_all_parts[part_id])
			image_points_for_pose_estimation.append(location * inv_scale_ratio + control_points_offset)


	if object_points_for_pose_estimation and image_points_for_pose_estimation:

		object_points_for_pose_estimation = np.array(object_points_for_pose_estimation).reshape(-1, 3)
		image_points_for_pose_estimation = np.array(image_points_for_pose_estimation).reshape(-1, 2)

		retval, rvec, tvec, inliers	= cv2.solvePnPRansac(object_points_for_pose_estimation, image_points_for_pose_estimation, camera_matrix, np.empty(0))

		to_tuple = lambda p : (int(p[0][0] * scale_ratio), int(p[0][1] * scale_ratio))

		projected_triangle_points, jac = cv2.projectPoints(triangles, rvec, tvec, camera_matrix, np.empty(0))
		for j in range(0, projected_triangle_points.shape[0], 3):
			cv2.line(color_img, to_tuple(projected_triangle_points[j]), to_tuple(projected_triangle_points[j+1]), (255, 255, 100))
			cv2.line(color_img, to_tuple(projected_triangle_points[j+1]), to_tuple(projected_triangle_points[j+2]), (255, 255, 100))
			cv2.line(color_img, to_tuple(projected_triangle_points[j+2]), to_tuple(projected_triangle_points[j]), (255, 255, 100))

	cv2.putText(color_img, str(n), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

	print("done")

	if True: # write images for video
		cv2.imwrite("output/pose_estimation_" + str(n) + ".jpg", color_img)
		print(str(n) + " / " + str(num_input_files))
		n = n + 1
		if n == num_input_files:
			sys.exit(1)
		continue

	cv2.imshow("Pose estimation", color_img)
	key = cv2.waitKey(0)

	print("key", key)

	if key == 27:         # wait for ESC key to exit
		cv2.destroyAllWindows()
		sys.exit(0)
	elif key == 81: # left arrow
		n = max(n - 1, 0)
	elif key == 82: # up arrow
		n = n + 5
	elif key == 84: # down arrow
		n = n - 5
	else:
		n = n + 1
