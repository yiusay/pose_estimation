import os
import sys
import numpy as np
import math
import cv2
import pickle

# scale image initially so that crop_size covers the target feature well
scale_ratio = 0.25 # this should be 1/3?

control_point_offset = np.array([[0.05, 0, 0], [-0.05, 0, 0], [0, 0.05, 0], [0, -0.05, 0], [0, 0, 0.05], [0, 0, -0.05]])

def generate_output_directories(output_dir, num_data):
	if os.path.exists(output_dir) == False:
		os.mkdir(output_dir)

	for i in range(num_data):
		part_dir = os.path.join(output_dir, str(i))
		if os.path.exists(part_dir) == False:
			os.mkdir(part_dir)

	background_dir = os.path.join(output_dir, "background")
	if os.path.exists(background_dir) == False:
		os.mkdir(background_dir)

	for i in range(num_data):
		part_dir = os.path.join(output_dir, "64x64_" + str(i))
		if os.path.exists(part_dir) == False:
			os.mkdir(part_dir)


def read_poses(path_to_data):
	file_name = os.path.join(path_to_data, "poseGT.txt")

	f = open(file_name, 'r')

	dict = {}

	with f as open_file_object:
		for line in open_file_object:
			elements = line.split()
			floats = [float(x) for x in elements[1:]]
			dict[elements[0]] = floats

	return dict

def read_point_file(file_path):
	points = []
	with open(file_path) as f:
		content = f.readlines()

		for line in content:
			elements = line.split()
			points.append([float(x) for x in elements])

	return points

def clamp(x, min_val, max_val):
	return min(max(x, min_val), max_val)

def difference_of_Gaussians(img):
	#run a 5x5 gaussian blur then a 3x3 gaussian blr
	kernel1 = cv2.getGaussianKernel(10, 1)
	kernel2 = cv2.getGaussianKernel(10, 3)

	dog_img = cv2.sepFilter2D(img, -1, kernel1 - kernel2, kernel1 - kernel2)
	dog_img = cv2.normalize(dog_img, dog_img, alpha = 0, beta = 255, norm_type=cv2.NORM_MINMAX)

	return dog_img

def output_cropped_image(x, y, img, output_path, crop_size, apply_dog = False):
	half_crop_size = int(crop_size / 2)
	x_min = x - half_crop_size
	x_max = x + half_crop_size
	y_min = y - half_crop_size
	y_max = y + half_crop_size
	if x_min < 0 or x_max >= np.size(img,1) or y_min < 0 or y_max >= np.size(img,0):
		return False

	crop = img[y_min:y_max, x_min:x_max]
	if apply_dog:
		crop = difference_of_Gaussians(crop)

	if crop.shape != (crop_size, crop_size):
		print("crop.shape", crop.shape, " something is wrong..")
		print(x_min, x_max, y_min, y_max)
		sys.exit(1)

	cv2.imwrite(output_path, crop)

	return True

def write_part_detection_training_img(img, i, annotation_projected_points, data_id, output_dir):
	crop_size = 32

	for j in range(len(annotation_projected_points)):
		projected_point = (annotation_projected_points[j] * scale_ratio).astype(int)
		#print(annotation_projected_points[j], projected_point)
		output_path = os.path.join(output_dir, str(j), str(data_id) + "_" + str(i).zfill(5) + ".png")
		output_cropped_image(projected_point[0][0], projected_point[0][1], img, output_path, crop_size, True)

	# generate background example
	while True:
		half_crop_size = int(crop_size / 2)
		x = np.random.randint(half_crop_size, np.size(img,1)-half_crop_size)
		y = np.random.randint(half_crop_size, np.size(img,0)-half_crop_size)
		close_to_annotated_points = False
		for j in range(len(annotation_projected_points)):
			projected_point = (annotation_projected_points[j] * scale_ratio).astype(int)
			if abs(x - projected_point[0][0]) <= half_crop_size or abs(y - projected_point[0][1]) <= half_crop_size:
				close_to_annotated_points = True
				break

		if close_to_annotated_points == False:
			# region not containing any annotated part is found
			output_path = os.path.join(output_dir, "background", str(data_id) + "_" + str(i).zfill(5) + ".png")
			output_cropped_image(x, y, img, output_path, crop_size, True)
			break

def write_control_points_detection_training_img(img, i, annotation_projected_points, data_id, output_dir):
	crop_size = 64
	is_cropped_image_saved = []

	for j in range(len(annotation_projected_points)):
		projected_point = (annotation_projected_points[j] * scale_ratio).astype(int)
		#print(annotation_projected_points[j], projected_point)
		output_path = os.path.join(output_dir, "64x64_" + str(j), str(data_id) + "_" + str(i).zfill(5) + ".png")
		success = output_cropped_image(projected_point[0][0], projected_point[0][1], img, output_path, crop_size)
		is_cropped_image_saved.append(success)

	return is_cropped_image_saved


def extract_images(data_id, data_dir, output_dir, all_projected_control_points):
	visualization = False

	path_to_data = os.path.join(data_dir, "3D Rigid Tracking from RGB Images Dataset/video" + str(data_id))

	if visualization == True:
		marker_3d_points = np.array(read_point_file(os.path.join(path_to_data, "markers3dPoints.txt")))
		marker_3d_points.reshape(-1, 3)

	control_points_for_all_parts = []
	for annotation_pt in annotation_3d_points:
		control_pts_for_part = []
		for offset in control_point_offset:
			control_pts_for_part.append(annotation_pt + offset)
		
		control_points_for_all_parts.append(control_pts_for_part)

	control_points_for_all_parts = np.array(control_points_for_all_parts)
	assert control_points_for_all_parts.shape == (4, 6, 3)

	Rts = read_poses(path_to_data)
	camera_matrix = np.float32([[2666.67, 0, 960], [0, 2666.67, 540], [0, 0, 1.]])

	#index = 0

	for i in range(0, len(Rts)):
		file_name = "frame" + str(i).zfill(5) + ".png"
		img_path = os.path.join(path_to_data, file_name)
		img = cv2.imread(img_path)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = cv2.resize(img, None, fx=scale_ratio, fy=scale_ratio, interpolation=cv2.INTER_AREA)

		Rt = Rts[file_name]

		if math.isnan(Rt[0]) == True:
			continue

		annotation_projected_points, jac = cv2.projectPoints(annotation_3d_points, np.array(Rt[0:3]), np.array(Rt[3:6]), camera_matrix, np.float32([[0, 0, 0, 0, 0]]))
		assert annotation_projected_points.shape == (4, 1, 2)

		write_part_detection_training_img(img, i, annotation_projected_points, data_id, output_dir)

		is_cropped_image_saved = write_control_points_detection_training_img(img, i, annotation_projected_points, data_id, output_dir)

		for j in range(control_points_for_all_parts.shape[0]):
			if is_cropped_image_saved[j] == False:
				continue

			projected_control_points_for_part, jac = cv2.projectPoints(control_points_for_all_parts[j], np.array(Rt[0:3]), np.array(Rt[3:6]), camera_matrix, np.float32([[0, 0, 0, 0, 0]]))
			# assert projected_control_points_for_part.shape == (6, 1, 2)

			# set relative offset from annotation_projected_points[j] here
			all_projected_control_points[j].append(projected_control_points_for_part - annotation_projected_points[j])

		# debug
		if False:
			color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
			for part_id, projected_control_points_for_part in enumerate(all_projected_control_points):
				for idx, point in enumerate(projected_control_points_for_part[-1]):
					color_code = idx % 6
					color = (0, 0, 255)
					if color_code == 2 or color_code == 3:
						color = (0, 255, 0)
					if color_code == 4 or color_code == 5:
						color = (255, 0, 0)

					point = (annotation_projected_points[part_id] + point) * scale_ratio
					cv2.circle(color_img, (int(point[0][0]), int(point[0][1])), 5, color, 1)

				cv2.putText(color_img, str(part_id), (int(point[0][0]), int(point[0][1])),
							cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

			cv2.imwrite(os.path.join("debug", str(i) + ".png"), color_img)

		if i%100 == 0:
			print(i, "processed..", flush=True)

		if visualization:
			for point in annotation_projected_points:
				print("point.shape", point.shape)
				cv2.circle(img, (int(point[0][0]), int(point[0][1])), 10, (0, 255, 0), 2)

			marker_projected_points, jac = cv2.projectPoints(marker_3d_points, np.array(Rt[0:3]), np.array(Rt[3:6]), camera_matrix, np.empty(0))
			for point in marker_projected_points:
				print("point.shape", point.shape)
				cv2.circle(img, (int(point[0][0]), int(point[0][1])), 10, (0, 100, 255), 2)

			cv2.imwrite('out.png',img)
			cv2.imshow('image',img)
			k = cv2.waitKey(0)
			if k == 27:         # wait for ESC key to exit
			    cv2.destroyAllWindows()
			    sys.exit(0)
			# elif k == ord('s'): # wait for 's' key to save and exit
			#     cv2.imwrite('messigray.png',img)
			#     cv2.destroyAllWindows()



if len(sys.argv) != 2:
	print("usage: " + sys.argv[0] + " data_dir")
	sys.exit(1)

data_dir = sys.argv[1]

annotation_3d_points = np.array(read_point_file(os.path.join(data_dir, "3D Rigid Tracking from RGB Images Dataset/annotations2D/BOX_Test_Annotations3DPoints.txt")))
annotation_3d_points.reshape(-1, 3)

# extract only 4 representative parts
annotation_3d_points = np.delete(annotation_3d_points, [1, 3, 5, 7], 0)
assert annotation_3d_points.shape == (4, 3)

output_dir = os.path.join(data_dir, "training")
generate_output_directories(output_dir, len(annotation_3d_points))

all_projected_control_points = []
for i in range(len(annotation_3d_points)):
	all_projected_control_points.append([])

for data_id in range(1,6):
	extract_images(data_id, data_dir, output_dir, all_projected_control_points)

for part_id, projected_control_points_for_part in enumerate(all_projected_control_points):
	file_name = "projected_control_points_" + str(part_id) + ".data"
	with open(os.path.join(data_dir, "training", file_name), 'wb') as filehandle:
		# store the data as binary data stream
		pickle.dump(projected_control_points_for_part, filehandle)
