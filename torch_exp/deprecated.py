def print_result(result, output_image: mp.Image, timestamp_ms: int):
	# print('pose landmarker result: {}'.format(result))
	angle_2d, coods = parse_angle_from_2d(result, 'elbow')
	logger.info("angle 2d: {:.0f}".format(angle_2d))


def process_stream_v2_UI():
	if 'cap' not in st.session_state:
		cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
		cap.set(3, 800)
		cap.set(4, 600)
		st.session_state['cap'] = cap
	else:
		cap = st.session_state['cap']

	recording_delay = 5
	recording_length_sec = 5  # todo change it to 10

	col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
	start_recording = col1.button(f"▶ Start Recording ({recording_delay}s delay)", use_container_width=True)
	countdown_container = st.empty()
	frame_placeholder = st.empty()

	if start_recording:
		# asyncio.run(start_countdown(recording_delay, countdown_container))
		# stop_recording = col2.button("⏸ Stop Recording", use_container_width=True)

		time_string = time.strftime("%m-%d-%Y--%H-%M-%S", time.localtime())
		recording_file_name = f'recording_stream_{time_string}.mp4'
		logger.error(recording_file_name)

		recording_start_time = time.time()

		height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		out_file = cv2.VideoWriter(VIDEO_FILES_DIR + recording_file_name, -1, 30, (width, height))

		while cap.isOpened():
			ret, frame = cap.read()

			if frame is None:
				st.error("Failed to fetch frame. Retrying...")
				time.sleep(0.3)
				continue

			frame_placeholder.image(frame, channels='BGR')

			# if stop_recording:
			# 	logger.warning(f"Stop button pressed. Stopping recording..")
			# 	break

			if time.time() > recording_start_time + recording_length_sec:
				logger.warning(f"{recording_length_sec} seconds passed. Stopping recording..")
				break

			countdown_container.header(
				'Recording ends in: {}s'.format(int(recording_start_time + recording_length_sec - time.time()))
			)

			out_file.write(frame)

		frame_placeholder.empty()
		countdown_container.header('')
		out_file.release()
		process_video_file(recording_file_name)
	else:
		while cap.isOpened():
			ret, frame = cap.read()

			if frame is None:
				st.error("Failed to fetch frame. Retrying...")
				time.sleep(0.3)
				continue

			frame_placeholder.image(frame, channels='BGR')

	cap.release()


def process_stream():
	# cap = cv2.VideoCapture(-1, cv2.CAP_DSHOW)  # todo cap_dashboard doesn't not prevent from getting cv2 errors
	if 'cap' not in st.session_state:
		cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
		st.session_state['cap'] = cap
	else:
		cap = st.session_state['cap']

	# todo start recording button starts 5 seconds timer. After which it'll start recording for 10 seconds.
	#  Video will be saved and processed
	col1, col2, col3 = st.columns([1, 1, 1])
	st.error("features are not yet implemented")
	start_recording = col1.button("▶ Start Recording", use_container_width=True)
	stop_recording = col2.button("⏸ Stop Recording", use_container_width=True)
	reset_button_pressed = col3.button("🔄 Reset settings", use_container_width=True)

	frame_placeholder = st.empty()

	base_options = python.BaseOptions(model_asset_path=POSE_MODEL)
	options = vision.PoseLandmarkerOptions(
		base_options=base_options,
		output_segmentation_masks=True,
		running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
		result_callback=print_result
	)
	detector = vision.PoseLandmarker.create_from_options(options)

	while cap.isOpened():
		ret, frame = cap.read()

		if frame is None:
			st.error("Failed to fetch frame. Retrying...")
			time.sleep(1)
			continue

		mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
		detector.detect_async(mp_image, int(time.time() * 1000))

		# todo render frame without overlay
		frame_placeholder.image(frame, channels='BGR')

		if reset_button_pressed:
			reset_stream_settings(cap)

	logger.error("Stopping video stream...")
	cap.release()
	cv2.destroyAllWindows()


def simple_stream():
	if 'cap' not in st.session_state:
		cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
		st.session_state['cap'] = cap
	else:
		cap = st.session_state['cap']
	# cap = cv2.VideoCapture(0)  # todo cap_dashboard doesn't not prevent from getting cv2 errors

	# reset_button_pressed = st.button("Reset stream settings")
	frame_placeholder = st.empty()

	while cap.isOpened():
		ret, frame = cap.read()

		if frame is None:
			st.error("Failed to fetch frame. Retrying...")
			time.sleep(1)
			continue

		frame_placeholder.image(frame, channels='BGR')


def calculate_angle_3d(a, b, c):
	v1 = np.array([a[0] - b[0], a[1] - b[1], a[2] - b[2]])
	v2 = np.array([c[0] - b[0], c[1] - b[1], c[2] - b[2]])

	radians = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1))
	angle = np.abs(radians * 180 / np.pi)

	if angle > 180:
		angle = 360 - angle

	return angle


def reset_stream_settings(cap):
	# cap.set(cv2.CAP_PROP_SETTINGS, 1)  # todo - opens camera settings

	# cap.set(cv2.CAP_PROP_FPS, 30)
	# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
	# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

	cap.set(cv2.CAP_PROP_BRIGHTNESS, 24)
	cap.set(cv2.CAP_PROP_CONTRAST, 221)
	cap.set(cv2.CAP_PROP_SATURATION, 588)
	cap.set(cv2.CAP_PROP_SHARPNESS, 172)
	cap.set(cv2.CAP_PROP_GAMMA, 18)
	cap.set(cv2.CAP_PROP_GAIN, 1)

# model_pose = YOLO('yolov8n-pose.pt')

# model_pose_pred = model_pose(
# 	source='https://images.westend61.de/0000876136pw/side-view-of-athlete-riding-bicycle-on-road-CAVF19014.jpg',
# 	show=True,
# 	save=True
# )  # predict on an image

# logger.info(model_pose_pred)

# ================================================
