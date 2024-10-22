import cv2
import pickle
import numpy as np
import onnxruntime

def index_function(idx, max_val=192, max_idx=32):
    return (max_val) * ((idx/max_idx)**2)

class SupercomboConstants:
    """
    Openpilot supercombo model constants
    """
    # time and distance indices
    IDX_N = 33
    T_IDXS = [index_function(idx, max_val=10.0) for idx in range(IDX_N)]
    X_IDXS = [index_function(idx, max_val=192.0) for idx in range(IDX_N)]
    LEAD_T_IDXS = [0., 2., 4., 6., 8., 10.]
    LEAD_T_OFFSETS = [0., 2., 4.]
    META_T_IDXS = [2., 4., 6., 8., 10.]

    # model inputs constants
    MODEL_FREQ = 20
    FEATURE_LEN = 512
    FULL_HISTORY_BUFFER_LEN = 99
    HISTORY_BUFFER_LEN = 24
    DESIRE_LEN = 8
    TRAFFIC_CONVENTION_LEN = 2
    LAT_PLANNER_STATE_LEN = 4
    LATERAL_CONTROL_PARAMS_LEN = 2
    PREV_DESIRED_CURV_LEN = 1

    # model outputs constants
    FCW_THRESHOLDS_5MS2 = np.array([.05, .05, .15, .15, .15], dtype=np.float32)
    FCW_THRESHOLDS_3MS2 = np.array([.7, .7], dtype=np.float32)
    FCW_5MS2_PROBS_WIDTH = 5
    FCW_3MS2_PROBS_WIDTH = 2

    DISENGAGE_WIDTH = 5
    POSE_WIDTH = 6
    WIDE_FROM_DEVICE_WIDTH = 3
    SIM_POSE_WIDTH = 6
    LEAD_WIDTH = 4
    LANE_LINES_WIDTH = 2
    ROAD_EDGES_WIDTH = 2
    PLAN_WIDTH = 15
    DESIRE_PRED_WIDTH = 8
    LAT_PLANNER_SOLUTION_WIDTH = 4
    DESIRED_CURV_WIDTH = 1

    NUM_LANE_LINES = 4
    NUM_ROAD_EDGES = 2

    LEAD_TRAJ_LEN = 6
    DESIRE_PRED_LEN = 4

    PLAN_MHP_N = 5
    LEAD_MHP_N = 2
    PLAN_MHP_SELECTION = 1
    LEAD_MHP_SELECTION = 3

    FCW_THRESHOLD_5MS2_HIGH = 0.15
    FCW_THRESHOLD_5MS2_LOW = 0.05
    FCW_THRESHOLD_3MS2 = 0.7

    CONFIDENCE_BUFFER_LEN = 5
    RYG_GREEN = 0.01165
    RYG_YELLOW = 0.06157

    POLY_PATH_DEGREE = 4


class SupercomboModel:
    """Class for the Openpilot supercombo model"""

    def __init__(self, supercombo_path, supercombo_metadata_path):
        self.supercombo_model = onnxruntime.InferenceSession(supercombo_path, providers=['CUDAExecutionProvider'])

        with open(supercombo_metadata_path, 'rb') as f:
            model_metadata = pickle.load(f)
            self.output_slices = model_metadata['output_slices']

        # Model input initializations
        self.desire = np.zeros((1, 25, 8), dtype=np.float16)
        self.traffic_convention = np.array([[0,1]], dtype=np.float16)
        self.lateral_control_params = np.array([[0.2, 0.3]], dtype=np.float16)
        self.prev_desired_curv = np.zeros((25, 1), dtype=np.float16)
        self.full_featres = np.zeros((99, 512), dtype=np.float16)


        self.img_input_width = 512
        self.img_input_height = 256
        self.img_crop_size = 2/3
        self.wide_img_crop_size = None

        self.output_parser = SupercomboOutputParser()


    def predict(self, imgs, wide_imgs, prev_output):
        # Preprocess input images
        img1, img2 = imgs
        img1_yuv420 = self.convert_to_yuv420_6_channels(img1, self.img_crop_size)
        img2_yuv420 = self.convert_to_yuv420_6_channels(img2, self.img_crop_size)
        input_imgs = np.vstack((img1_yuv420, img2_yuv420))[np.newaxis, ...]

        wide_img1, wide_img2 = wide_imgs
        wide_img1_yuv420 = self.convert_to_yuv420_6_channels(wide_img1, self.wide_img_crop_size)
        wide_img2_yuv420 = self.convert_to_yuv420_6_channels(wide_img2, self.wide_img_crop_size)
        big_input_imgs = np.vstack((wide_img1_yuv420, wide_img2_yuv420))[np.newaxis, ...]

        # Update model inputs with with output data from the previous prediction
        if prev_output is not None:
            self.prev_desired_curv[:-1] = self.prev_desired_curv[1:]
            self.prev_desired_curv[-1] = prev_output['desired_curvature'][0, :]

            self.full_featres[:-1] = self.full_featres[1:]
            self.full_featres[-1] =  prev_output['hidden_state'][0, :]

        idxs = np.arange(-4,-100,-4)[::-1]
        features_buffer = self.full_featres[idxs][np.newaxis, ...]#.flatten()
        
        #start_time = time.time()
        # Run the model
        output = self.supercombo_model.run(None, {"input_imgs": input_imgs, 
                                                  "big_input_imgs": big_input_imgs, 
                                                  "desire": self.desire, 
                                                  "traffic_convention": self.traffic_convention, 
                                                  "lateral_control_params": self.lateral_control_params, 
                                                  "prev_desired_curv": self.prev_desired_curv[np.newaxis, ...], 
                                                  "features_buffer": features_buffer})
        #inference_time = time.time() - start_time
        #print(f"Inference Time: {inference_time:.4f}s")
        
        # Parse the model output
        parsed_output = self.output_parser.parse_outputs(self.slice_outputs(np.array(output[0][0])))
        
        return parsed_output
    
    def crop_images(self, image, crop_size):
        """Crops a smaller image from the center of the original image
        :param image: input image
        :param crop_size: crop size relative to the original image
        :return: cropped image
        """
        # Get image dimensions (height, width, number of channels)
        height, width, _ = image.shape

        # Calculate the size of the cropped region (half the original size)
        new_width = int(width * crop_size)
        new_height = int(height * crop_size)

        # Calculate the starting and ending points for the crop
        start_x = int((1 - crop_size) * width)
        start_y = int((1 - crop_size) * height)
        end_x = start_x + new_width
        end_y = start_y + new_height

        # Extract the crop from the center of the image
        cropped_image = image[start_y:end_y, start_x:end_x]

        return cropped_image
    

    def convert_to_yuv420_6_channels(self, image, crop_size):
        """Converts image to the YUV420 format with six channels as requred by
        Openpilot supercombo model

        :param image: input image
        :param crop_size: crop size relative to the original image
        :return: 6-channel image in YUV420 format
        """

        if crop_size is not None:
            image = self.crop_images(image, crop_size)
        
        # Resize the image to 256x512 as required
        img = cv2.resize(image, (self.img_input_width, self.img_input_height))
        
        # Convert the image to YUV format (YUV420p)
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420)

        # Extract Y, U, V channels
        height, width = img.shape[:2]
        y_plane = img_yuv[:height, :]  # Full resolution Y
        u_plane = img_yuv[height : height + height//4].reshape((-1, height//2, width//2))  # Half resolution U
        v_plane = img_yuv[height + height//4 :].reshape((-1, height//2, width//2))  # Half resolution V
        
        # Prepare the final 6-channel output
        output = np.zeros((6, height//2, width//2), dtype=np.uint8)
        
        # Channels 0,1,2,3 represent Y using different strides
        # https://github.com/commaai/openpilot/tree/master/selfdrive/modeld/models
        output[0] = y_plane[::2, ::2]  # Y[::2, ::2]
        output[1] = y_plane[::2, 1::2]  # Y[::2, 1::2]
        output[2] = y_plane[1::2, ::2]  # Y[1::2, ::2]
        output[3] = y_plane[1::2, 1::2]  # Y[1::2, 1::2]
        
        # Channel 4 represents half resolution U
        output[4] = u_plane
        
        # Channel 5 represents half resolution V
        output[5] = v_plane
        
        return output
    
    def slice_outputs(self, model_outputs):
        parsed_model_outputs = {k: model_outputs[np.newaxis, v] for k, v in self.output_slices.items()}
        return parsed_model_outputs
    

class SupercomboOutputParser:
    """
    Class for parsing the output of the Openpilot supercombo model
    """

    def __init__(self, ignore_missing=False):
        self.ignore_missing = ignore_missing

    def safe_exp(self, x, out=None):
        # -11 is around 10**14, more causes float16 overflow
        return np.exp(np.clip(x, -np.inf, 11), out=out)

    def sigmoid(self, x):
        return 1.0 / (1.0 + self.safe_exp(-x))
    
    def softmax(self, x, axis=-1):
        x -= np.max(x, axis=axis, keepdims=True)
        if x.dtype == np.float32 or x.dtype == np.float64:
            self.safe_exp(x, out=x)
        else:
            x = self.safe_exp(x)
        x /= np.sum(x, axis=axis, keepdims=True)
        return x

    def check_missing(self, outs, name):
        if name not in outs and not self.ignore_missing:
            raise ValueError(f"Missing output {name}")
        return name not in outs

    def parse_categorical_crossentropy(self, name, outs, out_shape=None):
        if self.check_missing(outs, name):
            return
        raw = outs[name]
        if out_shape is not None:
            raw = raw.reshape((raw.shape[0],) + out_shape)
        outs[name] = self.softmax(raw, axis=-1)

    def parse_binary_crossentropy(self, name, outs):
        if self.check_missing(outs, name):
            return
        raw = outs[name]
        outs[name] = self.sigmoid(raw)

    def parse_mdn(self, name, outs, in_N=0, out_N=1, out_shape=None):
        if self.check_missing(outs, name):
            return
        raw = outs[name]
        raw = raw.reshape((raw.shape[0], max(in_N, 1), -1))

        n_values = (raw.shape[2] - out_N) // 2
        pred_mu = raw[:, :, :n_values]
        pred_std = self.safe_exp(raw[:, :, n_values : 2 * n_values])

        if in_N > 1:
            weights = np.zeros((raw.shape[0], in_N, out_N), dtype=raw.dtype)
            for i in range(out_N):
                weights[:, :, i - out_N] = self.softmax(raw[:, :, i - out_N], axis=-1)

            if out_N == 1:
                for fidx in range(weights.shape[0]):
                    idxs = np.argsort(weights[fidx][:, 0])[::-1]
                    weights[fidx] = weights[fidx][idxs]
                    pred_mu[fidx] = pred_mu[fidx][idxs]
                    pred_std[fidx] = pred_std[fidx][idxs]
            full_shape = tuple([raw.shape[0], in_N] + list(out_shape))
            outs[name + "_weights"] = weights
            outs[name + "_hypotheses"] = pred_mu.reshape(full_shape)
            outs[name + "_stds_hypotheses"] = pred_std.reshape(full_shape)

            pred_mu_final = np.zeros((raw.shape[0], out_N, n_values), dtype=raw.dtype)
            pred_std_final = np.zeros((raw.shape[0], out_N, n_values), dtype=raw.dtype)
            for fidx in range(weights.shape[0]):
                for hidx in range(out_N):
                    idxs = np.argsort(weights[fidx, :, hidx])[::-1]
                    pred_mu_final[fidx, hidx] = pred_mu[fidx, idxs[0]]
                    pred_std_final[fidx, hidx] = pred_std[fidx, idxs[0]]
        else:
            pred_mu_final = pred_mu
            pred_std_final = pred_std

        if out_N > 1:
            final_shape = tuple([raw.shape[0], out_N] + list(out_shape))
        else:
            final_shape = tuple(
                [
                    raw.shape[0],
                ]
                + list(out_shape)
            )
        outs[name] = pred_mu_final.reshape(final_shape)
        outs[name + "_stds"] = pred_std_final.reshape(final_shape)

    def parse_outputs(self, outs):
        self.parse_mdn("plan", outs,
            in_N=SupercomboConstants.PLAN_MHP_N,
            out_N=SupercomboConstants.PLAN_MHP_SELECTION,
            out_shape=(SupercomboConstants.IDX_N, SupercomboConstants.PLAN_WIDTH),
        )
        self.parse_mdn("lane_lines", outs,
            in_N=0,
            out_N=0,
            out_shape=(
                SupercomboConstants.NUM_LANE_LINES,
                SupercomboConstants.IDX_N,
                SupercomboConstants.LANE_LINES_WIDTH,
            ),
        )
        self.parse_mdn("road_edges", outs,
            in_N=0,
            out_N=0,
            out_shape=(
                SupercomboConstants.NUM_ROAD_EDGES,
                SupercomboConstants.IDX_N,
                SupercomboConstants.LANE_LINES_WIDTH,
            ),
        )
        self.parse_mdn(
            "pose", outs, in_N=0, out_N=0, out_shape=(SupercomboConstants.POSE_WIDTH,)
        )
        self.parse_mdn("road_transform", outs,
            in_N=0,
            out_N=0,
            out_shape=(SupercomboConstants.POSE_WIDTH,),
        )
        self.parse_mdn("wide_from_device_euler", outs,
            in_N=0,
            out_N=0,
            out_shape=(SupercomboConstants.WIDE_FROM_DEVICE_WIDTH,),
        )
        self.parse_mdn("lead", outs,
            in_N=SupercomboConstants.LEAD_MHP_N,
            out_N=SupercomboConstants.LEAD_MHP_SELECTION,
            out_shape=(SupercomboConstants.LEAD_TRAJ_LEN, SupercomboConstants.LEAD_WIDTH),
        )
        if "lat_planner_solution" in outs:
            self.parse_mdn("lat_planner_solution", outs,
                in_N=0,
                out_N=0,
                out_shape=(
                    SupercomboConstants.IDX_N,
                    SupercomboConstants.LAT_PLANNER_SOLUTION_WIDTH,
                ),
            )
        if "desired_curvature" in outs:
            self.parse_mdn("desired_curvature", outs,
                in_N=0,
                out_N=0,
                out_shape=(SupercomboConstants.DESIRED_CURV_WIDTH,),
            )
        for k in ["lead_prob", "lane_lines_prob", "meta"]:
            self.parse_binary_crossentropy(k, outs)
        self.parse_categorical_crossentropy(
            "desire_state", outs, out_shape=(SupercomboConstants.DESIRE_PRED_WIDTH,)
        )
        self.parse_categorical_crossentropy("desire_pred", outs,
            out_shape=(
                SupercomboConstants.DESIRE_PRED_LEN,
                SupercomboConstants.DESIRE_PRED_WIDTH,
            ),
        )
        return outs