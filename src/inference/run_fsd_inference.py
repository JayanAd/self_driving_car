import tensorflow as tf
import cv2
import numpy as np
import colorsys
from ultralytics import YOLO
from typing import List, Tuple
import concurrent.futures
import time

from keras import config
config.enable_unsafe_deserialization()

def dummy(x):
    return x
class SteeringAnglePredictor:
    def __init__(self, model_path: str):
        # Load TF2 Keras model
        print(f"Loading steering angle model from: {model_path}")
        # self.model = tf.keras.models.load_model(model_path, compile=False)
        try:
            # Strategy 1: Load with custom objects
            self.model = tf.keras.models.load_model(
                model_path, 
                custom_objects={'dummy': dummy},
                compile=False
            )
            print("Model loaded successfully with custom objects.")
        except Exception as e:
            print(f"Failed with custom objects: {e}")
            try:
                # Strategy 2: Load with safe_mode=False (Keras 3)
                self.model = tf.keras.models.load_model(
                    model_path, 
                    compile=False,
                    safe_mode=False
                )
                print("Model loaded successfully with safe_mode=False.")
            except Exception as e2:
                print(f"Failed with safe_mode: {e2}")
                # Strategy 3: Try loading weights only
                raise RuntimeError(
                    "Could not load model. Please try recreating the model "
                    "architecture and loading weights separately."
                )
        print("Model loaded successfully.")

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image before feeding to model
        """
        # Crop bottom part (usually road area)
        cropped = image[-150:, :, :]
        resized = cv2.resize(cropped, (200, 66))
        normalized = resized.astype(np.float32) / 255.0
        return np.expand_dims(normalized, axis=0)  # Add batch dimension

    def predict_angle(self, image: np.ndarray) -> float:
        """
        Predict steering angle in degrees
        """
        preprocessed = self.preprocess_image(image)
        prediction = self.model.predict(preprocessed, verbose=0)[0][0]  # scalar
        angle_deg = prediction * 90.0  # tanh output scaled to [-90°, +90°]
        return float(angle_deg)



class ImageSegmentation:
    def __init__(self, lane_model_path: str, object_model_path: str):
        self.lane_model = YOLO(lane_model_path)
        self.object_model = YOLO(object_model_path)
        self.colors = self._generate_colors(len(self.object_model.names))

    @staticmethod
    def _generate_colors(num_class: int) -> List[Tuple[int, int, int]]:
        colors = []
        for i in range(num_class):
            hue = i / num_class
            rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
            colors.append(tuple(int(x * 255) for x in rgb))
        return colors

    def process(self, img: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        overlay = img.copy()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            lane_future = executor.submit(self.lane_model.predict, img, conf=0.5)
            object_future = executor.submit(self.object_model.predict, img, conf=0.5)
            lane_results = lane_future.result()
            object_results = object_future.result()

        self._draw_lane_overlay(overlay, lane_results)
        self._draw_object_overlay(overlay, object_results)
        return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    def _draw_lane_overlay(self, overlay: np.ndarray, lane_results):
        for result in lane_results:
            if result.masks is None:
                continue
            for mask in result.masks.xy:
                points = np.int32([mask])
                cv2.fillPoly(overlay, points, (142, 235, 144))

    def _draw_object_overlay(self, overlay: np.ndarray, object_results):
        for result in object_results:
            if result.masks is None:
                continue
            for mask, box in zip(result.masks.xy, result.boxes):
                class_id = int(box.cls[0])
                color = self.colors[class_id]
                points = np.int32([mask])
                cv2.fillPoly(overlay, points, color)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                label = f"{self.object_model.names[class_id]} {box.conf[0]:.2f}"
                (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(overlay, (x1, y1 - 20), (x1 + label_width, y1), color, -1)
                cv2.putText(overlay, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)



class SelfDrivingCarSimulator:
    def __init__(self, steering_model: SteeringAnglePredictor, segmentation_model: ImageSegmentation,
                 data_path: str, img_path: str):
        self.steering_model = steering_model
        self.segmentation_model = segmentation_model
        self.data_path = data_path
        self.img = cv2.imread(img_path)
        self.smoothed_angle = 0.0

        if self.img is None:
            raise FileNotFoundError(f"Steering wheel image not found: {img_path}")

        self.rows, self.cols, _ = self.img.shape

    def start_simulation(self, frame_interval: float = 1 / 30):
        i = 0
        while True:
            start_time = time.time()
            full_image = cv2.imread(f"{self.data_path}/{i}.jpg")

            if full_image is None:
                print(f"Image {self.data_path}/{i}.jpg not found. Ending simulation.")
                break

            with concurrent.futures.ThreadPoolExecutor() as executor:
                seg_future = executor.submit(self.segmentation_model.process, full_image)
                angle_future = executor.submit(self.steering_model.predict_angle, full_image)

                segmented_image = seg_future.result()
                degrees = angle_future.result()

            self._update_display(degrees, segmented_image, full_image)
            i += 1

            elapsed = time.time() - start_time
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    def _update_display(self, degrees, segmented_image, full_image):
        print(f"Predicted steering angle: {degrees:.2f}°")
        # Smooth steering angle
        delta = degrees - self.smoothed_angle
        if delta != 0:
            self.smoothed_angle += 0.2 * (abs(delta) ** (2.0 / 3.0)) * np.sign(delta)

        M = cv2.getRotationMatrix2D((self.cols / 2, self.rows / 2), -self.smoothed_angle, 1)
        dst = cv2.warpAffine(self.img, M, (self.cols, self.rows))

        cv2.imshow("Original Frame", full_image)
        cv2.imshow("Segmented Frame", segmented_image)
        cv2.imshow("Steering Wheel", dst)



if __name__ == "__main__":
    steering_predictor = SteeringAnglePredictor("saved_models/regression_model/angle_regressor_model.keras")
    image_segmentation = ImageSegmentation(
        "saved_models/lane_segmentation_model/best_yolo11_lane_segmentation.pt",
        "saved_models/object_detection_model/yolo11s-seg.pt"
    )

    simulator = SelfDrivingCarSimulator(
        steering_model=steering_predictor,
        segmentation_model=image_segmentation,
        data_path="data/driving_dataset",
        img_path="data/steering_wheel_image.png"
    )

    simulator.start_simulation()
