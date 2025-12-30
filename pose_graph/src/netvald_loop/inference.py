import tensorflow as tf
import cv2
import numpy as np
import time

class ImageInference:
    _model = None

    def __init__(self, model_path):
        if ImageInference._model is None:
            print("Loading model...")
            ImageInference._model = tf.saved_model.load(model_path)
            self.infer = ImageInference._model.signatures['serving_default']
        else:
            print("Model already loaded.")
    
    def infer_image(self, image_path):
        start_time = time.time()
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (752, 480))
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)
        image = np.expand_dims(image, axis=-1)

        result = self.infer(image=tf.convert_to_tensor(image))
        descriptor = result['descriptor'].numpy().astype(float)
        
        if descriptor.ndim > 1:
            descriptor = descriptor.flatten()
        
        descriptor_list = descriptor.tolist()

        elapsed_time = time.time() - start_time
        # print(f"Time to compute descriptor for {image_path}: {elapsed_time:.4f} seconds")
        
        return [float(x) if isinstance(x, (int, float)) else float(x[0]) for x in descriptor_list]
