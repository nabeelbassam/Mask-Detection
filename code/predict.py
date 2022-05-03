from utils import Utils
import numpy as np


class Predictor:
    def __init__(self, image_path):
        self.model = Utils.get_model()
        self.image = Utils.load_image(image_path, show=True)

    def predict(self):
        pred = self.model.predict(self.image)

        if np.argmax(pred, axis=1) == 1:
            print("Without Mask")
        else:
            print("With Mask")

