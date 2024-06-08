import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

class PredictionPipeline:
    def __init__(self):
        self.model = load_model('artifacts/model/model.h5')
        self.target_size = (181, 181)  # Adjusted target size

    def predict(self, img_path):
        img = image.load_img(img_path, target_size=self.target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Rescale the image array
        prediction = self.model.predict(img_array)
        class_label = 'Lesion' if prediction[0][0] > 0.5 else 'Normal'
        if class_label == 'Lesion':
            class_label='Normal'
        else:
            class_label='Lesion'
        return class_label
