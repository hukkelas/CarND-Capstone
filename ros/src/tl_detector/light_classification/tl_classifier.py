from styx_msgs.msg import TrafficLight
from light_classification.train import get_model, idx2color
import os
import numpy as np
from keras.models import load_model
from keras import backend as K

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        self.classifier = get_model()
        fp = os.path.dirname(os.path.realpath(__file__))
        fp = os.path.join(fp, "model_weights_v3.h5")

        self.classifier = load_model(fp)
        self.classifier._make_predict_function()
        self.graph = K.tf.get_default_graph()
    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        with self.graph.as_default():

            image = image.astype(np.float32) / 128 - 1
            image = image[np.newaxis, :, :, :]
            cls = self.classifier.predict(image, batch_size=1)
            print(cls.shape)
            cls = np.argmax(cls.squeeze())
            cls = idx2color[cls]
        
        
        if cls == "red":
            return TrafficLight.RED
        if cls == "yellow":
            return TrafficLight.YELLOW
        if cls == "green":
            return TrafficLight.GREEN
        return TrafficLight.UNKNOWN
