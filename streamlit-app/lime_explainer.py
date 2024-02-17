import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import keras
import numpy as np

base = 'D:/gitrepos/Deep-Fake-Generated-People-Facial-Recognition/streamlit-app'
model = keras.models.load_model(f'{base}/model_dfake-face_softmax.h5')

def explanation(img_data):
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(img_data.reshape((128,128,3)), model.predict,top_labels=2, hide_color=0, num_samples=1000)
    

    temp_1, mask_1 = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=2, hide_rest=True)
    temp_2, mask_2 = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=2, hide_rest=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,15))
    ax1.imshow(mark_boundaries(temp_1, mask_1))
    ax2.imshow(mark_boundaries(temp_2, mask_2))
    ax1.axis('off')
    ax2.axis('off')
    
    return fig