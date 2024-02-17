import streamlit as st
import numpy as np
import os
from model import image_pre,predict
from lime_explainer import explanation


def load_image():
    uploaded_file = st.file_uploader(label='Pick an image to test')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return image_data


def main():
    # Markdown code for the badge
    badge_markdown = "[![License](https://img.shields.io/github/license/marreapato/Deep-Fake-Generated-People-Facial-Recognition.svg?logo=github&style=social)](https://github.com/marreapato/Deep-Fake-Generated-People-Facial-Recognition)"
    badge_markdown_lkd = "[![Linkedin](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lucas-rabelo-ab58492a1/?locale=en_US)"

    st.title('🤖 Fake Person AI Generated Face Identifier v1.1.0 🕵️‍♂️')
    # Render the badge using st.markdown
    st.markdown(badge_markdown, unsafe_allow_html=True)
    st.markdown(badge_markdown_lkd, unsafe_allow_html=True)
    image_data = load_image()

    if st.button('Run on image') and image_data:
        st.write('Calculating results...')
        getting_data = image_pre(image_data)
        result = predict(getting_data)
        st.write('Probability of not being an AI Generated Face: ', str(result*100)[0:5]+"%")

        explanation_fig = explanation(getting_data)
        
        st.write("Explanation (Green color had a positive contribution to the prediction whereas red color had a negative contribution)")
        st.pyplot(explanation_fig)
        
        
          
    st.markdown("![Alt Text](https://i.gifer.com/ZdPG.gif)")
    
    # Description
    st.write("This web app is your go-to tool for distinguishing real faces from AI-generated ones! 👀")
    st.write("Using state-of-the-art Convolutional Neural Network (CNN) technology, we've trained our model on a robust dataset of 140,000 images. 💡")
    st.write("Just upload a picture of a person's face, and our AI will analyze it, providing you with the probability of whether the face is real or fake. 📸")

# What's New
    st.write("🚀 **What's New in Version 1.1.0:**")
    st.write("- XAI Implementation Lime Explanations")
    st.write("- Softmax Activation Function.")

# Future Updates
    st.write("🔜 **Future Updates:**")
    st.write("- We're already working on improving our model!")
    st.write("- Transfer learning will be employed to enhance accuracy.")
    st.write("- More extensive training data will be utilized for better performance.")

# Disclaimer
    st.write("*This tool when used together with human knowledge can help identify ai-generated people. Results may vary. (The model does not capture the effect of state-of-art models such as StableDiffusion, but it works well with samples from [This Person Does Not Exist's Website](https://thispersondoesnotexist.com/))* 📝")
    
    
if __name__ == '__main__':
    main()
