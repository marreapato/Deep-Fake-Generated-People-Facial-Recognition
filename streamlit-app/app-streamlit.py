import streamlit as st
import numpy as np
import os
from model import image_pre,predict


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

    st.title('Fake Person AI Generated Face Identifier')
    # Render the badge using st.markdown
    st.markdown(badge_markdown, unsafe_allow_html=True)
    st.markdown(badge_markdown_lkd, unsafe_allow_html=True)
    image_data = load_image()

    if st.button('Run on image') and image_data:
        st.write('Calculating results...')
        getting_data = image_pre(image_data)
        result = predict(getting_data)
        st.write('Probability of not being an AI Generated Face: ', str(result*100)[0:5]+"%")

    st.markdown("![Alt Text](https://i.gifer.com/ZdPG.gif)")
    
if __name__ == '__main__':
    main()