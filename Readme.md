
# Similix

Similix is an image similarity search application built using deep learning models for feature extraction, FAISS for similarity search, and Streamlit for the UI. The application allows users to upload an image and find the most similar images from a pre-processed dataset containing various objects like bikes, cars, cats, dogs, flowers, horses, and humans.

---

## 🚀 Features

- Image upload functionality
- Display of the query image
- Showcase of top matching images
- Clickable results to reveal image paths

## 🛠️ Technologies Used

- **Python**: Core programming language
- **TensorFlow (ResNet50)**: For image feature extraction
- **FAISS (Facebook AI Similarity Search)**: To perform efficient similarity search
- **Streamlit**: To create an interactive web application

## ⚙️ Installation

1. Clone the repository:

```sh
https://github.com/ashishpatel8736/similix.git
```

2. Navigate to the project directory:

```sh
cd similix
```

3. Install the required dependencies:

```sh
pip install -r requirements.txt
```

4. Preprocessing Steps:

- i.Preprocess the Dataset:

```sh
python preprocess.py
```

- ii.Feature Extraction:

```sh
python extract_features.py
```

- iii.Build FAISS Index:

```sh
python build_faiss_index.py
```

5. Run the application:

```sh
streamlit run app.py
```

## 📁 Dataset

- **Dataset Link**: [Kaggle](https://www.kaggle.com/datasets/ashishpatel8736/similix-image-dataset)

The dataset used for this project includes images from the following categories:

- **Bikes**: 365 images
- **Cars**: 420 images
- **Cats**: 202 images
- **Dogs**: 202 images
- **Flowers**: 210 images
- **Horses**: 202 images
- **Humans**: 202 images

---

## 👤 Author  
**Ashish Patel**  
[![GitHub](icons8-github-50.png)](https://github.com/ashishpatel8736) | [![LinkedIn](https://img.icons8.com/ios-filled/50/0077b5/linkedin.png)](https://www.linkedin.com/in/ashishpatel8736)

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
