# AI-Tools-Assignment

Sure! Here's a **well-structured README** for your **MNIST CNN model** and **Amazon Reviews NLP model**, written clearly so anyone can understand and run your projects.

You can save it as `README.md`.

---

# 📚 Machine Learning & Deep Learning Projects

This repository contains **two projects**:

1. **MNIST Digit Classifier (CNN)** – Deep Learning with TensorFlow
2. **Amazon Reviews NER & Sentiment Analysis** – NLP with spaCy

Both projects are ready to run in Python and Google Colab.

---

## 1️⃣ MNIST Digit Classifier (CNN)

**Goal:**
Classify handwritten digits (0–9) with **>95% accuracy** using a Convolutional Neural Network.

**Files:**

* `mnist_model.h5` – Trained CNN model
* `mnist_cnn.ipynb` – Jupyter notebook with training, evaluation, and visualization
* `app.py` – Streamlit app to deploy your model

**Requirements:**

```bash
pip install tensorflow streamlit pillow numpy matplotlib
```

**How to Run:**

### Training & Evaluation

1. Open `mnist_cnn.ipynb` in Jupyter or Colab
2. Run all cells to train and evaluate the CNN

### Streamlit Deployment

1. Place `mnist_model.h5` in the same folder as `app.py`
2. Run the Streamlit app:

```bash
streamlit run app.py
```

3. Upload a handwritten digit image to get a prediction
4. Optionally, deploy on [Streamlit Cloud](https://streamlit.io/cloud) for a public URL

**Notes:**

* Input images should be 28x28 pixels and grayscale
* The app also supports uploading JPG/PNG files

---

## 2️⃣ Amazon Reviews NER & Sentiment Analysis

**Goal:**
Perform **Named Entity Recognition (NER)** to extract product names and brands, and **analyze sentiment** (positive/negative/neutral) from Amazon product reviews.

**Files:**

* `reviews-dataset.csv` – 250,000+ sample reviews for testing
* `amazon_reviews_nlp.ipynb` – Jupyter notebook with NER, sentiment analysis, and visualization

**Requirements:**

```bash
pip install spacy pandas
python -m spacy download en_core_web_sm
```

**How to Run:**

1. Open `amazon_reviews_nlp.ipynb` in Jupyter or Colab
2. Make sure `reviews-dataset.csv` is uploaded
3. Run all cells:

   * Extract **brands** and **products** using spaCy
   * Perform **sentiment analysis** using a simple rule-based approach
   * View sample results and distribution of sentiments

**Notes:**

* Sentiment is detected using predefined keywords (positive/negative)
* You can extend the keywords or add custom rules for more accurate detection

---

## 🧠 Ethics Considerations

* **MNIST Model:** May perform poorly on handwritten digits outside the training distribution
* **Reviews NLP:** Rules may miss slang, sarcasm, or uncommon brands
* Mitigation: Use **data augmentation**, **custom NER rules**, and **fairness checks**

---

## 📊 Outputs

* **MNIST CNN:** Accuracy > 95%, confusion matrix, and sample predictions
* **Reviews NLP:** Extracted brands/products, sentiment label for each review, and sample output visualization

---

## 🏷️ Credits

* MNIST dataset: [Yann LeCun, MNIST](http://yann.lecun.com/exdb/mnist/)
* Amazon Reviews dataset: Simulated with realistic product reviews
* Tools: TensorFlow, spaCy, Streamlit, Pandas, NumPy, Matplotlib

---

## 📌 Quick Tips

* **Colab users:** Upload `reviews-dataset.csv` and `mnist_model.h5` to Colab before running notebooks
* **Streamlit users:** Make sure `app.py` and model file are in the same directory
* **Extendability:** Add more products/brands to spaCy rules or augment MNIST with rotated/noisy digits for better performance

---

