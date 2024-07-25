# BERT Sentiment Analysis

This repository provides an introductory practice project for training a BERT model for sentiment analysis using the IMDb dataset and uploading the trained model to Hugging Face.

## Table of Contents

- [Introduction](#introduction)
- [Setup](#setup)
- [Training the Model](#training-the-model)
- [Uploading to Hugging Face](#uploading-to-hugging-face)
- [Usage](#usage)
- [Contributing](#contributing)
- [Contact Information](#contact-information)

## Introduction

This project demonstrates how to fine-tune a pre-trained BERT model for sentiment analysis using the IMDb dataset. The trained model is then uploaded to Hugging Face, making it available for others to use.

## Setup

To run this project, you need to install the necessary dependencies. Use the following command to install them:

```bash
pip install transformers datasets torch huggingface_hub
```

## Training the Model

The training process is contained in the `train_and_upload_bert_sentiment_model.ipynb` notebook. Follow these steps to train the model:

1. Clone this repository and navigate to the directory:
    ```bash
    git clone https://github.com/your-username/bert-sentiment-analysis.git
    cd bert-sentiment-analysis
    ```

2. Open the `train_and_upload_bert_sentiment_model.ipynb` notebook using Jupyter Notebook or JupyterLab.

3. Run all cells in the notebook to train the model on the IMDb dataset. The training process includes:
    - Loading the IMDb dataset
    - Tokenizing the text data
    - Fine-tuning the pre-trained BERT model
    - Saving the trained model

## Uploading to Hugging Face

After training the model, you can upload it to Hugging Face using the `huggingface_hub` library. Follow these steps:

1. Log in to your Hugging Face account:
    ```bash
    huggingface-cli login
    ```

2. Use the following script in the notebook to upload your model:
    ```python
    from huggingface_hub import HfApi, HfFolder

    model_dir = "./sentiment-model"
    model_name = "sentiment-analysis-bert"

    api = HfApi()
    token = HfFolder.get_token()
    api.upload_folder(
        folder_path=model_dir,
        path_in_repo=".",
        repo_id="your-username/sentiment-analysis-bert",
        repo_type="model",
        token=token
    )
    ```

Replace `your-username` with your Hugging Face username and `sentiment-analysis-bert` with your desired model name.

## Usage

Once the model is uploaded to Hugging Face, you can use it in your projects. Here's an example of how to load and use the model for sentiment analysis:

```python
from transformers import pipeline

# Load the model from Hugging Face
model_name = "your-username/sentiment-analysis-bert"
classifier = pipeline("sentiment-analysis", model=model_name)

# Use the model to classify text
result = classifier("This movie was amazing!")
print(result)
```

## Contributing

Your contributions are welcome! If you have ideas for improvements or new features:

1. **Fork the Repository**
2. **Create a Branch:**
   ```bash
   git checkout -b feature-branch
   ```
3. **Commit Changes:**
   ```bash
   git commit -am 'Add new feature: description'
   ```
4. **Push to Branch:**
   ```bash
   git push origin feature-branch
   ```
5. **Submit a Pull Request**

## Contact

- **Email:** [jayakuma006@mymail.sim.edu.sg](mailto:jayakuma006@mymail.sim.edu.sg)
- **LinkedIn:** [Akilesh Jayakumar on LinkedIn](https://www.linkedin.com/in/akileshjayakumar/)
- **GitHub:** [Akilesh Jayakumar on GitHub](https://github.com/akileshjayakumar)
