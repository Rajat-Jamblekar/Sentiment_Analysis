# Sentiment Analysis Application

## Overview

This is a sentiment analysis application built using Flask and Hugging Face's Transformers library. The application allows users to analyze the sentiment of text input, including support for multiple languages. It utilizes state-of-the-art transformer models like BERT and RoBERTa for accurate sentiment classification.

## Features

- Analyze sentiment of text input or uploaded `.txt` files.
- Support for multiple languages using transformer models.
- User-friendly web interface for easy interaction.
- Detailed output including sentiment label and confidence scores.

## Technologies Used

- Python
- Flask
- Hugging Face Transformers
- NLTK (Natural Language Toolkit)
- HTML/CSS/JavaScript for frontend

## Installation
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

pip install -r requirements.txt

you will need to install nltk and download nltk punkt 

python app.py
### Prerequisites

Make sure you have Python 3.6 or higher installed on your machine. You can download it from [python.org](https://www.python.org/downloads/).

### Clone the Repository

```bash
git clone https://github.com/Rajat-Jamblekar/Sentiment_Analysis.git


License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
Hugging Face for providing the Transformers library.
NLTK for natural language processing tools.
The open-source community for their contributions and support