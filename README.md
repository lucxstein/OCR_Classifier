# Classificador de Documentos PDF com Tesseract-OCR 

## Overview

Este projeto é um classificador de documentos PDF que utiliza a biblioteca Tesseract para reconhecimento óptico de caracteres (OCR) e um modelo de machine learning (SVM) para classificar os documentos em categorias específicas.

## Requisitos

Certifique-se de ter os seguintes requisitos instalados antes de executar o código:

- Python 3.x
- Bibliotecas Python: pdf2image, pytesseract, scikit-learn, joblib, gradio e + (bibliotecas necessárias no arquivo 'requirements.txt')

Certifique-se de que o Tesseract OCR e Poppler (necessário para funcionamento da biblioteca pdf2image) estejam instalados em seu sistema e o caminho esteja configurado corretamente no código.

## Estrutura do Projeto

- **`app.py`**: Contém o código de deploy do modelo que pode ser testado no ambiente Spaces do HuggingFace: (https://huggingface.co/spaces/lucxstein/classificador_contrato_social)
  
- **`NB_model_classifier.ipynb`**: Contém o Notebook com a estruturação e treinamento do modelo de ML.
  
- **`model.pkl`**: Arquivo pickle do modelo SVM.
  
- **`vectorizer.pkl`**: Arquivo pickle do vetorizador TF-IDF.
