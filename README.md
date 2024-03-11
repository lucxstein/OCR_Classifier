# Classificador de Documentos PDF com Tesseract-OCR 

## Overview

Este projeto visa classificar documentos em categorias específicas usando um modelo de Machine Learning (SVM) e vetorização TF-IDF. O código está estruturado para extrair texto de imagens contidas em documentos PDFs e treinar um modelo para classificá-los em diferentes categorias.

## Requisitos

Certifique-se de ter os seguintes requisitos instalados antes de executar o código:

- Python 3.x
- Bibliotecas Python: pdf2image, pytesseract, scikit-learn, joblib, gradio

Certifique-se de que o Tesseract OCR e Poppler estejam instalados em seu sistema e o caminho esteja configurado corretamente no código.

## Estrutura do Projeto

- **`app.py`**: Contém o código de deploy do modelo que pode ser testado no ambiente Spaces do HuggingFace: (https://huggingface.co/spaces/lucxstein/classificador_contrato_social)
- **`NB_Classifier.ipynb`**: Contém o Notebook com a estruturação e treinamento do modelo de ML.
- **`model.pkl`**: Arquivo pickle do modelo SVM.
- **`vectorizer.pkl`**: Arquivo pickle do vetorizador TF-IDF.
