import gradio as gr
import joblib
import pytesseract
from pdf2image import convert_from_path

# Carregar o modelo SVM treinado
svm_model = joblib.load('modelo.pkl')

# Carregar o vetorizador TF-IDF
vectorizer = joblib.load('vectorizer.pkl')

# Função para extrair texto de uma imagem usando OCR
def extract_text_from_image(image_path):
    return pytesseract.image_to_string(image_path)

# Função para classificar o documento PDF
def classify_pdf(file):
    # Converter o PDF em uma lista de imagens
    images = convert_from_path(file.name)
    # Extrair texto de cada imagem
    text = [extract_text_from_image(image) for image in images]
    # Vetorizar o texto usando o vetorizador TF-IDF
    text_vectorized = vectorizer.transform(text)
    # Classificar o texto usando o modelo SVM
    predicted_class = svm_model.predict(text_vectorized)
    return predicted_class[0]

# Interface do Gradio
app = gr.Interface(fn=classify_pdf, 
                   inputs="file",
                   outputs="label",
                   title="Classificador de Documento Societário")

app.launch()