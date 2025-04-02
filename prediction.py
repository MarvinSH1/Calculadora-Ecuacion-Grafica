import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from flask import Blueprint, request, jsonify

# Crear un blueprint para la parte de predicción
prediction_bp = Blueprint('prediction_bp', __name__)
records = []  # Lista para almacenar los registros de predicción

@prediction_bp.route('/add_record', methods=['POST'])
def add_record():
    data = request.json
    if not data:
        return jsonify({"error": "No se recibieron datos"}), 400
    records.append(data)
    return jsonify({'message': 'Registro agregado', 'records': records})

@prediction_bp.route('/predict', methods=['POST'])
def predict():
    if len(records) < 2:
        return jsonify({"error": "Se requieren al menos dos registros para predecir."}), 400

    try:
        # Extraer datos numéricos de los registros
        aspirantes = [int(record["aspirantes"]) for record in records]
        ingresados = [int(record["ingresados"]) for record in records]
        egresados = [int(record["egresados"]) for record in records]
    except Exception as e:
        return jsonify({"error": str(e)})

    # Generar un array de índices para cada registro
    años = np.arange(len(aspirantes))
    # Suponemos que el siguiente período es el siguiente índice
    futuro_años = np.append(años, años[-1] + 1)
    
    # Ajuste lineal para predecir tendencias
    pred_aspirantes = np.poly1d(np.polyfit(años, aspirantes, 1))(futuro_años)
    pred_ingresados = np.poly1d(np.polyfit(años, ingresados, 1))(futuro_años)
    pred_egresados = np.poly1d(np.polyfit(años, egresados, 1))(futuro_años)
    
    # Graficar las tendencias
    plt.figure(figsize=(8, 5))
    plt.plot(futuro_años, pred_aspirantes, label='Aspirantes', marker='o')
    plt.plot(futuro_años, pred_ingresados, label='Ingresados', marker='s')
    plt.plot(futuro_años, pred_egresados, label='Egresados', marker='^')
    plt.legend()
    plt.grid()
    plt.xlabel('Período')
    plt.ylabel('Cantidad de estudiantes')
    plt.title('Tendencia de Aspirantes, Ingresados y Egresados')
    
    # Guardar la gráfica en un buffer y codificar en base64
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return jsonify({"plot_url": plot_url})
