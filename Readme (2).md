# 🤖 Proyecto NLP RoboCup — Tokenización, Clasificación y Planificación (BERT + T5)

Este repositorio contiene el desarrollo de un sistema de comprensión y planificación de lenguaje natural aplicado al dominio **RoboCup@Home**, centrado en la generación de comandos estructurados para robots de servicio (por ejemplo, Pepper o Darwin-OP).  

El proyecto combina **clasificación de tokens (NER)** con **modelos de planificación secuencial (T5)**, permitiendo convertir instrucciones humanas en estructuras comprensibles por el robot.

---

## 📁 Estructura del repositorio

```
.
├── bert_token_classification_model/
│   └── checkpoint-339/              → Pesos del modelo BERT fine-tuneado para clasificación de tokens
├── t5_plan_model/
│   └── ...                          → Pesos del modelo T5 para planificación
├── Ruta.ipynb                       → Notebook principal: flujo de entrenamiento y evaluación de modelos
├── Tokenizar.ipynb                  → Notebook de tokenización y preprocesamiento de dataset
├── dataset.jsonl                    → Dataset base (formato JSON Lines)
├── dataset_for_t5.csv               → Dataset adaptado para entrenamiento del T5
├── dataset_with_slots.jsonl         → Dataset enriquecido con “slots” semánticos
├── dataset_with_slots_and_plan.jsonl→ Dataset con slots + plan completo
├── .gitattributes                   → Configuración de Git LFS (para subir modelos grandes)
├── requirements.txt                 → Dependencias principales del proyecto
└── README.md                        → Este documento
```

---

## ⚙️ Requisitos e instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/JuanRivera182003/tu-repo.git
cd tu-repo
```

### 2. Instalar dependencias

Crea el entorno virtual y luego instala las librerías necesarias:

```bash
python -m venv venv
source venv/bin/activate   # En Linux / macOS
venv\Scripts\activate      # En Windows

pip install -r requirements.txt
```

Si aún no tienes el archivo `requirements.txt`, crea uno con este contenido:

```text
torch>=2.0.0
transformers>=4.38.0
datasets>=2.18.0
pandas
numpy
scikit-learn
tqdm
```

*(opcional)* si planeas visualizar o ejecutar notebooks:
```text
jupyter
notebook
```

---

## 📘 Descripción técnica

### 1. Tokenización y Clasificación (BERT)
- Modelo base: `bert-base-cased`
- Tarea: **Token Classification / NER**
- Entrenado en los datasets JSONL con etiquetas de comandos RoboCup.
- Objetivo: identificar entidades como **acción**, **objeto**, **ubicación**, **agente**, etc.

### 2. Planificación Secuencial (T5)
- Modelo base: `t5-small`
- Tarea: **Conditional Text Generation**
- Entrada: texto anotado con los “slots” generados por el modelo BERT.
- Salida: **plan estructurado** para el robot (ej. `go(kitchen); grasp(cup); return(table)`).

---

## 🧩 Flujo de trabajo

1. **Tokenizar y etiquetar texto** con `Tokenizar.ipynb`
   - Limpieza y división del dataset.
   - Entrenamiento o carga del modelo BERT.
   - Predicción de etiquetas por token.

2. **Generar dataset estructurado** (`dataset_with_slots.jsonl`)
   - A partir de las etiquetas anteriores, construir pares entrada–salida para el modelo T5.

3. **Entrenar planificador (T5)** con `Ruta.ipynb`
   - Entrenamiento del modelo T5.
   - Evaluación y pruebas de generación de secuencias.

4. **Ejecutar inferencia final**
   ```python
   from transformers import AutoTokenizer, AutoModelForTokenClassification, T5Tokenizer, T5ForConditionalGeneration

   # === Tokenización ===
   tok = AutoTokenizer.from_pretrained("bert_token_classification_model/checkpoint-339")
   model = AutoModelForTokenClassification.from_pretrained("bert_token_classification_model/checkpoint-339")

   inputs = tok("Bring me the red cup from the kitchen", return_tensors="pt")
   outputs = model(**inputs)
   # Procesar logits para obtener etiquetas

   # === Planificación ===
   t5_tok = T5Tokenizer.from_pretrained("t5_plan_model")
   t5_model = T5ForConditionalGeneration.from_pretrained("t5_plan_model")

   plan_in = "action: bring; object: cup; color: red; location: kitchen"
   plan_ids = t5_model.generate(t5_tok(plan_in, return_tensors="pt").input_ids)
   print(t5_tok.decode(plan_ids[0], skip_special_tokens=True))
   ```

---

## 🤖 Referencia RoboCup — Generador de Comandos

Este trabajo se basa en el **Generador de Comandos NLP** del repositorio oficial de RoboCup@Home, disponible en:

🔗 [https://github.com/RoboCupAtHome/command-generator](https://github.com/RoboCupAtHome/command-generator)

Allí se definió la estructura base de **slots** y **plantillas de comandos** que fueron adaptadas y enriquecidas en este proyecto.

---

## 📊 Resultados esperados

- **Modelo BERT**: precisión de etiquetado > 90% en entidades principales.  
- **Modelo T5**: generación coherente de planes en formato lógico ejecutable.  
- **Integración**: permite pasar directamente de texto natural → estructura semántica → plan ejecutable por el robot.

---

## 🧾 Créditos y licencia

**Autor principal:** Juan Andrés Rivera Gutiérrez  
**Institución:** Universidad Santo Tomás – USTA 🇨🇴 / Universidad Andrés Bello – UNAB 🇨🇱  
**Año:** 2025  
**Licencia:** MIT License (ver archivo `LICENSE` si se incluye)

Basado en modelos y librerías de:
- [🤗 Hugging Face Transformers](https://huggingface.co/transformers/)
- [RoboCup@Home Command Generator](https://github.com/RoboCupAtHome/command-generator)

---

## 💡 Notas adicionales

- Los notebooks fueron limpiados de metadatos para visualizar correctamente en GitHub.  
- Los modelos grandes usan [Git LFS](https://git-lfs.github.com/).  
- Puedes agregar una carpeta `scripts/` si deseas incluir pipelines de entrenamiento automatizados.  
- Se recomienda incluir:
  - `.gitignore`
  - `requirements.txt`
  - `LICENSE`
  - y opcionalmente `config.json` con parámetros de los modelos.

---

## ✅ Próximos pasos

1. Subir el archivo `requirements.txt` al repo.  
2. (Opcional) Crear un `LICENSE` con tipo MIT.  
3. Agregar badges de estado (por ejemplo: versión de Python, modelo usado, licencia).  
4. Si deseas despliegue automático en **Hugging Face Spaces** o **Colab**, puedo dejarte el `app.py` y `Space README.md` listos.
