# 📚 Sistema RAG - Asistente de Documentos

Un sistema completo de Retrieval-Augmented Generation (RAG) 100% gratuito y local para consultar tus documentos PDF.

## 🌟 Características

- **100% Gratuito** - Sin costos de API ni suscripciones
- **Totalmente Local** - Tus documentos nunca salen de tu computadora
- **Interfaz Amigable** - Interfaz web moderna con Streamlit
- **Procesamiento Automático** - Carga y procesa PDFs automáticamente
- **Memoria Conversacional** - Mantiene el contexto de la conversación
- **Búsqueda Inteligente** - Encuentra información relevante en segundos

## 🚀 Inicio Rápido

### 1. Activar Entorno Virtual
```bash
source .venv/bin/activate
```

### 2. Iniciar Interfaz Gráfica
```bash
streamlit run app.py
```

### 3. Usar el Sistema
1. Abre tu navegador en la URL que aparece (generalmente `http://localhost:8501`)
2. Agrega archivos PDF a la carpeta `documentos/`
3. Presiona "🚀 Inicializar Sistema" en la barra lateral
4. ¡Empieza a chatear con tus documentos!

## 📁 Estructura del Proyecto

```
├── documentos/           # Coloca tus PDFs aquí
├── repositorio_faiss/   # Índice vectorial (se crea automáticamente)
├── app.py              # Interfaz gráfica principal
├── main.py             # API FastAPI (opcional)
├── requirements.txt    # Dependencias
└── .env               # Variables de entorno
```

## 🛠️ Tecnologías Utilizadas

- **Streamlit** - Interfaz gráfica web
- **LangChain** - Framework para RAG
- **TinyLlama** - Modelo de lenguaje local (1.1B parámetros)
- **Sentence Transformers** - Embeddings multilingües
- **FAISS** - Almacenamiento vectorial eficiente
- **PyPDF** - Procesamiento de PDFs

## 💡 Ejemplos de Uso

Una vez inicializado el sistema, puedes hacer preguntas como:

- "¿Qué es la norma ISO 27001?"
- "¿Cuáles son los requisitos para la certificación?"
- "Explica el proceso de auditoría"
- "¿Qué controles de seguridad se mencionan?"

## 🔧 Configuración Avanzada

### API FastAPI (Opcional)
Si prefieres una API en lugar de interfaz gráfica:

```bash
uvicorn main:app --reload
```

La API estará disponible en `http://localhost:8000`

### Endpoints Disponibles
- `GET /` - Estado del sistema
- `GET /health` - Verificación de salud
- `POST /chat` - Consultas RAG

## 📋 Requisitos

- Python 3.9+
- Espacio en disco: ~2GB para modelos
- Memoria RAM: 4GB+ recomendado

## 🐛 Solución de Problemas

### Si el modelo no carga:
- Asegúrate de tener suficiente RAM
- Espera unos minutos (la primera descarga tarda)

### Si no encuentra documentos:
- Verifica que los PDFs estén en la carpeta `documentos/`
- Asegúrate de que los archivos tengan extensión `.pdf`

### Si la interfaz no se inicia:
- Verifica que el entorno virtual esté activado
- Reinstala las dependencias: `pip install -r requirements.txt`

## 🤝 Contribuir

¡Siéntete libre de mejorar este proyecto!

## 📄 Licencia

MIT License - Libre para uso personal y comercial.
