# 🤖 AMSAC AI - Sistema de Consulta de Documentos

Sistema inteligente de consulta de documentos PDF usando Google Gemini AI con interfaz web moderna y ligera.

## 🌟 Características

- **IA Avanzada** - Usa Google Gemini para respuestas inteligentes
- **Interfaz Moderna** - Diseño web profesional con Bootstrap 5
- **Procesamiento Automático** - Carga y procesa PDFs automáticamente
- **FastAPI** - API REST rápida y eficiente
- **Ligero** - Mínimas dependencias para fácil instalación
- **Fácil de Usar** - Chat interactivo en tiempo real

## 🚀 Inicio Rápido

### 1. Instalar Dependencias
```bash
pip install -r requirements.txt
```

### 2. Configurar API Key
Edita el archivo `.env` y agrega tu API key de Google Gemini:
```
GEMINI_API_KEY=tu_api_key_aqui
```

### 3. Iniciar Aplicación
```bash
python app.py
```

### 4. Usar el Sistema
1. Abre tu navegador en `http://localhost:9000`
2. Agrega archivos PDF a la carpeta `documentos/`
3. ¡Empieza a chatear con tus documentos!

## 📁 Estructura del Proyecto

```
├── .venv/              # Entorno virtual Python
├── documentos/         # Coloca tus PDFs aquí
├── templates/          # Plantillas HTML
│   └── chat.html      # Interfaz de chat
├── static/            # Archivos estáticos (CSS, JS, imágenes)
├── app.py             # Aplicación principal FastAPI
├── requirements.txt   # Dependencias mínimas del proyecto
└── .env              # Variables de entorno (API keys)
```

## 🛠️ Tecnologías Utilizadas

- **FastAPI** - Framework web moderno y rápido
- **Google Gemini AI** - Modelo de IA para respuestas inteligentes
- **PyMuPDF** - Procesamiento de documentos PDF
- **Bootstrap 5** - Framework CSS para interfaz moderna
- **Uvicorn** - Servidor ASGI de alto rendimiento

## 💡 Ejemplos de Uso

Una vez iniciado el sistema, puedes hacer preguntas como:

- "¿Qué es la norma ISO 27001?"
- "¿Cuáles son los requisitos para la certificación?"
- "Explica el proceso de auditoría"
- "¿Qué controles de seguridad se mencionan?"

## 🔧 Endpoints API Disponibles

- `GET /` - Interfaz de chat principal
- `GET /health` - Estado del sistema
- `POST /chat` - Enviar consultas
- `GET /status` - Información detallada del sistema

## 📋 Requisitos

- Python 3.9+
- Google Gemini API Key (configurar en `.env`)
- Memoria RAM: 1GB+ (versión ligera)

## 🔑 Configuración API Key

1. Obtén tu API key en [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Edita el archivo `.env` y agrega:
```
GEMINI_API_KEY=tu_api_key_aqui
```

## 🐛 Solución de Problemas

### Si no encuentra documentos:
- Verifica que los PDFs estén en la carpeta `documentos/`
- Asegúrate de que los archivos tengan extensión `.pdf`

### Si hay error de API:
- Verifica que tu API key de Gemini sea válida
- Revisa el archivo `.env`
- Asegúrate de que la API key no esté marcada como filtrada

### Si la aplicación no inicia:
- Verifica que el entorno virtual esté activado
- Reinstala las dependencias: `pip install -r requirements.txt`

### Si las respuestas tardan mucho:
- El sistema tiene un timeout de 45 segundos para evitar bloqueos
- Intenta con preguntas más específicas

## 📄 Licencia

MIT License - Libre para uso personal y comercial.
