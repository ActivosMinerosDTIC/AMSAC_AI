from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
from pathlib import Path
import fitz  # PyMuPDF
import requests
import logging
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configuración
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurar API key de Gemini desde .env
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyB4GXfKMabtd7ioO9tcYVjN63vkWmgcRak")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent"

# Estado global
class GlobalState:
    def __init__(self):
        self.documentos_texto = {}
        self.gemini_model = None
        self.initialized = False

state = GlobalState()

def initialize_system():
    """Inicializa el sistema automáticamente"""
    if state.initialized:
        return True
    
    try:
        logger.info("🚀 Iniciando sistema AMSAC AI...")
        
        # Cargar modelo Gemini
        try:
            # Probar conexión con la API REST de Gemini
            try:
                headers = {
                    'Content-Type': 'application/json',
                    'X-goog-api-key': GEMINI_API_KEY
                }
                data = {
                    "contents": [{
                        "parts": [{"text": "test"}]
                    }]
                }
                
                logger.info(f"🔑 Usando API key: {GEMINI_API_KEY[:20]}...")
                logger.info(f"🌐 URL: {GEMINI_API_URL}")
                
                response = requests.post(GEMINI_API_URL, headers=headers, json=data, timeout=10)
                
                logger.info(f"📡 Respuesta HTTP: {response.status_code}")
                
                if response.status_code == 200:
                    state.gemini_model = "gemini-flash-latest"
                    logger.info("✅ Modelo Gemini gemini-flash-latest cargado exitosamente")
                else:
                    logger.error(f"❌ Error cargando Gemini: {response.status_code}")
                    logger.error(f"❌ Respuesta: {response.text[:500]}")
                    # No fallar, continuar sin IA
                    logger.warning("⚠️ Continuando sin modelo de IA...")
                
            except Exception as e:
                logger.error(f"❌ Error cargando Gemini: {e}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error cargando Gemini: {e}")
            return False
        
        # Procesar documentos PDF
        documentos_path = Path("documentos")
        if documentos_path.exists():
            pdf_files = list(documentos_path.glob("*.pdf"))
            
            if pdf_files:
                logger.info(f"📄 Encontrados {len(pdf_files)} archivos PDF:")
                
                for pdf_file in pdf_files:
                    logger.info(f"   📄 Procesando: {pdf_file.name}")
                    
                    try:
                        # Extraer texto del PDF
                        doc = fitz.open(pdf_file)
                        texto = ""
                        num_paginas = len(doc)
                        
                        for page_num, page in enumerate(doc):
                            page_text = page.get_text()
                            if page_text.strip():
                                texto += f"\n\n--- DOCUMENTO: {pdf_file.name} | PÁGINA {page_num + 1} ---\n\n"
                                texto += page_text
                        
                        doc.close()
                        
                        if texto.strip():
                            state.documentos_texto[pdf_file.name] = {
                                'content': texto,
                                'pages': num_paginas,
                                'size': pdf_file.stat().st_size
                            }
                            logger.info(f"   ✅ {pdf_file.name}: {num_paginas} páginas, {len(texto)} caracteres")
                    
                    except Exception as e:
                        logger.error(f"   ❌ Error procesando {pdf_file.name}: {e}")
                
                if state.documentos_texto:
                    state.initialized = True
                    logger.info(f"✅ Sistema inicializado con {len(state.documentos_texto)} documentos")
                    
                    # Mostrar resumen
                    for nombre, info in state.documentos_texto.items():
                        size_mb = info['size'] / (1024 * 1024)
                        logger.info(f"   📋 {nombre}: {info['pages']} páginas, {size_mb:.2f} MB")
                    
                    return True
        
        logger.warning("⚠️ No se encontraron documentos PDF en la carpeta 'documentos/'")
        return False
        
    except Exception as e:
        logger.error(f"❌ Error inicializando sistema: {e}")
        return False

# Lifespan handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Iniciando AMSAC AI API...")
    initialize_system()
    yield
    # Shutdown
    logger.info("Apagando AMSAC AI API...")

# Crear aplicación FastAPI
app = FastAPI(
    title="AMSAC AI API",
    description="Sistema Inteligente de Consulta de Documentos",
    version="1.0.0",
    lifespan=lifespan
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Montar archivos estáticos
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception as e:
    logger.warning(f"No se pudo montar directorio static: {e}")

# Modelos Pydantic
from typing import List, Dict, Any

class ChatRequest(BaseModel):
    question: str
    history: List[Dict[str, Any]] = []

class ChatResponse(BaseModel):
    answer: str
    source_documents: List[Dict[str, Any]] = []

class HealthResponse(BaseModel):
    status: str
    initialized: bool
    documents_count: int
    chunks_count: int

def generar_respuesta_gemini(pregunta, contexto):
    """Genera respuesta usando Gemini API REST"""
    if not state.gemini_model:
        return "Modelo Gemini no disponible"
    
    try:
        contexto_limitado = contexto[:12000]
        prompt = f"""Eres un asistente experto de AMSAC que responde preguntas basándose únicamente en el contexto proporcionado.

CONTEXTO COMPLETO DEL DOCUMENTO:
{contexto_limitado}

PREGUNTA DEL USUARIO: {pregunta}

INSTRUCCIONES IMPORTANTES:
1. Responde ÚNICAMENTE basándote en la información del contexto proporcionado
2. Si la información no está en el contexto, indícalo claramente
3. Sé claro, detallado y profesional
4. Responde en español
5. Usa un tono servicial y experto
6. Estructura tu respuesta de forma clara y organizada
7. Incluye todos los detalles relevantes del contexto
8. **IMPORTANTE**: Usa formato markdown para hacer la respuesta más visual:
   - Usa **negritas** para puntos importantes
   - Usa • para viñetas
   - Usa emojis relevantes (📍, 🏢, 💻, 👥, ✅)
   - Usa numeración para listas ordenadas
   - Usa separadores --- para secciones
   - Usa > para citas importantes

Respuesta completa y detallada:"""

        headers = {
            'Content-Type': 'application/json',
            'X-goog-api-key': GEMINI_API_KEY
        }
        data = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }
        
        response = requests.post(GEMINI_API_URL, headers=headers, json=data, timeout=45)
        
        if response.status_code == 200:
            result = response.json()
            if 'candidates' in result and len(result['candidates']) > 0:
                respuesta = result['candidates'][0]['content']['parts'][0]['text'].strip()
                
                # Limpiar prefijos no deseados
                prefijos_a_eliminar = [
                    "Respuesta completa y detallada:",
                    "Respuesta:",
                    "Basado en el contexto,",
                    "Según el documento,",
                ]
                
                for prefijo in prefijos_a_eliminar:
                    if respuesta.startswith(prefijo):
                        respuesta = respuesta[len(prefijo):].strip()
                
                return respuesta
            else:
                return "No pude generar una respuesta completa. Intenta con otra pregunta."
        else:
            logger.error(f"Error API Gemini: {response.status_code} - {response.text}")
            return f"Error generando respuesta: {response.status_code}"
            
    except requests.Timeout:
        logger.error("Error generando respuesta: timeout de Gemini")
        return "La generación tomó demasiado tiempo. Intenta con una pregunta más específica."
        
    except Exception as e:
        logger.error(f"Error generando respuesta: {e}")
        return f"Error generando respuesta: {str(e)}"

# Endpoints
@app.get("/", response_class=FileResponse)
async def read_index():
    """Sirve la página de chat"""
    return FileResponse("templates/chat.html")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Verifica el estado del sistema"""
    if not state.initialized:
        initialize_system()
    
    return HealthResponse(
        status="healthy" if state.initialized else "initializing",
        initialized=state.initialized,
        documents_count=len(state.documentos_texto),
        chunks_count=sum(len(doc['content']) for doc in state.documentos_texto.values()) if state.documentos_texto else 0
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Procesa una pregunta del usuario"""
    logger.info(f"Recibida pregunta: {request.question}")
    
    if not state.initialized:
        logger.warning("Sistema no inicializado, intentando inicializar...")
        if not initialize_system():
            logger.error("No se pudo inicializar el sistema")
            # En lugar de fallar, devolver un mensaje informativo
            return ChatResponse(
                answer="⚠️ **Sistema no disponible temporalmente**\n\nEl sistema de IA no está disponible porque la API key de Gemini necesita ser actualizada. Por favor, crea una nueva API key en [Google AI Studio](https://aistudio.google.com/app/apikey) y actualízala en el archivo `.env`.",
                source_documents=[]
            )
    
    if not request.question.strip():
        logger.warning("Pregunta vacía recibida")
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía")
    
    try:
        logger.info(f"Generando respuesta para: {request.question}")
        
        # Combinar todo el contexto de los documentos
        contexto = "\n\n".join([
            f"=== DOCUMENTO: {nombre} ===\n{info['content']}"
            for nombre, info in state.documentos_texto.items()
        ])
        
        logger.info(f"Contexto generado: {len(contexto)} caracteres")
        
        # Generar respuesta
        respuesta = generar_respuesta_gemini(request.question, contexto)
        logger.info(f"Respuesta generada: {len(respuesta)} caracteres")
        
        # Preparar fuentes
        fuentes = [
            {
                'content': info['content'][:500] + "...",  # Primeros 500 caracteres
                'source': nombre,
                'pages': info['pages'],
                'relevance': 1.0
            }
            for nombre, info in state.documentos_texto.items()
        ]
        
        logger.info("Enviando respuesta")
        return ChatResponse(
            answer=respuesta,
            source_documents=fuentes
        )
        
    except Exception as e:
        logger.error(f"Error en chat: {e}", exc_info=True)
        return ChatResponse(
            answer=f"Error procesando la solicitud: {str(e)}",
            source_documents=[]
        )

@app.get("/status")
async def get_status():
    """Obtiene estado detallado del sistema"""
    try:
        documentos_info = []
        for nombre, info in state.documentos_texto.items():
            size_mb = info['size'] / (1024 * 1024)
            documentos_info.append({
                'name': nombre,
                'pages': info['pages'],
                'size_mb': round(size_mb, 2)
            })
        
        return {
            "initialized": state.initialized,
            "documents": documentos_info,
            "total_documents": len(documentos_info),
            "gemini_available": state.gemini_model is not None
        }
    except Exception as e:
        logger.error(f"Error en status: {e}")
        return {
            "initialized": state.initialized,
            "documents": [],
            "total_documents": 0,
            "gemini_available": state.gemini_model is not None,
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=9000, reload=True)
