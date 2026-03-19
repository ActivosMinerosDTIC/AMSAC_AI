from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
from pathlib import Path
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import google.generativeai as genai
import logging
from contextlib import asynccontextmanager

# Configuración
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurar API key de Gemini
GEMINI_API_KEY = "AIzaSyCUbrUkz43cM5MIteanV6mVcPt3Anrg5pc"
genai.configure(api_key=GEMINI_API_KEY)

# Estado global
class GlobalState:
    def __init__(self):
        self.chunks = []
        self.index = None
        self.model = None
        self.gemini_model = None
        self.documentos = []
        self.initialized = False

state = GlobalState()

# Función de inicialización
def initialize_system():
    """Inicializa el sistema automáticamente"""
    if state.initialized:
        return True
    
    try:
        logger.info("Iniciando sistema AMSAC AI...")
        
        # Cargar modelos con manejo de errores
        try:
            state.model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Modelo de embeddings cargado")
        except Exception as e:
            logger.error(f"❌ Error cargando modelo de embeddings: {e}")
            return False
        
        try:
            state.gemini_model = genai.GenerativeModel('gemini-flash-latest')
            logger.info("✅ Modelo Gemini cargado")
        except Exception as e:
            logger.error(f"❌ Error cargando Gemini: {e}")
            return False
        
        # Procesar documentos
        documentos_path = Path("documentos")
        if documentos_path.exists():
            pdf_files = list(documentos_path.glob("*.pdf"))
            
            if pdf_files:
                logger.info(f"Encontrados {len(pdf_files)} archivos PDF:")
                for pdf_file in pdf_files:
                    logger.info(f"  📄 {pdf_file.name}")
                
                todos_los_chunks = []
                documentos_info = []
                
                for pdf_file in pdf_files:
                    logger.info(f"📄 Procesando: {pdf_file.name}")
                    
                    # Extraer texto
                    doc = fitz.open(pdf_file)
                    texto = ""
                    num_paginas = len(doc)
                    
                    for page_num, page in enumerate(doc):
                        page_text = page.get_text()
                        if page_text.strip():
                            # Agregar metadata del documento y página
                            texto += f"\n\n--- DOCUMENTO: {pdf_file.name} | PÁGINA {page_num + 1} ---\n\n"
                            texto += page_text
                    
                    doc.close()
                    
                    if texto.strip():
                        # Dividir en chunks más grandes para mejor contexto
                        chunks = []
                        chunk_size = 1200  # Aumentado para más contexto
                        overlap = 200
                        
                        for i in range(0, len(texto), chunk_size - overlap):
                            chunk = texto[i:i+chunk_size]
                            if chunk.strip():
                                chunks.append({
                                    'content': chunk,
                                    'source': pdf_file.name,
                                    'chunk_index': len(chunks),
                                    'total_chunks': 0  # Se actualizará después
                                })
                        
                        # Actualizar total_chunks
                        for chunk in chunks:
                            chunk['total_chunks'] = len(chunks)
                        
                        todos_los_chunks.extend(chunks)
                        documentos_info.append({
                            'name': pdf_file.name,
                            'pages': num_paginas,
                            'chunks': len(chunks),
                            'size': pdf_file.stat().st_size
                        })
                        
                        logger.info(f"  ✅ {len(chunks)} fragmentos creados de {num_paginas} páginas")
                
                if todos_los_chunks:
                    logger.info(f"📊 Creando embeddings para {len(todos_los_chunks)} fragmentos...")
                    try:
                        embeddings = state.model.encode([chunk['content'] for chunk in todos_los_chunks], batch_size=16, show_progress_bar=True)
                        
                        logger.info("🔍 Creando índice FAISS...")
                        dimension = embeddings.shape[1]
                        state.index = faiss.IndexFlatL2(dimension)
                        state.index.add(embeddings.astype('float32'))
                        
                        # Guardar estado
                        state.chunks = todos_los_chunks
                        state.documentos = documentos_info
                        state.initialized = True
                        
                        logger.info(f"✅ Sistema inicializado:")
                        logger.info(f"   📄 {len(pdf_files)} documentos procesados")
                        logger.info(f"   🔍 {len(todos_los_chunks)} fragmentos totales")
                        logger.info(f"   📊 Índice FAISS creado con {dimension} dimensiones")
                        
                        # Mostrar resumen de documentos
                        for doc_info in documentos_info:
                            size_mb = doc_info['size'] / (1024 * 1024)
                            logger.info(f"   📋 {doc_info['name']}: {doc_info['pages']} páginas, {doc_info['chunks']} fragmentos, {size_mb:.2f} MB")
                        
                        return True
                    except Exception as e:
                        logger.error(f"❌ Error creando embeddings: {e}")
                        return False
        
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
app.mount("/static", StaticFiles(directory="static"), name="static")

# Estado global
class GlobalState:
    def __init__(self):
        self.chunks = []
        self.index = None
        self.model = None
        self.gemini_model = None
        self.documentos = []
        self.initialized = False

state = GlobalState()

# Modelos Pydantic
class ChatRequest(BaseModel):
    question: str
    history: list = []

class ChatResponse(BaseModel):
    answer: str
    source_documents: list = []

class HealthResponse(BaseModel):
    status: str
    initialized: bool
    documents_count: int
    chunks_count: int

# Funciones de procesamiento
def initialize_system():
    """Inicializa el sistema automáticamente"""
    if state.initialized:
        return True
    
    try:
        logger.info("Iniciando sistema AMSAC AI...")
        
        # Cargar modelos
        state.model = SentenceTransformer('all-MiniLM-L6-v2')
        state.gemini_model = genai.GenerativeModel('gemini-flash-latest')
        
        # Procesar documentos
        documentos_path = Path("documentos")
        if documentos_path.exists():
            pdf_files = list(documentos_path.glob("*.pdf"))
            
            if pdf_files:
                todos_los_chunks = []
                documentos_nombres = []
                
                for pdf_file in pdf_files:
                    logger.info(f"Procesando: {pdf_file.name}")
                    
                    # Extraer texto
                    doc = fitz.open(pdf_file)
                    texto = ""
                    for page in doc:
                        texto += page.get_text()
                    doc.close()
                    
                    if texto:
                        # Dividir en chunks
                        chunks = []
                        for i in range(0, len(texto), 800):
                            chunk = texto[i:i+800]
                            if chunk.strip():
                                chunks.append(chunk)
                        
                        todos_los_chunks.extend(chunks)
                        documentos_nombres.append(pdf_file.name)
                        logger.info(f"  ✅ {len(chunks)} fragmentos creados")
                
                if todos_los_chunks:
                    # Crear embeddings
                    logger.info("Creando embeddings...")
                    embeddings = state.model.encode(todos_los_chunks, batch_size=16)
                    
                    # Crear índice
                    logger.info("Creando índice FAISS...")
                    dimension = embeddings.shape[1]
                    state.index = faiss.IndexFlatL2(dimension)
                    state.index.add(embeddings.astype('float32'))
                    
                    # Guardar estado
                    state.chunks = todos_los_chunks
                    state.documentos = documentos_nombres
                    state.initialized = True
                    
                    logger.info(f"✅ Sistema inicializado: {len(todos_los_chunks)} fragmentos de {len(pdf_files)} documentos")
                    return True
        
        logger.warning("No se encontraron documentos para procesar")
        return False
        
    except Exception as e:
        logger.error(f"Error inicializando sistema: {e}")
        return False

def buscar_similares(pregunta, k=8):  # Aumentado para buscar en más documentos
    """Busca chunks similares a la pregunta"""
    if not state.initialized or not state.index:
        logger.error("Sistema no inicializado o índice no disponible")
        return []
    
    try:
        query_embedding = state.model.encode([pregunta])
        distances, indices = state.index.search(query_embedding.astype('float32'), k)
        
        resultados = []
        documentos_encontrados = set()
        
        for i, idx in enumerate(indices[0]):
            if idx < len(state.chunks):
                chunk = state.chunks[idx]
                
                # Manejar tanto formato antiguo (solo contenido) como nuevo (con metadata)
                if isinstance(chunk, str):
                    # Formato antiguo - solo contenido
                    chunk_data = {
                        'content': chunk,
                        'source': 'Documento desconocido',
                        'chunk_index': i,
                        'total_chunks': len(state.chunks)
                    }
                else:
                    # Formato nuevo - con metadata
                    chunk_data = chunk
                
                resultados.append({
                    'content': chunk_data['content'],
                    'distance': float(distances[0][i]),
                    'index': int(idx),
                    'source': chunk_data.get('source', 'Documento desconocido'),
                    'chunk_index': chunk_data.get('chunk_index', i),
                    'total_chunks': chunk_data.get('total_chunks', len(state.chunks))
                })
                documentos_encontrados.add(chunk_data.get('source', 'Documento desconocido'))
                logger.info(f"Resultado {i}: {chunk_data.get('source', 'Documento desconocido')} (distancia={distances[0][i]:.4f})")
            else:
                logger.warning(f"Índice {idx} fuera de rango (chunks: {len(state.chunks)})")
        
        logger.info(f"🔍 Búsqueda completada: {len(resultados)} resultados de {len(documentos_encontrados)} documentos")
        return resultados
    except Exception as e:
        logger.error(f"❌ Error en búsqueda: {e}")
        return []

def generar_respuesta_gemini(pregunta, contexto):
    """Genera respuesta usando Gemini"""
    if not state.gemini_model:
        return "Modelo Gemini no disponible"
    
    try:
        # Prompt más completo y estructurado con formato
        prompt = f"""Eres un asistente experto de AMSAC que responde preguntas basándose únicamente en el contexto proporcionado.

CONTEXTO COMPLETO DEL DOCUMENTO:
{contexto}

PREGUNTA DEL USUARIO: {pregunta}

INSTRUCCIONES IMPORTANTES:
1. Responde ÚNICAMENTE basándote en la información del contexto proporcionado
2. Si la información no está en el contexto, indícalo claramente
3. Sé claro, detallado y profesional
4. Responde en español
5. Usa un tono servicial y experto
6. Estructura tu respuesta de forma clara y organizada
7. Incluye todos los detalles relevantes del contexto
8. No limites la longitud de la respuesta
9. **IMPORTANTE**: Usa formato markdown para hacer la respuesta más visual:
   - Usa **negritas** para puntos importantes
   - Usa • para viñetas
   - Usa emojis relevantes (📍, 🏢, 💻, 👥, ✅)
   - Usa numeración para listas ordenadas
   - Usa separadores --- para secciones
   - Usa > para citas importantes

Respuesta completa y detallada:"""

        response = state.gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=2000,  # Aumentado para respuestas más largas
                temperature=0.1,
                top_p=0.8,
                top_k=40,
            )
        )
        
        # Procesar respuesta completa
        if response.text:
            respuesta = response.text.strip()
            
            # Limpiar posibles prefijos no deseados
            prefijos_a_eliminar = [
                "Respuesta completa y detallada:",
                "Respuesta:",
                "Basado en el contexto,",
                "Según el documento,",
                "De acuerdo con los Términos de Referencia proporcionados",
                "la respuesta es la siguiente:"
            ]
            
            for prefijo in prefijos_a_eliminar:
                if respuesta.startswith(prefijo):
                    respuesta = respuesta[len(prefijo):].strip()
            
            return respuesta
        else:
            return "No pude generar una respuesta completa. Intenta con otra pregunta."
            
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
        documents_count=len(state.documentos) if state.documentos else 0,
        chunks_count=len(state.chunks)
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Procesa una pregunta del usuario"""
    if not state.initialized:
        if not initialize_system():
            raise HTTPException(status_code=503, detail="Sistema no disponible")
    
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía")
    
    try:
        # Buscar información relevante
        resultados = buscar_similares(request.question)
        
        if not resultados:
            return ChatResponse(
                answer="No encontré información relevante para tu pregunta. Intenta con otras palabras.",
                source_documents=[]
            )
        
        # Generar respuesta
        contexto = "\n\n".join([r['content'] for r in resultados])  # Usar contexto completo
        respuesta = generar_respuesta_gemini(request.question, contexto)
        
        # Preparar fuentes (contenido completo con metadata)
        fuentes = []
        fuentes_por_documento = {}
        
        for resultado in resultados:
            fuente = {
                'content': resultado['content'],  # Contenido completo sin truncar
                'source': resultado['source'],
                'chunk_index': resultado['chunk_index'],
                'total_chunks': resultado['total_chunks'],
                'relevance': 1 / (1 + resultado['distance'])
            }
            fuentes.append(fuente)
            
            # Agrupar por documento para mejor visualización
            if resultado['source'] not in fuentes_por_documento:
                fuentes_por_documento[resultado['source']] = []
            fuentes_por_documento[resultado['source']].append(fuente)
        
        return ChatResponse(
            answer=respuesta,
            source_documents=fuentes
        )
        
    except Exception as e:
        logger.error(f"Error en chat: {e}")
        raise HTTPException(status_code=500, detail="Error procesando la solicitud")

@app.get("/status")
async def get_status():
    """Obtiene estado detallado del sistema"""
    try:
        documentos_info = []
        if state.documentos:
            # Si es una lista de nombres (versión antigua)
            if isinstance(state.documentos[0], str):
                for doc_name in state.documentos:
                    documentos_info.append({
                        'name': doc_name,
                        'pages': 'N/A',
                        'chunks': 'N/A',
                        'size_mb': 'N/A'
                    })
            # Si es una lista de diccionarios (versión nueva)
            else:
                for doc in state.documentos:
                    size_mb = doc.get('size', 0) / (1024 * 1024)
                    documentos_info.append({
                        'name': doc.get('name', 'Desconocido'),
                        'pages': doc.get('pages', 0),
                        'chunks': doc.get('chunks', 0),
                        'size_mb': round(size_mb, 2)
                    })
        
        return {
            "initialized": state.initialized,
            "documents": documentos_info,
            "total_documents": len(documentos_info),
            "total_chunks": len(state.chunks),
            "gemini_available": state.gemini_model is not None,
            "model_loaded": state.model is not None
        }
    except Exception as e:
        logger.error(f"Error en status: {e}")
        return {
            "initialized": state.initialized,
            "documents": [],
            "total_documents": 0,
            "total_chunks": len(state.chunks),
            "gemini_available": state.gemini_model is not None,
            "model_loaded": state.model is not None,
            "error": str(e)
        }

# Inicialización al arrancar - eliminado, ahora usa lifespan

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_web:app", host="0.0.0.0", port=8000, reload=True)
