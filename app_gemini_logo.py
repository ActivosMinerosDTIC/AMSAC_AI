import os
import streamlit as st
from pathlib import Path
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from io import BytesIO
import time
import google.generativeai as genai
from dotenv import load_dotenv
import requests

# Cargar variables de entorno
load_dotenv()

# Configurar API key de Gemini
GEMINI_API_KEY = "AIzaSyCUbrUkz43cM5MIteanV6mVcPt3Anrg5pc"
genai.configure(api_key=GEMINI_API_KEY)

# Configuración de la página con logo
st.set_page_config(
    page_title="Sistema RAG con Gemini - AMSAC",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Función para cargar logo
def load_logo():
    """Carga el logo de AMSAC"""
    try:
        logo_url = "https://www.amsac.pe/wp-content/uploads/2023/08/cropped-logo-amsac-1024x739.png"
        response = requests.get(logo_url)
        if response.status_code == 200:
            return BytesIO(response.content)
        else:
            return None
    except Exception as e:
        return None

# Cache para modelos
@st.cache_resource
def get_embedding_model():
    """Carga y cachea el modelo de embeddings"""
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def get_gemini_model():
    """Carga y cachea el modelo Gemini"""
    try:
        # Usar el endpoint directo que funciona con curl
        model = genai.GenerativeModel('gemini-flash-latest')
        return model
    except Exception as e:
        st.error(f"Error cargando Gemini: {e}")
        return None

# Función simple para extraer texto de PDF
def extraer_texto_pdf(pdf_path):
    """Extrae texto de un archivo PDF usando PyMuPDF"""
    try:
        doc = fitz.open(pdf_path)
        texto = ""
        for page in doc:
            texto += page.get_text()
        doc.close()
        return texto
    except Exception as e:
        st.error(f"Error leyendo PDF: {e}")
        return ""

# Función para dividir texto en chunks
def dividir_texto(texto, chunk_size=1000, overlap=200):
    """Divide el texto en chunks superpuestos"""
    chunks = []
    for i in range(0, len(texto), chunk_size - overlap):
        chunk = texto[i:i + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
    return chunks

# Función para crear embeddings
def crear_embeddings(chunks, model):
    """Crea embeddings para los chunks de texto"""
    try:
        with st.spinner("🧠 Creando embeddings..."):
            embeddings = model.encode(chunks, show_progress_bar=True, batch_size=16)
        return embeddings
    except Exception as e:
        st.error(f"Error creando embeddings: {e}")
        return None

# Función para crear índice FAISS
def crear_indice_faiss(embeddings):
    """Crea un índice FAISS para búsqueda vectorial"""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    return index

# Función para buscar chunks relevantes
def buscar_similares(pregunta, model, index, chunks, k=3):
    """Busca chunks más similares a la pregunta"""
    try:
        start_time = time.time()
        query_embedding = model.encode([pregunta])
        distances, indices = index.search(query_embedding.astype('float32'), k)
        
        resultados = []
        for i, idx in enumerate(indices[0]):
            if idx < len(chunks):
                resultados.append({
                    'chunk': chunks[idx],
                    'distancia': distances[0][i],
                    'indice': idx
                })
        
        search_time = time.time() - start_time
        return resultados, search_time
    except Exception as e:
        st.error(f"Error en búsqueda: {e}")
        return [], 0

# Función para generar respuesta con Gemini
def generar_respuesta_gemini(pregunta, contexto, model):
    """Genera respuesta usando Gemini"""
    if model is None:
        return "❌ Modelo Gemini no disponible. Verifica tu API key."
    
    try:
        # Crear prompt para Gemini
        prompt = f"""Eres un asistente experto que responde preguntas basándose únicamente en el contexto proporcionado.

CONTEXTO DEL DOCUMENTO:
{contexto}

PREGUNTA DEL USUARIO: {pregunta}

INSTRUCCIONES IMPORTANTES:
1. Responde ÚNICAMENTE basándote en la información del contexto proporcionado
2. Si la información no está en el contexto, indícalo claramente
3. Sé claro, conciso y profesional
4. Responde en español
5. Usa un tono servicial y experto
6. Estructura tu respuesta de forma clara y organizada

Respuesta:"""

        # Generar respuesta con Gemini
        with st.spinner("💎 Gemini generando respuesta..."):
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=1000,
                    temperature=0.1,
                    top_p=0.8,
                    top_k=40,
                )
            )
            
            if response.text:
                return response.text.strip()
            else:
                return "No pude generar una respuesta. Intenta con otra pregunta."
                
    except Exception as e:
        st.error(f"Error con Gemini: {e}")
        return f"Error generando respuesta con Gemini: {str(e)}"

# Función de respaldo simple
def generar_respuesta_simple(pregunta, contexto):
    """Genera respuesta simple sin IA"""
    if not contexto.strip():
        return "No encontré información relevante en los documentos para responder tu pregunta."
    
    # Extraer frases clave del contexto
    frases = contexto.split('.')[:4]  # Primeras 4 frases
    frases_relevantes = [f.strip() for f in frases if f.strip() and len(f.strip()) > 20]
    
    if frases_relevantes:
        respuesta = f"Basado en los documentos, sobre '{pregunta}' encontré:\n\n"
        for i, frase in enumerate(frases_relevantes, 1):
            respuesta += f"• {frase}.\n"
        return respuesta
    else:
        return f"Encontré información relacionada con tu pregunta en el documento, pero el contexto es limitado."

# Interfaz principal
def main():
    # Header con logo de AMSAC
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        # Cargar y mostrar logo
        logo_bytes = load_logo()
        if logo_bytes:
            st.image(logo_bytes, width=150)
        else:
            st.write("🏢")
    
    with col2:
        st.markdown("---")
        st.title("💎 Sistema RAG con Gemini")
        st.markdown("### 🏢 **AMSAC - Asesoría y Servicios de Consultoría**")
        st.markdown("*Sistema Inteligente de Consulta de Documentos*")
    
    with col3:
        st.write("")  # Espacio vacío para balance
    
    st.markdown("---")
    
    # Verificar API key
    if GEMINI_API_KEY:
        st.success("🔑 API Key de Gemini configurada")
    else:
        st.error("🔑 API Key de Gemini no configurada")
    
    # Estado de la aplicación
    if 'chunks' not in st.session_state:
        st.session_state.chunks = []
    if 'index' not in st.session_state:
        st.session_state.index = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'documentos' not in st.session_state:
        st.session_state.documentos = []
    if 'gemini_model' not in st.session_state:
        st.session_state.gemini_model = None
    
    # Sidebar con branding de AMSAC
    with st.sidebar:
        # Logo en sidebar
        logo_bytes = load_logo()
        if logo_bytes:
            st.image(logo_bytes, width=120)
            st.markdown("---")
        
        st.header("💎 Configuración Gemini")
        
        # Estado de API
        if GEMINI_API_KEY:
            st.success("🔑 Gemini API Key OK")
        else:
            st.error("🔑 Gemini API Key faltante")
        
        # Botón para procesar documentos
        if st.button("📄 Procesar Documentos", type="primary"):
            procesar_documentos()
        
        # Botón para cargar Gemini
        if st.button("💎 Cargar Gemini", type="secondary"):
            if st.session_state.chunks:
                with st.spinner("Cargando Gemini..."):
                    model = get_gemini_model()
                    if model:
                        st.session_state.gemini_model = model
                        st.success("✅ Gemini cargado")
                    else:
                        st.error("❌ Error cargando Gemini")
            else:
                st.warning("⚠️ Procesa documentos primero")
        
        # Mostrar estado
        if st.session_state.chunks:
            st.success(f"✅ {len(st.session_state.chunks)} chunks")
            st.write(f"📄 Documentos: {len(st.session_state.documentos)}")
            for doc in st.session_state.documentos:
                st.write(f"  • {doc}")
        
        if st.session_state.gemini_model:
            st.success("💎 Gemini: Activo")
            st.info("🤖 Modelo: Gemini Flash Latest")
        else:
            st.warning("💎 Gemini: Inactivo")
        
        # Información del modelo
        st.subheader("ℹ️ Gemini Info")
        st.write("**Gemini Flash Latest** por Google")
        st.write("- 🆓 API key gratuita")
        st.write("- ⚡ Ultra rápido")
        st.write("- 🧠 Alta calidad")
        st.write("- 🌐 Requiere internet")
        
        # Footer AMSAC
        st.markdown("---")
        st.markdown("### 🏢 **AMSAC**")
        st.write("Asesoría y Servicios de Consultoría")
        st.write("Sistema RAG Inteligente")
    
    # Área principal
    if not st.session_state.chunks:
        st.info("👋 ¡Bienvenido al Sistema RAG con Gemini de AMSAC!")
        st.write("### 💎 Características:")
        st.write("- 💎 **Gemini Flash Latest** - IA de Google")
        st.write("- 🆓 **API key gratuita** - Sin costos")
        st.write("- ⚡ **Ultra rápido** - Respuestas en 1-2 segundos")
        st.write("- 🧠 **Alta calidad** - Respuestas expertas")
        st.write("- 🏢 **Branding AMSAC** - Identidad corporativa")
        
        st.write("### 📋 Pasos:")
        st.write("1. **Agrega archivos PDF** a la carpeta `documentos/`")
        st.write("2. **Presiona 'Procesar Documentos'**")
        st.write("3. **Carga Gemini** (con tu API key)")
        st.write("4. **Empieza a preguntar** - ¡Respuestas de Google!")
        
        # Mostrar archivos PDF actuales
        documentos_path = Path("documentos")
        if documentos_path.exists():
            pdf_files = list(documentos_path.glob("*.pdf"))
            if pdf_files:
                st.write("📁 **PDFs encontrados:**")
                for pdf in pdf_files:
                    st.write(f"  • {pdf.name}")
            else:
                st.write("📂 La carpeta `documentos/` está vacía")
    else:
        # Chat interface con Gemini y branding AMSAC
        st.header("💬 Preguntas con Gemini - AMSAC")
        
        # Input para pregunta
        pregunta = st.text_input("¿Qué quieres saber?", key="pregunta_input")
        
        if st.button("🔍 Preguntar a Gemini") and pregunta:
            if not GEMINI_API_KEY:
                st.error("❌ Configura tu API key de Gemini primero")
                return
            
            with st.spinner("🔍 Buscando información..."):
                resultados, search_time = buscar_similares(
                    pregunta, 
                    st.session_state.model, 
                    st.session_state.index, 
                    st.session_state.chunks
                )
                
                if resultados:
                    # Preparar contexto para Gemini
                    contexto = "\n\n".join([r['chunk'] for r in resultados])
                    
                    # Generar respuesta con Gemini
                    if st.session_state.gemini_model:
                        respuesta = generar_respuesta_gemini(pregunta, contexto, st.session_state.gemini_model)
                        tipo_respuesta = "💎 Respuesta de Gemini"
                    else:
                        respuesta = generar_respuesta_simple(pregunta, contexto)
                        tipo_respuesta = "📄 Respuesta Simple (Modo Respaldo)"
                    
                    # Mostrar respuesta
                    st.subheader(tipo_respuesta)
                    st.write(respuesta)
                    
                    # Mostrar métricas
                    if st.session_state.gemini_model:
                        st.success(f"💎 Respuesta generada por Gemini en {search_time:.2f}s")
                    
                    # Mostrar fuentes
                    with st.expander("📚 Fuentes utilizadas"):
                        for i, resultado in enumerate(resultados, 1):
                            st.write(f"**Fragmento {i}** (Relevancia: {1/(1+resultado['distancia']):.2f}):")
                            st.write(resultado['chunk'][:400] + "..." if len(resultado['chunk']) > 400 else resultado['chunk'])
                            st.write("---")
                else:
                    st.warning("No se encontró información relevante para tu pregunta.")
                    st.info("💡 Intenta con otras palabras o se más específico.")

def procesar_documentos():
    """Procesa todos los PDFs de la carpeta documentos"""
    documentos_path = Path("documentos")
    
    if not documentos_path.exists():
        st.error("La carpeta 'documentos/' no existe.")
        return
    
    pdf_files = list(documentos_path.glob("*.pdf"))
    
    if not pdf_files:
        st.warning("No se encontraron archivos PDF.")
        return
    
    with st.spinner(f"📄 Procesando {len(pdf_files)} documentos..."):
        # Obtener modelo cacheado
        model = get_embedding_model()
        
        todos_los_chunks = []
        st.session_state.documentos = []
        
        for pdf_file in pdf_files:
            st.write(f"📄 Procesando: {pdf_file.name}")
            
            # Extraer texto
            texto = extraer_texto_pdf(pdf_file)
            if texto:
                # Dividir en chunks
                chunks = dividir_texto(texto)
                todos_los_chunks.extend(chunks)
                st.session_state.documentos.append(pdf_file.name)
                st.write(f"   ✅ {len(chunks)} fragmentos creados")
        
        if todos_los_chunks:
            # Crear embeddings
            embeddings = crear_embeddings(todos_los_chunks, model)
            
            if embeddings is not None:
                # Crear índice
                st.write("📊 Creando índice de búsqueda...")
                index = crear_indice_faiss(embeddings)
                
                # Guardar en sesión
                st.session_state.chunks = todos_los_chunks
                st.session_state.index = index
                st.session_state.model = model
                
                st.success(f"✅ Procesados {len(todos_los_chunks)} fragmentos de {len(pdf_files)} documentos")
                st.info("💎 ¡Listo para preguntas con Gemini!")
            else:
                st.error("Error creando embeddings")
        else:
            st.error("No se pudo extraer texto de los documentos")

if __name__ == "__main__":
    main()
