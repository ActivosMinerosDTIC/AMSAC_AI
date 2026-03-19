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

# Configuración de la página estilo Google
st.set_page_config(
    page_title="AMSAC AI - Sistema Inteligente",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS estilo Google Material Design
st.markdown("""
<style>
/* Google Material Design Styles */
.main-header {
    background: linear-gradient(135deg, #4285f4 0%, #34a853 100%);
    padding: 1rem 2rem;
    border-radius: 8px;
    margin-bottom: 2rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.google-search-box {
    background: white;
    border: 1px solid #e0e0e0;
    border-radius: 24px;
    padding: 12px 20px;
    font-size: 16px;
    width: 100%;
    box-shadow: 0 1px 6px rgba(32,33,36,.28);
    transition: box-shadow 0.2s;
}

.google-search-box:focus {
    outline: none;
    box-shadow: 0 1px 6px rgba(32,33,36,.38);
    border-color: #4285f4;
}

.google-button {
    background: #f8f9fa;
    border: 1px solid #f8f9fa;
    border-radius: 4px;
    color: #3c4043;
    font-family: 'Google Sans',Roboto,arial,sans-serif;
    font-size: 14px;
    margin: 11px 4px;
    padding: 0 16px;
    line-height: 27px;
    height: 36px;
    min-width: 54px;
    text-align: center;
    cursor: pointer;
    user-select: none;
}

.google-button:hover {
    box-shadow: 0 1px 1px rgba(0,0,0,.1);
    background-color: #f8f9fa;
    border: 1px solid #dadce0;
    color: #202124;
}

.result-card {
    background: white;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 16px;
    box-shadow: 0 1px 6px rgba(32,33,36,.28);
    border: 1px solid #e0e0e0;
}

.result-title {
    color: #1a0dab;
    font-size: 20px;
    font-weight: 400;
    margin-bottom: 4px;
    cursor: pointer;
}

.result-title:hover {
    text-decoration: underline;
}

.result-snippet {
    color: #4d5156;
    font-size: 14px;
    line-height: 1.4;
}

.result-url {
    color: #202124;
    font-size: 14px;
    margin-bottom: 4px;
}

.sidebar-google {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 16px;
}

.status-indicator {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 8px;
}

.status-active {
    background: #34a853;
}

.status-inactive {
    background: #ea4335;
}

.status-warning {
    background: #fbbc04;
}

.google-footer {
    background: #f2f2f2;
    border-top: 1px solid #e4e4e4;
    padding: 15px 30px;
    font-size: 14px;
    color: #70757a;
    margin-top: 2rem;
}

/* Animaciones */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.fade-in {
    animation: fadeIn 0.5s ease-out;
}

/* Logo styling */
.logo-container {
    text-align: center;
    margin-bottom: 2rem;
}

.logo-img {
    max-width: 200px;
    height: auto;
    border-radius: 8px;
}

/* Search results container */
.search-results {
    max-width: 652px;
    margin: 0 auto;
}

/* Chat bubble style */
.chat-bubble {
    background: #f8f9fa;
    border-radius: 18px;
    padding: 12px 16px;
    margin-bottom: 12px;
    max-width: 80%;
}

.chat-bubble.user {
    background: #e3f2fd;
    margin-left: auto;
    text-align: right;
}

.chat-bubble.assistant {
    background: #f8f9fa;
    margin-right: auto;
}
</style>
""", unsafe_allow_html=True)

# Cache para modelos
@st.cache_resource
def get_embedding_model():
    """Carga y cachea el modelo de embeddings"""
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def get_gemini_model():
    """Carga y cachea el modelo Gemini"""
    try:
        model = genai.GenerativeModel('gemini-flash-latest')
        return model
    except Exception as e:
        return None

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

# Funciones de procesamiento (simplificadas)
def extraer_texto_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        texto = ""
        for page in doc:
            texto += page.get_text()
        doc.close()
        return texto
    except:
        return ""

def dividir_texto(texto, chunk_size=1000, overlap=200):
    chunks = []
    for i in range(0, len(texto), chunk_size - overlap):
        chunk = texto[i:i + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def crear_embeddings(chunks, model):
    try:
        embeddings = model.encode(chunks, show_progress_bar=False, batch_size=16)
        return embeddings
    except:
        return None

def crear_indice_faiss(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    return index

def buscar_similares(pregunta, model, index, chunks, k=3):
    try:
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
        return resultados
    except:
        return []

def generar_respuesta_gemini(pregunta, contexto, model):
    if model is None:
        return "❌ Modelo no disponible"
    
    try:
        prompt = f"Basado en este contexto: {contexto[:1000]}... Responde a: {pregunta}. Sé breve y directo."
        
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=500,
                temperature=0.1,
            )
        )
        
        return response.text.strip() if response.text else "No pude generar respuesta"
    except:
        return "Error generando respuesta"

# Inicialización automática
def inicializar_sistema():
    """Inicializa el sistema automáticamente"""
    if 'sistema_inicializado' not in st.session_state:
        st.session_state.sistema_inicializado = False
        st.session_state.chunks = []
        st.session_state.index = None
        st.session_state.model = None
        st.session_state.documentos = []
        st.session_state.gemini_model = None
        st.session_state.chat_history = []
    
    # Auto-inicializar si no está listo
    if not st.session_state.sistema_inicializado:
        with st.spinner("🚀 Inicializando sistema AMSAC AI..."):
            try:
                # Cargar modelos
                model = get_embedding_model()
                gemini_model = get_gemini_model()
                
                # Procesar documentos automáticamente
                documentos_path = Path("documentos")
                if documentos_path.exists():
                    pdf_files = list(documentos_path.glob("*.pdf"))
                    
                    if pdf_files:
                        todos_los_chunks = []
                        documentos_nombres = []
                        
                        for pdf_file in pdf_files:
                            texto = extraer_texto_pdf(pdf_file)
                            if texto:
                                chunks = dividir_texto(texto)
                                todos_los_chunks.extend(chunks)
                                documentos_nombres.append(pdf_file.name)
                        
                        if todos_los_chunks:
                            embeddings = crear_embeddings(todos_los_chunks, model)
                            if embeddings is not None:
                                index = crear_indice_faiss(embeddings)
                                
                                st.session_state.chunks = todos_los_chunks
                                st.session_state.index = index
                                st.session_state.model = model
                                st.session_state.documentos = documentos_nombres
                                st.session_state.gemini_model = gemini_model
                                st.session_state.sistema_inicializado = True
                                
            except Exception as e:
                st.error(f"Error en inicialización: {e}")

# Header estilo Google
def mostrar_header():
    logo_bytes = load_logo()
    
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col1:
        if logo_bytes:
            st.image(logo_bytes, width=120)
    
    with col2:
        st.markdown("""
        <div class="main-header">
            <h1 style="color: white; margin: 0; font-size: 28px;">AMSAC AI</h1>
            <p style="color: rgba(255,255,255,0.9); margin: 4px 0 0 0;">Sistema Inteligente de Consulta de Documentos</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Estado del sistema
        if st.session_state.get('sistema_inicializado', False):
            st.markdown("""
            <div style="text-align: right; margin-top: 20px;">
                <span class="status-indicator status-active"></span>
                <span style="color: white;">Activo</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align: right; margin-top: 20px;">
                <span class="status-indicator status-warning"></span>
                <span style="color: white;">Iniciando...</span>
            </div>
            """, unsafe_allow_html=True)

# Barra de búsqueda estilo Google
def mostrar_barra_busqueda():
    st.markdown('<div class="search-results">', unsafe_allow_html=True)
    
    # Input de búsqueda estilo Google
    col1, col2 = st.columns([6, 1])
    
    with col1:
        pregunta = st.text_input(
            "",
            placeholder="Pregunta sobre tus documentos...",
            key="google_search",
            label_visibility="collapsed"
        )
    
    with col2:
        buscar = st.button("🔍 Buscar", type="primary")
    
    return pregunta, buscar

# Mostrar resultados estilo Google
def mostrar_resultados(pregunta, resultados, respuesta):
    if not resultados:
        st.markdown("""
        <div class="result-card fade-in">
            <p style="color: #70757a; text-align: center; padding: 20px;">
                No se encontraron resultados para tu búsqueda.
            </p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Respuesta principal
    st.markdown(f"""
    <div class="result-card fade-in">
        <div class="result-title">Respuesta AMSAC AI</div>
        <div class="result-snippet">{respuesta}</div>
        <div class="result-url">amsac.ai • Sistema Inteligente</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Fuentes
    with st.expander("📚 Fuentes consultadas"):
        for i, resultado in enumerate(resultados, 1):
            snippet = resultado['chunk'][:200] + "..." if len(resultado['chunk']) > 200 else resultado['chunk']
            st.markdown(f"""
            <div class="result-card">
                <div class="result-title">Fuente {i}</div>
                <div class="result-snippet">{snippet}</div>
                <div class="result-url">Relevancia: {1/(1+resultado['distancia']):.2f}</div>
            </div>
            """, unsafe_allow_html=True)

# Sidebar minimalista
def mostrar_sidebar():
    with st.sidebar:
        st.markdown('<div class="sidebar-google">', unsafe_allow_html=True)
        
        # Logo pequeño
        logo_bytes = load_logo()
        if logo_bytes:
            st.image(logo_bytes, width=80)
        
        st.markdown("### 🤖 AMSAC AI")
        
        # Estado del sistema
        if st.session_state.get('sistema_inicializado', False):
            st.markdown("""
            <span class="status-indicator status-active"></span>
            <span>Sistema Activo</span>
            """, unsafe_allow_html=True)
            st.success(f"📄 {len(st.session_state.get('documentos', []))} documentos")
            st.success(f"🔍 {len(st.session_state.get('chunks', []))} fragmentos")
        else:
            st.markdown("""
            <span class="status-indicator status-warning"></span>
            <span>Inicializando...</span>
            """, unsafe_allow_html=True)
        
        # Botón de reinicio
        if st.button("🔄 Reiniciar Sistema"):
            st.session_state.sistema_inicializado = False
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

# Función principal
def main():
    # Inicializar automáticamente
    inicializar_sistema()
    
    # Mostrar header
    mostrar_header()
    
    # Sidebar minimalista
    mostrar_sidebar()
    
    # Espacio en blanco
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Barra de búsqueda
    pregunta, buscar = mostrar_barra_busqueda()
    
    # Procesar búsqueda
    if (buscar and pregunta) or (pregunta and 'ultima_busqueda' not in st.session_state):
        if st.session_state.get('sistema_inicializado', False):
            with st.spinner("🔍 Buscando información..."):
                resultados = buscar_similares(
                    pregunta, 
                    st.session_state.model, 
                    st.session_state.index, 
                    st.session_state.chunks
                )
                
                if resultados:
                    contexto = "\n\n".join([r['chunk'] for r in resultados])
                    respuesta = generar_respuesta_gemini(pregunta, contexto, st.session_state.gemini_model)
                    
                    # Guardar en historial
                    st.session_state.chat_history.append({
                        'pregunta': pregunta,
                        'respuesta': respuesta,
                        'resultados': resultados
                    })
                    
                    # Mostrar resultados
                    mostrar_resultados(pregunta, resultados, respuesta)
                else:
                    mostrar_resultados(pregunta, [], "No encontré información relevante.")
            
            st.session_state.ultima_busqueda = pregunta
        else:
            st.warning("🔄 El sistema está inicializando, espera un momento...")
    
    # Mostrar historial reciente
    if st.session_state.get('chat_history') and len(st.session_state.chat_history) > 0:
        st.markdown("### 🔍 Búsquedas recientes")
        for i, chat in enumerate(st.session_state.chat_history[-3:], 1):
            with st.expander(f"💬 {chat['pregunta'][:50]}..."):
                st.markdown(f"""
                <div class="chat-bubble user">
                    <strong>Tú:</strong> {chat['pregunta']}
                </div>
                <div class="chat-bubble assistant">
                    <strong>AMSAC AI:</strong> {chat['respuesta']}
                </div>
                """, unsafe_allow_html=True)
    
    # Footer estilo Google
    st.markdown("""
    <div class="google-footer">
        <div style="text-align: center;">
            © 2024 AMSAC - Asesoría y Servicios de Consultoría | Sistema Inteligente RAG con Gemini
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
