import os
import streamlit as st
from pathlib import Path
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from io import BytesIO
import time
from dotenv import load_dotenv
from openai import OpenAI

# Cargar variables de entorno
load_dotenv()

# Configuración de la página
st.set_page_config(
    page_title="Sistema RAG con DeepSeek",
    page_icon="🧠",
    layout="wide"
)

# Cache para modelos
@st.cache_resource
def get_embedding_model():
    """Carga y cachea el modelo de embeddings"""
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def get_deepseek_client():
    """Carga y cachea el cliente de DeepSeek"""
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    
    if not api_key:
        st.error("❌ No se encontró API key. Configura OPENROUTER_API_KEY o DEEPSEEK_API_KEY en .env")
        return None
    
    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )

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
            embeddings = model.encode(chunks, show_progress_bar=True, batch_size=32)
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
    except Exception as e:
        st.error(f"Error en búsqueda: {e}")
        return []

# Función para generar respuesta con DeepSeek
def generar_respuesta_deepseek(pregunta, contexto, client):
    """Genera una respuesta usando DeepSeek"""
    if client is None:
        return "❌ Cliente de DeepSeek no configurado. Verifica tu API key."
    
    try:
        with st.spinner("🧠 DeepSeek generando respuesta..."):
            # Crear prompt para DeepSeek
            system_prompt = """Eres un asistente experto que responde preguntas basándose únicamente en el contexto proporcionado. 

Reglas importantes:
- Responde basándote ÚNICAMENTE en la información del contexto
- Si la información no está en el contexto, indícalo claramente
- Sé claro, conciso y profesional
- Responde en español
- Usa un tono servicial y experto"""

            user_prompt = f"""Contexto del documento:
{contexto}

Pregunta del usuario: {pregunta}

Por favor, responde a la pregunta basándote únicamente en el contexto proporcionado."""

            # Llamar a DeepSeek
            response = client.chat.completions.create(
                model="deepseek/deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1000,
                temperature=0.1,
                stream=False
            )
            
            respuesta = response.choices[0].message.content
            return respuesta
            
    except Exception as e:
        st.error(f"Error con DeepSeek: {e}")
        return f"Error generando respuesta con DeepSeek: {str(e)}"

# Interfaz principal
def main():
    st.title("🧠 Sistema RAG con DeepSeek")
    st.markdown("---")
    
    # Verificar API key
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        st.error("🔑 **API Key no configurada**")
        st.info("Para usar DeepSeek, configura tu API key:")
        st.code("OPENROUTER_API_KEY=tu_api_key_aqui")
        st.info("O crea un archivo .env con esa línea")
    
    # Estado de la aplicación
    if 'chunks' not in st.session_state:
        st.session_state.chunks = []
    if 'index' not in st.session_state:
        st.session_state.index = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'documentos' not in st.session_state:
        st.session_state.documentos = []
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        # Estado de API
        if api_key:
            st.success("🔑 API Key configurada")
        else:
            st.error("🔑 API Key faltante")
        
        # Botón para procesar documentos
        if st.button("📄 Procesar Documentos", type="primary"):
            procesar_documentos()
        
        # Mostrar estado
        if st.session_state.chunks:
            st.success(f"✅ {len(st.session_state.chunks)} chunks procesados")
            st.write(f"📄 Documentos: {len(st.session_state.documentos)}")
            for doc in st.session_state.documentos:
                st.write(f"  • {doc}")
        else:
            st.warning("⚠️ Sin documentos procesados")
    
    # Área principal
    if not st.session_state.chunks:
        st.info("👋 ¡Bienvenido al sistema RAG con DeepSeek!")
        st.write("### 🧠 Características:")
        st.write("- 🤖 **IA DeepSeek** - Respuestas de alta calidad")
        st.write("- 📚 **Basado en documentos** - Solo usa tu información")
        st.write("- ⚡ **Rápido y preciso** - Búsqueda vectorial optimizada")
        
        st.write("### 📋 Pasos:")
        st.write("1. **Configura tu API key** de OpenRouter/DeepSeek")
        st.write("2. **Agrega archivos PDF** a la carpeta `documentos/`")
        st.write("3. **Presiona 'Procesar Documentos'**")
        st.write("4. **Empieza a preguntar** - ¡Respuestas expertas!")
        
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
        # Chat interface con DeepSeek
        st.header("💬 Preguntas con DeepSeek")
        
        # Input para pregunta
        pregunta = st.text_input("¿Qué quieres saber?", key="pregunta_input")
        
        if st.button("🔍 Preguntar a DeepSeek") and pregunta:
            if not api_key:
                st.error("❌ Configura tu API key primero")
                return
            
            with st.spinner("🔍 Buscando información..."):
                resultados = buscar_similares(
                    pregunta, 
                    st.session_state.model, 
                    st.session_state.index, 
                    st.session_state.chunks
                )
                
                if resultados:
                    # Preparar contexto para DeepSeek
                    contexto = "\n\n".join([r['chunk'] for r in resultados])
                    
                    # Generar respuesta con DeepSeek
                    client = get_deepseek_client()
                    respuesta = generar_respuesta_deepseek(pregunta, contexto, client)
                    
                    # Mostrar respuesta
                    st.subheader("🧠 Respuesta de DeepSeek")
                    st.write(respuesta)
                    
                    # Mostrar fuentes
                    with st.expander("📚 Fuentes utilizadas"):
                        for i, resultado in enumerate(resultados, 1):
                            st.write(f"**Fragmento {i}** (Relevancia: {1/(1+resultado['distancia']):.2f}):")
                            st.write(resultado['chunk'][:300] + "..." if len(resultado['chunk']) > 300 else resultado['chunk'])
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
                st.info("🧠 ¡Listo para preguntas con DeepSeek!")
            else:
                st.error("Error creando embeddings")
        else:
            st.error("No se pudo extraer texto de los documentos")

if __name__ == "__main__":
    main()
