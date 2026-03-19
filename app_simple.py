import os
import streamlit as st
from pathlib import Path
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from io import BytesIO

# Configuración de la página
st.set_page_config(
    page_title="Sistema RAG Simple",
    page_icon="📚",
    layout="wide"
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
def crear_embeddings(chunks, model_name="all-MiniLM-L6-v2"):
    """Crea embeddings para los chunks de texto"""
    try:
        model = SentenceTransformer(model_name)
        embeddings = model.encode(chunks, show_progress_bar=True)
        return embeddings, model
    except Exception as e:
        st.error(f"Error creando embeddings: {e}")
        return None, None

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
                    'distancia': distances[0][i]
                })
        return resultados
    except Exception as e:
        st.error(f"Error en búsqueda: {e}")
        return []

# Interfaz principal
def main():
    st.title("📚 Sistema RAG Simple")
    st.markdown("---")
    
    # Estado de la aplicación
    if 'chunks' not in st.session_state:
        st.session_state.chunks = []
    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = None
    if 'index' not in st.session_state:
        st.session_state.index = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'documentos' not in st.session_state:
        st.session_state.documentos = []
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuración")
        
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
        st.info("👋 ¡Bienvenido! Por favor:")
        st.write("1. **Agrega archivos PDF** a la carpeta `documentos/`")
        st.write("2. **Presiona 'Procesar Documentos'** en la barra lateral")
        
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
        # Chat interface
        st.header("💬 Preguntas sobre tus Documentos")
        
        # Input para pregunta
        pregunta = st.text_input("¿Qué quieres saber?", key="pregunta_input")
        
        if st.button("🔍 Buscar") and pregunta:
            with st.spinner("🔍 Buscando información..."):
                resultados = buscar_similares(
                    pregunta, 
                    st.session_state.model, 
                    st.session_state.index, 
                    st.session_state.chunks
                )
                
                if resultados:
                    st.subheader("📄 Resultados Encontrados:")
                    for i, resultado in enumerate(resultados, 1):
                        with st.expander(f"Fragmento {i} (Similitud: {1/(1+resultado['distancia']):.2f})"):
                            st.write(resultado['chunk'])
                else:
                    st.warning("No se encontró información relevante.")

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
    
    with st.spinner(f"Procesando {len(pdf_files)} documentos..."):
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
            st.write("🧠 Creando embeddings...")
            embeddings, model = crear_embeddings(todos_los_chunks)
            
            if embeddings is not None:
                # Crear índice
                st.write("📊 Creando índice de búsqueda...")
                index = crear_indice_faiss(embeddings)
                
                # Guardar en sesión
                st.session_state.chunks = todos_los_chunks
                st.session_state.embeddings = embeddings
                st.session_state.index = index
                st.session_state.model = model
                
                st.success(f"✅ Procesados {len(todos_los_chunks)} fragmentos de {len(pdf_files)} documentos")
            else:
                st.error("Error creando embeddings")
        else:
            st.error("No se pudo extraer texto de los documentos")

if __name__ == "__main__":
    main()
