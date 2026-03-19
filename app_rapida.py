import os
import streamlit as st
from pathlib import Path
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from io import BytesIO
import time

# Configuración de la página
st.set_page_config(
    page_title="Sistema RAG Rápido",
    page_icon="⚡",
    layout="wide"
)

# Cache para modelos pesados
@st.cache_resource
def get_embedding_model():
    """Carga y cachea el modelo de embeddings"""
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def get_text_splitter():
    """Configuración del divisor de texto"""
    return {
        'chunk_size': 800,  # Más pequeño para más velocidad
        'overlap': 150
    }

# Función simple para extraer texto de PDF
def extraer_texto_pdf(pdf_path):
    """Extrae texto de un archivo PDF usando PyMuPDF"""
    try:
        doc = fitz.open(pdf_path)
        texto = ""
        # Limitar a primeras 50 páginas para mayor velocidad
        for page in doc[:50]:
            texto += page.get_text()
        doc.close()
        return texto
    except Exception as e:
        st.error(f"Error leyendo PDF: {e}")
        return ""

# Función para dividir texto en chunks
def dividir_texto(texto, chunk_size=800, overlap=150):
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

# Función para generar respuesta rápida (sin IA generativa)
def generar_respuesta_rapida(pregunta, resultados):
    """Genera una respuesta rápida basada en los chunks encontrados"""
    if not resultados:
        return "No encontré información relevante en los documentos para responder tu pregunta."
    
    # Extraer información clave
    contexto = "\n\n".join([r['chunk'] for r in resultados])
    
    # Crear respuesta estructurada
    respuesta = f"""## 📄 Información encontrada sobre: "{pregunta}"

Basado en los documentos, aquí está la información relevante:

"""
    
    for i, resultado in enumerate(resultados, 1):
        # Extraer las primeras 2-3 frases más relevantes
        chunk = resultado['chunk']
        frases = chunk.split('.')[:3]  # Primeras 3 frases
        resumen = '. '.join(frases).strip()
        
        if resumen:
            respuesta += f"**Punto {i}:** {resumen}.\n\n"
    
    respuesta += f"""
---
📊 **Fuente:** Documento procesado  
🎯 **Relevancia:** {1/(1+resultados[0]['distancia']):.2f}  
⚡ **Tiempo de búsqueda:** {resultados[0].get('tiempo', 0):.2f} segundos
"""
    
    return respuesta

# Interfaz principal
def main():
    st.title("⚡ Sistema RAG Ultra Rápido")
    st.markdown("---")
    
    # Estado de la aplicación
    if 'chunks' not in st.session_state:
        st.session_state.chunks = []
    if 'index' not in st.session_state:
        st.session_state.index = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'documentos' not in st.session_state:
        st.session_state.documentos = []
    if 'tiempo_procesamiento' not in st.session_state:
        st.session_state.tiempo_procesamiento = 0
    
    # Sidebar
    with st.sidebar:
        st.header("⚡ Configuración Rápida")
        
        # Botón para procesar documentos
        if st.button("📄 Procesar Documentos", type="primary"):
            procesar_documentos()
        
        # Mostrar estado y métricas
        if st.session_state.chunks:
            st.success(f"✅ {len(st.session_state.chunks)} chunks")
            st.write(f"📄 Documentos: {len(st.session_state.documentos)}")
            st.write(f"⏱️ Procesamiento: {st.session_state.tiempo_procesamiento:.2f}s")
            
            for doc in st.session_state.documentos:
                st.write(f"  • {doc}")
        else:
            st.warning("⚠️ Sin documentos procesados")
        
        # Opciones de configuración
        st.subheader("🔧 Configuración")
        k_chunks = st.slider("Chunks a buscar", min_value=1, max_value=10, value=3)
        st.session_state.k_chunks = k_chunks
    
    # Área principal
    if not st.session_state.chunks:
        st.info("👋 ¡Bienvenido al sistema RAG Ultra Rápido!")
        st.write("### 🚀 Características:")
        st.write("- ⚡ **Búsqueda instantánea** - Respuestas en menos de 1 segundo")
        st.write("- 📊 **Resultados estructurados** - Información organizada")
        st.write("- 🎯 **Alta precisión** - Encuentra lo relevante rápidamente")
        
        st.write("### 📋 Pasos:")
        st.write("1. **Agrega archivos PDF** a la carpeta `documentos/`")
        st.write("2. **Presiona 'Procesar Documentos'** en la barra lateral")
        st.write("3. **Empieza a preguntar** - ¡Respuestas instantáneas!")
        
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
        # Chat interface ultra rápido
        st.header("💬 Preguntas Ultra Rápidas")
        
        # Input para pregunta
        col1, col2 = st.columns([4, 1])
        with col1:
            pregunta = st.text_input("¿Qué quieres saber?", key="pregunta_input")
        with col2:
            buscar = st.button("🔍 Buscar", type="primary")
        
        # Buscar automáticamente si hay pregunta
        if (buscar and pregunta) or (pregunta and 'ultima_pregunta' not in st.session_state):
            with st.spinner("⚡ Buscando información..."):
                start_time = time.time()
                
                resultados, search_time = buscar_similares(
                    pregunta, 
                    st.session_state.model, 
                    st.session_state.index, 
                    st.session_state.chunks,
                    k=st.session_state.k_chunks
                )
                
                # Añadir tiempo a cada resultado
                for resultado in resultados:
                    resultado['tiempo'] = search_time
                
                total_time = time.time() - start_time
                
                if resultados:
                    # Generar respuesta rápida
                    respuesta = generar_respuesta_rapida(pregunta, resultados)
                    
                    # Mostrar respuesta
                    st.markdown(respuesta)
                    
                    # Mostrar métricas de rendimiento
                    st.success(f"⚡ **Respuesta generada en {total_time:.2f} segundos**")
                    
                    # Opción para ver detalles
                    with st.expander("📚 Ver fragmentos completos"):
                        for i, resultado in enumerate(resultados, 1):
                            st.write(f"**Fragmento {i}** (Relevancia: {1/(1+resultado['distancia']):.2f}):")
                            st.write(resultado['chunk'][:500] + "..." if len(resultado['chunk']) > 500 else resultado['chunk'])
                            st.write("---")
                else:
                    st.warning("No se encontró información relevante para tu pregunta.")
                    st.info("💡 Intenta con otras palabras o se más específico.")
            
            st.session_state.ultima_pregunta = pregunta
        
        # Limpiar pregunta para nueva búsqueda
        if st.button("🗑️ Nueva Pregunta"):
            st.session_state.ultima_pregunta = None
            st.rerun()

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
    
    start_time = time.time()
    
    with st.spinner(f"⚡ Procesando {len(pdf_files)} documentos a máxima velocidad..."):
        # Obtener modelo cacheado
        model = get_embedding_model()
        splitter_config = get_text_splitter()
        
        todos_los_chunks = []
        st.session_state.documentos = []
        
        for pdf_file in pdf_files:
            st.write(f"📄 Procesando: {pdf_file.name}")
            
            # Extraer texto
            texto = extraer_texto_pdf(pdf_file)
            if texto:
                # Dividir en chunks
                chunks = dividir_texto(texto, splitter_config['chunk_size'], splitter_config['overlap'])
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
                
                # Calcular tiempo total
                st.session_state.tiempo_procesamiento = time.time() - start_time
                
                st.success(f"✅ Procesados {len(todos_los_chunks)} fragmentos en {st.session_state.tiempo_procesamiento:.2f} segundos")
                st.info("🚀 ¡Listo para búsquedas ultra rápidas!")
            else:
                st.error("Error creando embeddings")
        else:
            st.error("No se pudo extraer texto de los documentos")

if __name__ == "__main__":
    main()
