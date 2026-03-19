import os
import streamlit as st
from pathlib import Path
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from io import BytesIO
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import logging

# Configurar logging para ver errores
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de la página
st.set_page_config(
    page_title="Sistema RAG Estable",
    page_icon="🛡️",
    layout="wide"
)

# Cache para modelos
@st.cache_resource
def get_embedding_model():
    """Carga y cachea el modelo de embeddings"""
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def get_stable_model():
    """Carga un modelo más estable y robusto"""
    try:
        with st.spinner("🤖 Cargando modelo estable..."):
            # Usar un modelo más pequeño y estable
            model_name = "microsoft/DialoGPT-medium"  # Más estable que TinyLlama
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,  # Usar float32 para más estabilidad
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Crear pipeline con configuración conservadora
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=150,  # Más corto para evitar errores
                temperature=0.1,      # Más bajo para más estabilidad
                do_sample=False,        # Desactivar sampling para evitar probs
                top_k=50,
                repetition_penalty=1.0,
                pad_token_id=tokenizer.eos_token_id,
                return_full_text=False,
                clean_up_tokenization_spaces=True
            )
            
            return pipe
    except Exception as e:
        logger.error(f"Error cargando modelo: {e}")
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

# Función para generar respuesta estable
def generar_respuesta_estable(pregunta, contexto, pipe):
    """Genera respuesta usando modelo estable con manejo de errores"""
    if pipe is None:
        return generar_respuesta_simple(pregunta, contexto)
    
    try:
        # Crear prompt simple y robusto
        prompt = f"Contexto: {contexto[:500]}... Pregunta: {pregunta} Respuesta:"
        
        # Generar respuesta con manejo de errores
        with st.spinner("🤖 Generando respuesta..."):
            try:
                result = pipe(
                    prompt,
                    max_new_tokens=100,
                    temperature=0.1,
                    do_sample=False,  # Determinista para evitar errores
                    pad_token_id=pipe.tokenizer.eos_token_id,
                    clean_up_tokenization_spaces=True
                )
                
                if result and len(result) > 0:
                    respuesta = result[0]['generated_text']
                    
                    # Limpiar respuesta
                    if "Respuesta:" in respuesta:
                        respuesta = respuesta.split("Respuesta:")[-1].strip()
                    else:
                        respuesta = respuesta[len(prompt):].strip()
                    
                    # Validar respuesta
                    if respuesta and len(respuesta) > 10:
                        return respuesta[:300] + "..." if len(respuesta) > 300 else respuesta
                    else:
                        return generar_respuesta_simple(pregunta, contexto)
                else:
                    return generar_respuesta_simple(pregunta, contexto)
                    
            except Exception as model_error:
                logger.error(f"Error en modelo: {model_error}")
                return generar_respuesta_simple(pregunta, contexto)
                
    except Exception as e:
        logger.error(f"Error general: {e}")
        return generar_respuesta_simple(pregunta, contexto)

# Función de respaldo simple
def generar_respuesta_simple(pregunta, contexto):
    """Genera respuesta simple sin IA"""
    if not contexto.strip():
        return "No encontré información relevante en los documentos para responder tu pregunta."
    
    # Extraer frases clave del contexto
    frases = contexto.split('.')[:3]  # Primeras 3 frases
    frases_relevantes = [f.strip() for f in frases if f.strip() and len(f.strip()) > 20]
    
    if frases_relevantes:
        respuesta = f"Basado en los documentos, sobre '{pregunta}' encontré:\n\n"
        for i, frase in enumerate(frases_relevantes, 1):
            respuesta += f"• {frase}.\n"
        return respuesta
    else:
        return f"Encontré información relacionada con tu pregunta, pero el contexto es limitado. El documento menciona temas relacionados con: {pregunta}."

# Interfaz principal
def main():
    st.title("🛡️ Sistema RAG Estable")
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
    if 'pipe_ia' not in st.session_state:
        st.session_state.pipe_ia = None
    
    # Sidebar
    with st.sidebar:
        st.header("🛡️ Configuración Estable")
        
        # Botón para procesar documentos
        if st.button("📄 Procesar Documentos", type="primary"):
            procesar_documentos()
        
        # Botón para cargar modelo estable
        if st.button("🤖 Cargar Modelo Estable", type="secondary"):
            if st.session_state.chunks:
                with st.spinner("Cargando modelo estable..."):
                    pipe = get_stable_model()
                    if pipe:
                        st.session_state.pipe_ia = pipe
                        st.success("✅ Modelo estable cargado")
                    else:
                        st.error("❌ Error cargando modelo")
            else:
                st.warning("⚠️ Procesa documentos primero")
        
        # Mostrar estado
        if st.session_state.chunks:
            st.success(f"✅ {len(st.session_state.chunks)} chunks")
            st.write(f"📄 Documentos: {len(st.session_state.documentos)}")
            for doc in st.session_state.documentos:
                st.write(f"  • {doc}")
        
        if st.session_state.pipe_ia:
            st.success("🤖 IA Estable: Activa")
        else:
            st.warning("🤖 IA Estable: Inactiva")
    
    # Área principal
    if not st.session_state.chunks:
        st.info("👋 ¡Bienvenido al Sistema RAG Estable!")
        st.write("### 🛡️ Características:")
        st.write("- 🛡️ **Máxima estabilidad** - Manejo robusto de errores")
        st.write("- 🤖 **Modelo confiable** - Sin problemas de probabilidad")
        st.write("- ⚡ **Respuestas rápidas** - Modo de respaldo incluido")
        st.write("- 🔄 **Siempre funcional** - Nunca falla completamente")
        
        st.write("### 📋 Pasos:")
        st.write("1. **Agrega archivos PDF** a la carpeta `documentos/`")
        st.write("2. **Presiona 'Procesar Documentos'**")
        st.write("3. **Carga Modelo Estable** (opcional)")
        st.write("4. **Empieza a preguntar** - ¡Siempre funciona!")
        
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
        # Chat interface estable
        st.header("💬 Preguntas Estables")
        
        # Input para pregunta
        pregunta = st.text_input("¿Qué quieres saber?", key="pregunta_input")
        
        if st.button("🔍 Preguntar") and pregunta:
            with st.spinner("🔍 Buscando información..."):
                resultados, search_time = buscar_similares(
                    pregunta, 
                    st.session_state.model, 
                    st.session_state.index, 
                    st.session_state.chunks
                )
                
                if resultados:
                    # Preparar contexto
                    contexto = "\n\n".join([r['chunk'] for r in resultados])
                    
                    # Generar respuesta
                    if st.session_state.pipe_ia:
                        respuesta = generar_respuesta_estable(pregunta, contexto, st.session_state.pipe_ia)
                        tipo_respuesta = "🤖 Respuesta con IA Estable"
                    else:
                        respuesta = generar_respuesta_simple(pregunta, contexto)
                        tipo_respuesta = "📄 Respuesta Simple (Modo Respaldo)"
                    
                    # Mostrar respuesta
                    st.subheader(tipo_respuesta)
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
                st.info("🛡️ ¡Sistema estable y listo!")
            else:
                st.error("Error creando embeddings")
        else:
            st.error("No se pudo extraer texto de los documentos")

if __name__ == "__main__":
    main()
