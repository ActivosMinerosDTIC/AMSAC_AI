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

# Configuración de la página
st.set_page_config(
    page_title="Sistema RAG Gratis Mejorado",
    page_icon="🆓",
    layout="wide"
)

# Cache para modelos
@st.cache_resource
def get_embedding_model():
    """Carga y cachea el modelo de embeddings"""
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def get_free_model():
    """Carga un modelo gratuito pero mejorado"""
    try:
        with st.spinner("🤖 Cargando modelo gratuito mejorado..."):
            # Usar Phi-3-mini - mejor que TinyLlama y gratuito
            model_name = "microsoft/Phi-3-mini-4k-instruct"
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                # Reducir uso de memoria
                low_cpu_mem_usage=True
            )
            
            # Crear pipeline optimizado
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=300,  # Más corto para mayor velocidad
                temperature=0.3,      # Más determinista
                do_sample=True,
                top_p=0.8,
                top_k=30,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
                # Optimizaciones
                truncation=True,
                return_full_text=False
            )
            
            return pipe
    except Exception as e:
        st.error(f"Error cargando modelo: {e}")
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

# Función para generar respuesta con modelo gratuito mejorado
def generar_respuesta_gratis(pregunta, contexto, pipe):
    """Genera respuesta usando modelo gratuito mejorado"""
    if pipe is None:
        return "⚠️ El modelo gratuito no está disponible. Mostrando contexto encontrado."
    
    try:
        # Crear prompt optimizado para Phi-3
        prompt = f"""<|user|>
Eres un asistente experto que responde preguntas basándose únicamente en el contexto proporcionado.

Contexto del documento:
{contexto}

Pregunta: {pregunta}

Responde basándote únicamente en la información del contexto. Sé claro y conciso.
<|end|>
<|assistant|>"""

        # Generar respuesta
        with st.spinner("🤖 Generando respuesta con IA gratuita..."):
            result = pipe(
                prompt,
                max_new_tokens=250,
                temperature=0.3,
                do_sample=True,
                top_p=0.8,
                pad_token_id=pipe.tokenizer.eos_token_id,
                truncation=True
            )
            
            # Extraer respuesta generada
            respuesta_completa = result[0]['generated_text']
            
            # Limpiar la respuesta
            if "<|assistant|>" in respuesta_completa:
                respuesta = respuesta_completa.split("<|assistant|>")[-1].strip()
            else:
                respuesta = respuesta_completa[len(prompt):].strip()
            
            # Limitar longitud
            if len(respuesta) > 800:
                respuesta = respuesta[:800] + "..."
            
            return respuesta
            
    except Exception as e:
        st.error(f"Error generando respuesta: {e}")
        return f"Error generando respuesta: {str(e)}"

# Interfaz principal
def main():
    st.title("🆓 Sistema RAG Gratis Mejorado")
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
        st.header("🆓 Configuración Gratis")
        
        # Botón para procesar documentos
        if st.button("📄 Procesar Documentos", type="primary"):
            procesar_documentos()
        
        # Botón para cargar modelo gratuito
        if st.button("🤖 Cargar IA Gratuita", type="secondary"):
            if st.session_state.chunks:
                pipe = get_free_model()
                if pipe:
                    st.session_state.pipe_ia = pipe
                    st.success("✅ Modelo gratuito cargado")
            else:
                st.warning("⚠️ Procesa documentos primero")
        
        # Mostrar estado
        if st.session_state.chunks:
            st.success(f"✅ {len(st.session_state.chunks)} chunks")
            st.write(f"📄 Documentos: {len(st.session_state.documentos)}")
            for doc in st.session_state.documentos:
                st.write(f"  • {doc}")
        
        if st.session_state.pipe_ia:
            st.success("🤖 IA Gratuita: Activa")
            st.info("🧠 Modelo: Phi-3-mini")
        else:
            st.warning("🤖 IA Gratuita: Inactiva")
        
        # Información del modelo
        st.subheader("ℹ️ Modelo Info")
        st.write("**Phi-3-mini** por Microsoft")
        st.write("- 100% gratuito")
        st.write("- 3.8B parámetros")
        st.write("- Mejor que TinyLlama")
        st.write("- Respuestas en 5-10s")
    
    # Área principal
    if not st.session_state.chunks:
        st.info("👋 ¡Bienvenido al Sistema RAG Gratis Mejorado!")
        st.write("### 🆓 Características:")
        st.write("- 🤖 **Phi-3-mini** - Modelo gratuito mejorado")
        st.write("- 🆓 **100% gratis** - Sin API keys ni costos")
        st.write("- ⚡ **Más rápido** - Respuestas en 5-10 segundos")
        st.write("- 🧠 **Más inteligente** - Mejor que TinyLlama")
        
        st.write("### 📋 Pasos:")
        st.write("1. **Agrega archivos PDF** a la carpeta `documentos/`")
        st.write("2. **Presiona 'Procesar Documentos'**")
        st.write("3. **Carga IA Gratuita** (Phi-3-mini)")
        st.write("4. **Empieza a preguntar** - ¡Respuestas mejoradas!")
        
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
        st.header("💬 Preguntas con IA Gratuita Mejorada")
        
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
                    
                    # Generar respuesta con IA gratuita
                    if st.session_state.pipe_ia:
                        respuesta = generar_respuesta_gratis(pregunta, contexto, st.session_state.pipe_ia)
                        
                        # Mostrar respuesta generada
                        st.subheader("🤖 Respuesta con IA Gratuita")
                        st.write(respuesta)
                        
                        # Mostrar métricas
                        st.success(f"⚡ Respuesta generada con IA gratuita")
                        
                        # Mostrar fuentes
                        with st.expander("📚 Fuentes utilizadas"):
                            for i, resultado in enumerate(resultados, 1):
                                st.write(f"**Fragmento {i}** (Relevancia: {1/(1+resultado['distancia']):.2f}):")
                                st.write(resultado['chunk'][:300] + "..." if len(resultado['chunk']) > 300 else resultado['chunk'])
                                st.write("---")
                    else:
                        # Modo sin IA - mostrar contexto organizado
                        st.subheader("📄 Información Encontrada")
                        
                        respuesta_simple = f"Basada en los documentos, sobre '{pregunta}' encontré:\n\n"
                        
                        for i, resultado in enumerate(resultados, 1):
                            respuesta_simple += f"**Punto {i}:** {resultado['chunk'][:200]}...\n\n"
                        
                        st.write(respuesta_simple)
                        
                        st.info("💡 Activa 'IA Gratuita' para respuestas mejoradas")
                        
                        # Mostrar detalles
                        with st.expander("📚 Ver fragmentos completos"):
                            for i, resultado in enumerate(resultados, 1):
                                st.write(f"**Fragmento {i}** (Relevancia: {1/(1+resultado['distancia']):.2f}):")
                                st.write(resultado['chunk'])
                                st.write("---")
                else:
                    st.warning("No se encontró información relevante para tu pregunta.")

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
                st.info("🆓 ¡Ahora carga la IA gratuita mejorada!")
            else:
                st.error("Error creando embeddings")
        else:
            st.error("No se pudo extraer texto de los documentos")

if __name__ == "__main__":
    main()
