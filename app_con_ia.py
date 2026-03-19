import os
import streamlit as st
from pathlib import Path
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from io import BytesIO
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Configuración de la página
st.set_page_config(
    page_title="Sistema RAG con IA Generativa",
    page_icon="🤖",
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
                    'distancia': distances[0][i],
                    'indice': idx
                })
        return resultados
    except Exception as e:
        st.error(f"Error en búsqueda: {e}")
        return []

# Función para cargar modelo de lenguaje generativo
def cargar_modelo_ia():
    """Carga un modelo de lenguaje local para generar respuestas"""
    try:
        with st.spinner("🤖 Cargando modelo de IA generativa (puede tardar minutos)..."):
            # Usar un modelo más pequeño y eficiente
            model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            
            # Configuración optimizada para Mac ARM
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Crear pipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )
            
            return pipe
    except Exception as e:
        st.error(f"Error cargando modelo de IA: {e}")
        return None

# Función para generar respuesta con IA
def generar_respuesta_ia(pregunta, contexto, pipe):
    """Genera una respuesta usando el modelo de lenguaje"""
    if pipe is None:
        return "⚠️ El modelo de IA no está disponible. Mostrando contexto encontrado."
    
    try:
        # Crear prompt para el modelo
        prompt = f"""Eres un asistente experto que responde preguntas basándose únicamente en el contexto proporcionado. 

Contexto del documento:
{contexto}

Pregunta del usuario: {pregunta}

Instrucciones:
- Responde basándote únicamente en la información del contexto
- Si la información no está en el contexto, indícalo amablemente
- Sé claro, conciso y profesional
- Responde en español

Respuesta:"""

        # Generar respuesta
        with st.spinner("🧠 Generando respuesta con IA..."):
            result = pipe(
                prompt,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=pipe.tokenizer.eos_token_id
            )
            
            # Extraer solo la respuesta generada
            respuesta_completa = result[0]['generated_text']
            
            # Encontrar dónde empieza la respuesta real
            if "Respuesta:" in respuesta_completa:
                respuesta = respuesta_completa.split("Respuesta:")[-1].strip()
            else:
                respuesta = respuesta_completa[len(prompt):].strip()
            
            # Limitar longitud si es muy larga
            if len(respuesta) > 1000:
                respuesta = respuesta[:1000] + "..."
            
            return respuesta
            
    except Exception as e:
        st.error(f"Error generando respuesta: {e}")
        return f"Error generando respuesta: {str(e)}"

# Interfaz principal
def main():
    st.title("🤖 Sistema RAG con IA Generativa")
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
    if 'pipe_ia' not in st.session_state:
        st.session_state.pipe_ia = None
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        # Botón para procesar documentos
        if st.button("📄 Procesar Documentos", type="primary"):
            procesar_documentos()
        
        # Botón para cargar modelo de IA
        if st.button("🤖 Cargar Modelo IA", type="secondary"):
            if st.session_state.chunks:
                pipe = cargar_modelo_ia()
                if pipe:
                    st.session_state.pipe_ia = pipe
                    st.success("✅ Modelo de IA cargado")
            else:
                st.warning("⚠️ Procesa documentos primero")
        
        # Mostrar estado
        if st.session_state.chunks:
            st.success(f"✅ {len(st.session_state.chunks)} chunks procesados")
            st.write(f"📄 Documentos: {len(st.session_state.documentos)}")
            for doc in st.session_state.documentos:
                st.write(f"  • {doc}")
        
        if st.session_state.pipe_ia:
            st.success("🤖 IA Generativa: Activa")
        else:
            st.warning("🤖 IA Generativa: Inactiva")
    
    # Área principal
    if not st.session_state.chunks:
        st.info("👋 ¡Bienvenido! Por favor:")
        st.write("1. **Agrega archivos PDF** a la carpeta `documentos/`")
        st.write("2. **Presiona 'Procesar Documentos'** en la barra lateral")
        st.write("3. **Opcional: Carga Modelo IA** para respuestas generativas")
        
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
        st.header("💬 Preguntas Inteligentes con IA")
        
        # Input para pregunta
        pregunta = st.text_input("¿Qué quieres saber?", key="pregunta_input")
        
        if st.button("🔍 Preguntar") and pregunta:
            with st.spinner("🔍 Buscando información..."):
                resultados = buscar_similares(
                    pregunta, 
                    st.session_state.model, 
                    st.session_state.index, 
                    st.session_state.chunks
                )
                
                if resultados:
                    # Preparar contexto para la IA
                    contexto = "\n\n".join([r['chunk'] for r in resultados])
                    
                    # Generar respuesta con IA o mostrar contexto
                    if st.session_state.pipe_ia:
                        respuesta = generar_respuesta_ia(pregunta, contexto, st.session_state.pipe_ia)
                        
                        # Mostrar respuesta generada
                        st.subheader("🤖 Respuesta Generada por IA")
                        st.write(respuesta)
                        
                        # Mostrar fuentes usadas
                        with st.expander("📚 Fuentes utilizadas"):
                            for i, resultado in enumerate(resultados, 1):
                                st.write(f"**Fragmento {i}** (Similitud: {1/(1+resultado['distancia']):.2f}):")
                                st.write(resultado['chunk'][:300] + "..." if len(resultado['chunk']) > 300 else resultado['chunk'])
                                st.write("---")
                    else:
                        # Modo sin IA - mostrar contexto organizado
                        st.subheader("📄 Información Encontrada")
                        
                        # Crear respuesta simple
                        respuesta_simple = f"Basada en los documentos, sobre '{pregunta}' encontré la siguiente información:\n\n"
                        
                        for i, resultado in enumerate(resultados, 1):
                            respuesta_simple += f"**Punto {i}:** {resultado['chunk'][:200]}...\n\n"
                        
                        st.write(respuesta_simple)
                        
                        # Mostrar detalles en expandible
                        with st.expander("📚 Ver fragmentos completos"):
                            for i, resultado in enumerate(resultados, 1):
                                st.write(f"**Fragmento {i}** (Similitud: {1/(1+resultado['distancia']):.2f}):")
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
                st.info("💡 Ahora puedes cargar el modelo de IA para respuestas generativas")
            else:
                st.error("Error creando embeddings")
        else:
            st.error("No se pudo extraer texto de los documentos")

if __name__ == "__main__":
    main()
