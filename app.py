import os
import streamlit as st
from pathlib import Path
from typing import List, Dict, Any
import time

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import torch

# Configuración de la página
st.set_page_config(
    page_title="Sistema RAG - Asistente de Documentos",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Variables globales
@st.cache_resource
def cargar_embeddings():
    """Carga los modelos de embeddings"""
    with st.spinner("🔄 Cargando modelos de embeddings..."):
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/distiluse-base-multilingual-cased-v2",
            model_kwargs={'device': 'cpu'}
        )
    return embeddings

@st.cache_resource
def cargar_modelo_llm():
    """Carga el modelo de lenguaje local"""
    try:
        with st.spinner("🤖 Cargando modelo de lenguaje local (puede tardar varios minutos)..."):
            # Configuración para cuantización
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            
            # Modelo pequeño y eficiente
            model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            
            # Crear pipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True,
                top_p=0.95,
                top_k=50,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )
            
            llm = HuggingFacePipeline(pipeline=pipe)
            return llm
    except Exception as e:
        st.error(f"❌ Error cargando el modelo: {e}")
        return None

def procesar_documentos(embeddings):
    """Procesa los documentos PDF y crea el índice vectorial"""
    documentos_path = Path("documentos")
    
    if not documentos_path.exists():
        st.warning("⚠️ La carpeta 'documentos/' no existe. Creándola...")
        documentos_path.mkdir(exist_ok=True)
        return None, []
    
    pdf_files = list(documentos_path.glob("*.pdf"))
    
    if not pdf_files:
        st.warning("⚠️ No se encontraron archivos PDF en la carpeta 'documentos/'")
        return None, []
    
    with st.spinner(f"📄 Procesando {len(pdf_files)} documentos PDF..."):
        all_docs = []
        document_sources = []
        
        for pdf_file in pdf_files:
            st.write(f"   Cargando: {pdf_file.name}")
            loader = PyPDFLoader(str(pdf_file))
            docs = loader.load()
            
            # Añadir metadata
            for doc in docs:
                doc.metadata["source"] = pdf_file.name
            
            all_docs.extend(docs)
            document_sources.append(pdf_file.name)
        
        # Dividir documentos
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len,
        )
        splits = text_splitter.split_documents(all_docs)
        
        # Crear índice FAISS
        repositorio_path = Path("repositorio_faiss")
        repositorio_path.mkdir(exist_ok=True)
        
        if (repositorio_path / "index.faiss").exists():
            vector_store = FAISS.load_local(
                str(repositorio_path), 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            vector_store.add_documents(splits)
        else:
            vector_store = FAISS.from_documents(splits, embeddings)
        
        vector_store.save_local(str(repositorio_path))
        
        return vector_store, document_sources

def generar_respuesta(contexto: str, pregunta: str, llm):
    """Genera una respuesta basada en el contexto"""
    if not contexto.strip():
        return "No encontré información relevante en los documentos para responder tu pregunta."
    
    if llm is not None:
        prompt_template = PromptTemplate(
            template="""Eres un asistente experto que responde preguntas basándose únicamente en el contexto 
            de los documentos proporcionados. Usa la información de los documentos para dar respuestas precisas 
            y detalladas. Si la información no está en los documentos, indica amablemente que no puedes responder 
            basándote en la documentación disponible. Responde en español de forma clara y concisa.
            
            Contexto:
            {contexto}
            
            Pregunta: {pregunta}
            
            Respuesta:""",
            input_variables=["contexto", "pregunta"]
        )
        
        prompt = prompt_template.format(contexto=contexto, pregunta=pregunta)
        respuesta = llm.invoke(prompt)
        return respuesta
    else:
        # Respuesta simple si no hay LLM
        return f"Basado en los documentos proporcionados:\n\n{contexto[:1000]}..."

# Interfaz principal
def main():
    st.title("📚 Sistema RAG - Asistente de Documentos")
    st.markdown("---")
    
    # Sidebar para configuración
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        # Botón para inicializar sistema
        if st.button("🚀 Inicializar Sistema", type="primary"):
            with st.spinner("Iniciando sistema..."):
                # Cargar embeddings
                embeddings = cargar_embeddings()
                st.success("✅ Embeddings cargados")
                
                # Cargar modelo LLM
                llm = cargar_modelo_llm()
                if llm:
                    st.success("✅ Modelo LLM cargado")
                else:
                    st.warning("⚠️ Modo simplificado (sin LLM)")
                
                # Procesar documentos
                vector_store, document_sources = procesar_documentos(embeddings)
                
                if vector_store:
                    st.success(f"✅ Sistema listo con {len(document_sources)} documentos")
                    st.session_state.vector_store = vector_store
                    st.session_state.llm = llm
                    st.session_state.document_sources = document_sources
                else:
                    st.error("❌ No se pudieron procesar los documentos")
        
        # Mostrar estado
        if 'vector_store' in st.session_state:
            st.success("🟢 Sistema Activo")
            st.write(f"📄 Documentos: {len(st.session_state.get('document_sources', []))}")
            if st.session_state.get('document_sources'):
                st.write("📚 Fuentes:")
                for doc in st.session_state.document_sources:
                    st.write(f"  • {doc}")
        else:
            st.warning("🔴 Sistema No Inicializado")
    
    # Área principal
    if 'vector_store' not in st.session_state:
        st.info("👋 ¡Bienvenido! Por favor:")
        st.write("1. **Agrega archivos PDF** a la carpeta `documentos/`")
        st.write("2. **Presiona 'Inicializar Sistema'** en la barra lateral")
        st.write("3. **Espera** a que se carguen los modelos")
        
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
        # Sistema activo - mostrar chat
        st.header("💬 Chat con tus Documentos")
        
        # Inicializar historial
        if 'mensajes' not in st.session_state:
            st.session_state.mensajes = []
        
        # Mostrar historial
        for mensaje in st.session_state.mensajes:
            with st.chat_message(mensaje["role"]):
                st.markdown(mensaje["content"])
                if mensaje.get("sources"):
                    st.write(f"📚 **Fuentes:** {', '.join(mensaje['sources'])}")
        
        # Input para pregunta
        pregunta = st.chat_input("¿Qué quieres saber sobre tus documentos?")
        
        if pregunta:
            # Añadir pregunta al historial
            st.session_state.mensajes.append({"role": "user", "content": pregunta})
            
            with st.chat_message("user"):
                st.markdown(pregunta)
            
            # Procesar pregunta
            with st.chat_message("assistant"):
                with st.spinner("🔍 Buscando información..."):
                    # Buscar documentos relevantes
                    docs = st.session_state.vector_store.similarity_search(pregunta, k=3)
                    
                    if not docs:
                        respuesta = "No encontré información relevante en los documentos para responder tu pregunta."
                        fuentes = []
                    else:
                        # Extraer contexto y fuentes
                        contexto = "\n\n".join([doc.page_content for doc in docs])
                        fuentes = list(set([doc.metadata.get("source", "Documento desconocido") for doc in docs]))
                        
                        # Generar respuesta
                        respuesta = generar_respuesta(
                            contexto, 
                            pregunta, 
                            st.session_state.llm
                        )
                    
                    st.markdown(respuesta)
                    if fuentes:
                        st.write(f"📚 **Fuentes:** {', '.join(fuentes)}")
            
            # Añadir respuesta al historial
            st.session_state.mensajes.append({
                "role": "assistant", 
                "content": respuesta,
                "sources": fuentes
            })
        
        # Botón para limpiar chat
        if st.button("🗑️ Limpiar Chat"):
            st.session_state.mensajes = []
            st.rerun()

if __name__ == "__main__":
    main()
