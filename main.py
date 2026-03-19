import os
import asyncio
from contextlib import asynccontextmanager
from typing import List, Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import torch

# Cargar variables de entorno
load_dotenv()

# Variables globales para el sistema RAG
embeddings = None
vector_store = None
llm = None
document_sources = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manager del ciclo de vida de la aplicación FastAPI"""
    global embeddings, vector_store, llm, document_sources
    
    print("🚀 Iniciando sistema RAG...")
    
    # 1. Configurar embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/distiluse-base-multilingual-cased-v2",
        model_kwargs={'device': 'cpu'}
    )
    print("✅ Embeddings configurados")
    
    # 2. Configurar el modelo de lenguaje local y gratuito
    print("🤖 Cargando modelo de lenguaje local...")
    
    try:
        # Configuración para cuantización (reducir uso de memoria)
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
        
        # Crear pipeline de texto
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
        print("✅ Modelo de lenguaje local configurado")
    except Exception as e:
        print(f"⚠️ Error cargando modelo local: {e}")
        print("🔄 Usando modo simplificado sin generación de lenguaje...")
        llm = None
    
    # 3. Procesar documentos PDF si existen
    documentos_path = Path("documentos")
    if documentos_path.exists() and any(documentos_path.glob("*.pdf")):
        print("📄 Procesando documentos PDF...")
        
        all_docs = []
        document_sources = []
        
        # Cargar todos los PDFs
        for pdf_file in documentos_path.glob("*.pdf"):
            print(f"   Cargando: {pdf_file.name}")
            loader = PyPDFLoader(str(pdf_file))
            docs = loader.load()
            
            # Añadir metadata con el nombre del archivo
            for doc in docs:
                doc.metadata["source"] = pdf_file.name
            
            all_docs.extend(docs)
            document_sources.append(pdf_file.name)
        
        # Dividir documentos en chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len,
        )
        splits = text_splitter.split_documents(all_docs)
        print(f"   📊 Documentos divididos en {len(splits)} chunks")
        
        # Crear o actualizar índice FAISS
        repositorio_path = Path("repositorio_faiss")
        repositorio_path.mkdir(exist_ok=True)
        
        if (repositorio_path / "index.faiss").exists():
            print("   📂 Cargando índice existente...")
            vector_store = FAISS.load_local(
                str(repositorio_path), 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            # Actualizar con nuevos documentos
            vector_store.add_documents(splits)
        else:
            print("   🆕 Creando nuevo índice...")
            vector_store = FAISS.from_documents(splits, embeddings)
        
        # Guardar índice actualizado
        vector_store.save_local(str(repositorio_path))
        print(f"   💾 Índice guardado en {repositorio_path}")
        
        print("✅ Sistema RAG configurado exitosamente")
        print(f"📚 Documentos procesados: {', '.join(document_sources)}")
    else:
        print("⚠️  No se encontraron documentos PDF en la carpeta 'documentos/'")
        print("   El sistema iniciará sin capacidad de RAG")
    
    yield
    
    print("🛑 Apagando sistema RAG...")

# Crear aplicación FastAPI
app = FastAPI(
    title="Sistema RAG con FastAPI",
    description="API para consultas inteligentes sobre documentos PDF",
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

# Modelos Pydantic
class ChatRequest(BaseModel):
    pregunta: str
    historial: List[Dict[str, str]] = []

class ChatResponse(BaseModel):
    respuesta: str
    fuentes: List[str]

def generar_respuesta_simple(contexto: str, pregunta: str) -> str:
    """Genera una respuesta simple basada en el contexto sin usar el modelo LLM"""
    if not contexto.strip():
        return "No encontré información relevante en los documentos para responder tu pregunta."
    
    # Respuesta simple basada en contexto
    return f"Basado en los documentos proporcionados:\n\n{contexto[:1000]}..."

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Endpoint para procesar preguntas con RAG"""
    
    if vector_store is None:
        raise HTTPException(
            status_code=503, 
            detail="El sistema RAG no está disponible. No se han procesado documentos."
        )
    
    try:
        # Recuperar documentos relevantes
        docs = vector_store.similarity_search(request.pregunta, k=3)
        
        if not docs:
            return ChatResponse(
                respuesta="No encontré información relevante en los documentos para responder tu pregunta.",
                fuentes=[]
            )
        
        # Extraer contexto y fuentes
        contexto = "\n\n".join([doc.page_content for doc in docs])
        fuentes = list(set([doc.metadata.get("source", "Documento desconocido") for doc in docs]))
        
        # Generar respuesta
        if llm is not None:
            # Usar el modelo LLM si está disponible
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
            
            prompt = prompt_template.format(contexto=contexto, pregunta=request.pregunta)
            respuesta = llm.invoke(prompt)
        else:
            # Usar respuesta simple si no hay LLM
            respuesta = generar_respuesta_simple(contexto, request.pregunta)
        
        return ChatResponse(
            respuesta=respuesta,
            fuentes=fuentes
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error procesando la solicitud: {str(e)}"
        )

@app.get("/")
async def root():
    """Endpoint de bienvenida"""
    return {
        "mensaje": "Sistema RAG con FastAPI",
        "estado": "activo",
        "documentos_procesados": len(document_sources),
        "fuentes_disponibles": document_sources,
        "modelo_llm": "TinyLlama local" if llm else "Modo simplificado"
    }

@app.get("/health")
async def health_check():
    """Endpoint para verificar el estado del sistema"""
    return {
        "status": "healthy",
        "rag_available": vector_store is not None,
        "documents_loaded": len(document_sources),
        "vector_store_ready": vector_store is not None,
        "llm_available": llm is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
