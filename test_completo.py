#!/usr/bin/env python3
"""
Test completo de funcionalidad del sistema RAG
"""

import os
import sys
from pathlib import Path

def test_importaciones():
    """Verifica que todas las librerías se importen correctamente"""
    print("🧪 Test 1: Importaciones...")
    try:
        import streamlit as st
        import fitz  # PyMuPDF
        from sentence_transformers import SentenceTransformer
        import faiss
        import numpy as np
        print("✅ Todas las librerías importadas correctamente")
        return True
    except Exception as e:
        print(f"❌ Error en importaciones: {e}")
        return False

def test_lectura_pdf():
    """Verifica que se pueda leer el PDF"""
    print("\n🧪 Test 2: Lectura de PDF...")
    try:
        import fitz
        pdf_path = Path('documentos/TDR Servicio de Ejecucion de actividades para lograr la certificacion de la norma ISO IEC 27001-2022v2[EP].pdf')
        
        if not pdf_path.exists():
            print("❌ No se encuentra el archivo PDF")
            return False
            
        doc = fitz.open(str(pdf_path))
        texto_total = ""
        
        for page in doc:
            texto_total += page.get_text()
        
        doc.close()
        
        if len(texto_total) > 1000:
            print(f"✅ PDF leído correctamente ({len(texto_total)} caracteres)")
            return True
        else:
            print("❌ El PDF parece estar vacío o muy corto")
            return False
            
    except Exception as e:
        print(f"❌ Error leyendo PDF: {e}")
        return False

def test_embeddings():
    """Verifica que funcione la creación de embeddings"""
    print("\n🧪 Test 3: Embeddings...")
    try:
        from sentence_transformers import SentenceTransformer
        
        print("   Cargando modelo...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print("   Creando embeddings de prueba...")
        test_texts = [
            "La norma ISO 27001 es un estándar de seguridad de la información",
            "La certificación requiere una auditoría externa",
            "Los controles de seguridad son fundamentales"
        ]
        
        embeddings = model.encode(test_texts)
        
        if embeddings.shape == (3, 384):
            print("✅ Embeddings creados correctamente")
            return True, model
        else:
            print(f"❌ Forma incorrecta de embeddings: {embeddings.shape}")
            return False, None
            
    except Exception as e:
        print(f"❌ Error en embeddings: {e}")
        return False, None

def test_faiss(embeddings_model):
    """Verifica que funcione FAISS"""
    print("\n🧪 Test 4: FAISS...")
    try:
        import faiss
        import numpy as np
        
        # Crear embeddings de prueba
        test_chunks = [
            "ISO 27001 es una norma internacional para gestión de seguridad",
            "La certificación demuestra cumplimiento con estándares",
            "Las auditorías verifican la implementación de controles"
        ]
        
        embeddings = embeddings_model.encode(test_chunks)
        
        # Crear índice
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))
        
        # Probar búsqueda
        query = "¿Qué es ISO 27001?"
        query_embedding = embeddings_model.encode([query])
        distances, indices = index.search(query_embedding.astype('float32'), k=2)
        
        if len(indices[0]) == 2:
            print("✅ FAISS funcionando correctamente")
            return True
        else:
            print("❌ FAISS no devolvió resultados esperados")
            return False
            
    except Exception as e:
        print(f"❌ Error en FAISS: {e}")
        return False

def test_streamlit_app():
    """Verifica que el archivo de Streamlit sea válido"""
    print("\n🧪 Test 5: Archivo Streamlit...")
    try:
        app_path = Path('app_simple.py')
        if not app_path.exists():
            print("❌ No existe app_simple.py")
            return False
            
        with open(app_path, 'r', encoding='utf-8') as f:
            contenido = f.read()
            
        # Verificar componentes clave
        componentes_requeridos = [
            'import streamlit',
            'st.set_page_config',
            'def main():',
            'if __name__ == "__main__":'
        ]
        
        for componente in componentes_requeridos:
            if componente not in contenido:
                print(f"❌ Falta componente: {componente}")
                return False
        
        print("✅ Archivo Streamlit válido")
        return True
        
    except Exception as e:
        print(f"❌ Error verificando archivo: {e}")
        return False

def main():
    """Ejecuta todos los tests"""
    print("🔍 Verificación completa del sistema RAG")
    print("=" * 50)
    
    tests = [
        test_importaciones,
        test_lectura_pdf,
        lambda: test_embeddings()[0],
        lambda: test_faiss(test_embeddings()[1]) if test_embeddings()[1] else False,
        test_streamlit_app
    ]
    
    resultados = []
    for test in tests:
        try:
            resultado = test()
            resultados.append(resultado)
        except Exception as e:
            print(f"❌ Error inesperado: {e}")
            resultados.append(False)
    
    print("\n" + "=" * 50)
    print("📊 RESUMEN DE TESTS")
    print("=" * 50)
    
    aprobados = sum(resultados)
    total = len(resultados)
    
    print(f"✅ Tests aprobados: {aprobados}/{total}")
    
    if aprobados == total:
        print("🎉 ¡SISTEMA COMPLETAMENTE FUNCIONAL!")
        print("\n🚀 Para iniciar la interfaz web:")
        print("   source .venv/bin/activate")
        print("   streamlit run app_simple.py")
    else:
        print("⚠️ Hay problemas que resolver")
    
    return aprobados == total

if __name__ == "__main__":
    main()
