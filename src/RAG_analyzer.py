from datetime import datetime
import os
from typing import List, Dict, Any
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
import logging

class GitHubRAGAnalyzer:
    def __init__(
        self,
        vector_store_path: str = "./vector_store",
        model_name: str = "mistral"  # Nombre del modelo en Ollama
    ):
        """
        Inicializa el analizador RAG para repositorios de GitHub usando Ollama.
        
        Args:
            vector_store_path: Ruta donde se almacenará la base de datos vectorial
            model_name: Nombre del modelo en Ollama (por defecto 'mistral')
        """
        self.vector_store_path = vector_store_path
        self.model_name = model_name
        self.setup_logging()
        self.initialize_components()

    def setup_logging(self):
        """Configura el sistema de logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def initialize_components(self):
        """Inicializa los componentes necesarios para RAG"""
        # Configurar Ollama para embeddings y LLM
        self.embeddings = OllamaEmbeddings(model=self.model_name)
        self.llm = Ollama(model=self.model_name)

        # Configurar el divisor de texto
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

    def process_repository(self, repo_path: str) -> Dict[str, Any]:
        """
        Procesa un repositorio de GitHub y devuelve los documentos procesados.
        
        Args:
            repo_path: Ruta al repositorio clonado
            
        Returns:
            Dict con los documentos procesados y la fecha de análisis
        """
        try:
            # Procesar archivos
            documents = self._process_files(repo_path)
            
            # Crear vector store (opcional, si Andrea lo necesita)
            vector_store = Chroma.from_documents(
                documents,
                self.embeddings,
                persist_directory=self.vector_store_path
            )
            
            # Devolver los documentos procesados y la fecha de análisis
            return {
                "documentos_procesados": documents,
                "fecha_análisis": str(datetime.now())
            }
            
        except Exception as e:
            self.logger.error(f"Error procesando repositorio: {str(e)}")
            raise

    def _process_files(self, repo_path: str) -> List[Any]:
        """
        Procesa todos los archivos del repositorio.
        
        Args:
            repo_path: Ruta al repositorio clonado
            
        Returns:
            Lista de documentos procesados
        """
        documents = []
        for root, _, files in os.walk(repo_path):
            for file in files:
                if self._should_process_file(file):
                    try:
                        file_path = os.path.join(root, file)
                        loader = TextLoader(file_path)
                        documents.extend(loader.load())
                    except Exception as e:
                        self.logger.warning(f"Error procesando archivo {file}: {str(e)}")
        
        return self.text_splitter.split_documents(documents)

    def _should_process_file(self, filename: str) -> bool:
        """
        Determina si un archivo debe ser procesado basado en su extensión.
        """
        valid_extensions = {
            '.py', '.js', '.java', '.cpp', '.c', '.h', '.cs', '.php',
            '.rb', '.go', '.rs', '.swift', '.kt', '.ts', '.html', '.css',
            '.md', '.txt', '.json', '.yaml', '.yml'
        }
        return any(filename.endswith(ext) for ext in valid_extensions)