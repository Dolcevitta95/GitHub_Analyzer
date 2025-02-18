import os
from typing import List, Dict, Any
from git import Repo
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
import tempfile
import logging
import datetime

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

    def process_repository(self, repo_url: str, briefing: str) -> Dict[str, Any]:
        """
        Procesa un repositorio de GitHub y lo compara con el briefing.
        
        Args:
            repo_url: URL del repositorio de GitHub
            briefing: Briefing técnico del proyecto
            
        Returns:
            Dict con los resultados del análisis
        """
        try:
            # Clonar repositorio
            with tempfile.TemporaryDirectory() as temp_dir:
                self.logger.info(f"Clonando repositorio: {repo_url}")
                Repo.clone_from(repo_url, temp_dir)
                
                # Procesar archivos
                documents = self._process_files(temp_dir)
                
                # Crear vector store
                vector_store = Chroma.from_documents(
                    documents,
                    self.embeddings,
                    persist_directory=self.vector_store_path
                )
                
                # Crear chain de análisis
                qa_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=vector_store.as_retriever()
                )
                
                # Realizar análisis
                return self._analyze_repository(qa_chain, briefing)
                
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

    def _analyze_repository(self, qa_chain: RetrievalQA, briefing: str) -> Dict[str, Any]:
        """
        Realiza el análisis del repositorio comparándolo con el briefing.
        """
        analysis_template = """
        Analiza el código del repositorio y proporciona un informe detallado considerando:
        1. Cumplimiento del briefing: {briefing}
        2. Tecnologías utilizadas
        3. Calidad del código
        4. Estructura del proyecto
        5. Recomendaciones de mejora
        
        Por favor, proporciona un análisis detallado para cada punto.
        """
        
        prompt = analysis_template.format(briefing=briefing)
        response = qa_chain.run(prompt)
        
        return {
            "análisis_completo": response,
            "fecha_análisis": str(datetime.now())
        }