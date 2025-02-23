# test_RAG_analyzer.py

import unittest
from unittest.mock import patch, MagicMock
from RAG_analyzer import GitHubRAGAnalyzer

# Test de inicialización
class TestRAGAnalyzer(unittest.TestCase):
    
    @patch('RAG_analyzer.GitHubAnalyzer')
    @patch('RAG_analyzer.ComplianceAnalyzer')
    @patch('RAG_analyzer.ChatGroq')
    def test_initialization(self, mock_chatgroq, mock_compliance_analyzer, mock_github_analyzer):
        # Inicialización sin pasar api_key explícita, se carga desde .env
        analyzer = GitHubRAGAnalyzer()
        self.assertEqual(analyzer.model_name, "mixtral-8x7b-32768")
        self.assertTrue(analyzer.api_key)  # La clave API debería haberse cargado
        self.assertIsNotNone(analyzer.github_analyzer)
        self.assertIsNotNone(analyzer.compliance_analyzer)

    # Test de detección del tipo de proyecto
    @patch('RAG_analyzer.GitHubAnalyzer')
    @patch('RAG_analyzer.ComplianceAnalyzer')
    @patch('RAG_analyzer.ChatGroq')
    def test_detect_project_type(self, mock_chatgroq, mock_compliance_analyzer, mock_github_analyzer):
        # Simulación de respuesta del LLM
        mock_chatgroq_instance = mock_chatgroq.return_value
        mock_chatgroq_instance.invoke.return_value.content = 'ml'
        
        analyzer = GitHubRAGAnalyzer(api_key="test_api_key")
        project_type = analyzer.detect_project_type("This is a machine learning project.")
        
        # Verificamos que el tipo de proyecto detectado sea 'ml'
        self.assertEqual(project_type, 'ml')
        
        # También probamos para NLP
        mock_chatgroq_instance.invoke.return_value.content = 'nlp'
        project_type = analyzer.detect_project_type("This is a NLP project.")
        self.assertEqual(project_type, 'nlp')

    # Test de análisis de requisitos del repositorio
    @patch('RAG_analyzer.GitHubAnalyzer.clone_repo')
    @patch('RAG_analyzer.GitHubAnalyzer.extract_text_from_repo')
    @patch('RAG_analyzer.GitHubAnalyzer.get_repo_stats')
    @patch('RAG_analyzer.ComplianceAnalyzer.extract_text_from_pdf')
    @patch('RAG_analyzer.GitHubRAGAnalyzer.detect_project_type')
    def test_analyze_requirements_completion(self, mock_detect_project_type, mock_extract_text_from_pdf, mock_get_repo_stats, mock_extract_text_from_repo, mock_clone_repo):
        # Simulamos la clonación del repositorio y la extracción de archivos
        mock_clone_repo.return_value = "/path/to/repo"
        mock_extract_text_from_repo.return_value = ["file1 content", "file2 content"]
        mock_get_repo_stats.return_value = {"commit_count": 10, "contributors": ["user1", "user2"], "languages": [{"name": "Python", "percentage": 80}]}
        mock_extract_text_from_pdf.return_value = "Briefing content"
        mock_detect_project_type.return_value = "ml"

        analyzer = GitHubRAGAnalyzer(api_key="test_api_key")
        result = analyzer.analyze_requirements_completion(repo_url="https://github.com/test/repo", briefing_path="path/to/briefing.pdf")
        
        # Verificamos que los datos devueltos son correctos
        self.assertIn("project_type", result)
        self.assertIn("repository_stats", result)
        self.assertIn("tier_analysis", result)
        self.assertEqual(result["project_type"], "ml")

    # Test de extracción de requisitos
    @patch('RAG_analyzer.GitHubRAGAnalyzer.llm.invoke')
    def test_extract_tier_requirements(self, mock_invoke):
        # Simulamos una respuesta JSON del LLM
        mock_invoke.return_value.content = '''
        {
            "nivel_esencial": ["Requisito 1", "Requisito 2"],
            "nivel_medio": ["Requisito 3"],
            "nivel_avanzado": ["Requisito 4"],
            "nivel_experto": []
        }
        '''
        analyzer = GitHubRAGAnalyzer(api_key="test_api_key")
        briefing_text = "This is a project briefing"
        result = analyzer.extract_tier_requirements(briefing_text)
        
        # Comprobamos la salida
        self.assertIn("nivel_esencial", result)
        self.assertIn("nivel_medio", result)
        self.assertEqual(len(result["nivel_esencial"]), 2)
        self.assertEqual(len(result["nivel_experto"]), 0)

if __name__ == "__main__":
    unittest.main()
