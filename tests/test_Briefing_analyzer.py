import unittest
from unittest.mock import Mock, patch, mock_open
import numpy as np
from compliance_analyzer import ComplianceAnalyzer

class TestComplianceAnalyzer(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.analyzer = ComplianceAnalyzer()
        self.sample_text = "This is a sample document text"
        self.mock_embedding = np.array([0.1, 0.2, 0.3])
        
    @patch('fitz.open')
    def test_extract_text_from_pdf_success(self, mock_fitz_open):
        """Test successful PDF text extraction"""
        # Mock PDF pages
        mock_page = Mock()
        mock_page.get_text.return_value = "Test content"
        mock_doc = Mock()
        mock_doc.__iter__.return_value = [mock_page, mock_page]
        mock_fitz_open.return_value = mock_doc

        result = self.analyzer.extract_text_from_pdf("test.pdf")
        self.assertEqual(result, "Test content Test content")
        mock_fitz_open.assert_called_once_with("test.pdf")

    @patch('fitz.open')
    def test_extract_text_from_pdf_failure(self, mock_fitz_open):
        """Test PDF text extraction with error"""
        mock_fitz_open.side_effect = Exception("PDF Error")
        
        result = self.analyzer.extract_text_from_pdf("test.pdf")
        self.assertEqual(result, "")

    @patch('langchain_huggingface.HuggingFaceEmbeddings.embed_query')
    def test_check_compliance_with_briefing_success(self, mock_embed_query):
        """Test successful compliance check"""
        # Mock embeddings
        mock_embed_query.side_effect = [
            np.array([0.1, 0.2, 0.3]),  # briefing embedding
            np.array([0.1, 0.2, 0.3]),  # high similarity doc
            np.array([-0.1, -0.2, -0.3])  # low similarity doc
        ]

        repo_docs = ["Document 1", "Document 2"]
        briefing_text = "Briefing text"

        results = self.analyzer.check_compliance_with_briefing(repo_docs, briefing_text)

        self.assertEqual(len(results), 2)
        self.assertTrue(results[0]["compliant"])  # High similarity should be compliant
        self.assertFalse(results[1]["compliant"])  # Low similarity should not be compliant

    @patch('langchain_huggingface.HuggingFaceEmbeddings.embed_query')
    def test_check_compliance_with_briefing_error(self, mock_embed_query):
        """Test compliance check with error"""
        mock_embed_query.side_effect = Exception("Embedding Error")

        results = self.analyzer.check_compliance_with_briefing(["doc"], "briefing")
        self.assertEqual(results, [])

    @patch.object(ComplianceAnalyzer, 'extract_text_from_pdf')
    @patch.object(ComplianceAnalyzer, 'check_compliance_with_briefing')
    def test_analyze_repository_compliance_success(self, mock_check_compliance, mock_extract_text):
        """Test successful repository compliance analysis"""
        mock_extract_text.return_value = "Briefing content"
        mock_check_compliance.return_value = [
            {"section": "doc1", "similarity": 80, "compliant": True},
            {"section": "doc2", "similarity": 60, "compliant": False}
        ]

        result = self.analyzer.analyze_repository_compliance(["doc1", "doc2"], "briefing.pdf")

        self.assertEqual(result["overall_compliance"], 50.0)
        self.assertEqual(result["total_sections"], 2)
        self.assertEqual(result["compliant_sections"], 1)
        self.assertEqual(len(result["detailed_results"]), 2)

    @patch.object(ComplianceAnalyzer, 'extract_text_from_pdf')
    def test_analyze_repository_compliance_empty_briefing(self, mock_extract_text):
        """Test repository compliance analysis with empty briefing"""
        mock_extract_text.return_value = ""

        result = self.analyzer.analyze_repository_compliance(["doc"], "briefing.pdf")

        self.assertEqual(result["overall_compliance"], 0)
        self.assertEqual(result["total_sections"], 0)
        self.assertEqual(result["compliant_sections"], 0)
        self.assertEqual(result["detailed_results"], [])

    def test_threshold_value(self):
        """Test default threshold value"""
        self.assertEqual(self.analyzer.threshold, 0.7)

if __name__ == '__main__':
    unittest.main()