import unittest
from unittest.mock import Mock, patch, mock_open
import os
import pandas as pd
import numpy as np
from github import Github, Repository, Branch, Commit, ContentFile, GithubException
from github_analyzer import GitHubAnalyzer

class TestGitHubAnalyzer(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        with patch.dict('os.environ', {'GITHUB_TOKEN': 'dummy_token'}):
            self.analyzer = GitHubAnalyzer()
        self.test_repo_url = "https://github.com/user/repo"
        
    def test_extract_repo_name(self):
        """Test repository name extraction from URLs"""
        test_cases = [
            ("https://github.com/user/repo", "user/repo"),
            ("https://github.com/user/repo/", "user/repo"),
            ("https://github.com/user/repo/tree/main", "user/repo"),
            ("https://github.com/org/project/tree/develop/folder", "org/project")
        ]
        
        for url, expected in test_cases:
            result = self.analyzer._extract_repo_name(url)
            self.assertEqual(result, expected)

    @patch('github.Github')
    def test_get_repo_stats_success(self, mock_github):
        """Test successful repository statistics retrieval"""
        # Mock repository and its components
        mock_repo = Mock(spec=Repository.Repository)
        mock_branch = Mock(spec=Branch.Branch)
        mock_commit = Mock(spec=Commit.Commit)
        mock_author = Mock()
        
        # Configure mocks
        mock_github.return_value.get_rate_limit.return_value.core.remaining = 1000
        mock_github.return_value.get_repo.return_value = mock_repo
        mock_branch.name = "main"
        mock_author.login = "test_user"
        mock_commit.author = mock_author
        
        mock_repo.get_branches.return_value = [mock_branch]
        mock_repo.get_commits.return_value = [mock_commit]
        mock_repo.get_languages.return_value = {"Python": 1000, "JavaScript": 500}
        
        # Execute test
        result = self.analyzer.get_repo_stats(self.test_repo_url)
        
        # Verify results
        self.assertEqual(result["branches"], ["main"])
        self.assertEqual(result["commit_count"], 1)
        self.assertIn("test_user", result["contributors"])
        self.assertEqual(len(result["languages"]), 2)

    @patch('github.Github')
    def test_get_repo_stats_rate_limit_exceeded(self, mock_github):
        """Test handling of API rate limit exceeded"""
        mock_github.return_value.get_rate_limit.return_value.core.remaining = 0
        
        result = self.analyzer.get_repo_stats(self.test_repo_url)
        self.assertEqual(result["error"], "API rate limit exceeded")

    @patch('github.Github')
    def test_get_repo_stats_error(self, mock_github):
        """Test error handling in repository statistics retrieval"""
        mock_github.return_value.get_repo.side_effect = GithubException(404, "Not found")
        
        result = self.analyzer.get_repo_stats(self.test_repo_url)
        self.assertEqual(result["commit_count"], 0)
        self.assertEqual(result["branches"], [])
        self.assertEqual(result["contributors"], {})
        self.assertEqual(result["languages"], [])

    @patch('os.path.exists')
    @patch('os.makedirs')
    @patch('os.system')
    @patch('github.Github')
    def test_clone_repo_success(self, mock_github, mock_system, mock_makedirs, mock_exists):
        """Test successful repository cloning"""
        # Mock repository and content
        mock_repo = Mock(spec=Repository.Repository)
        mock_content = Mock(spec=ContentFile.ContentFile)
        
        # Configure mocks
        mock_exists.return_value = True
        mock_content.type = "file"
        mock_content.path = "test.py"
        mock_content.decoded_content = b"test content"
        mock_repo.get_contents.return_value = [mock_content]
        mock_github.return_value.get_repo.return_value = mock_repo
        
        # Mock file operations
        mock_file = mock_open()
        with patch('builtins.open', mock_file):
            result = self.analyzer.clone_repo(self.test_repo_url)
        
        self.assertEqual(result, "cloned_repo")
        mock_makedirs.assert_called()
        mock_file().write.assert_called_with(b"test content")

    @patch('os.path.exists')
    @patch('os.makedirs')
    @patch('matplotlib.pyplot')
    @patch('seaborn.barplot')
    def test_generate_visualizations(self, mock_barplot, mock_plt, mock_makedirs, mock_exists):
        """Test visualization generation"""
        mock_exists.return_value = False
        
        test_stats = {
            'branches': ['main', 'develop'],
            'commit_count': 10,
            'contributors': {'user1': 5, 'user2': 5},
            'languages': [{'name': 'Python', 'percentage': 80}]
        }
        
        self.analyzer.generate_visualizations(test_stats)
        
        mock_makedirs.assert_called_once()
        self.assertEqual(mock_plt.figure.call_count, 2)
        self.assertEqual(mock_barplot.call_count, 2)
        mock_plt.savefig.assert_called()

    def test_extract_text_from_repo(self):
        """Test text extraction from repository files"""
        test_files = {
            'test.py': 'print("Hello")',
            'test.md': '# Header',
            'test.txt': 'Text content',
            'test.js': 'console.log("test")',
            'test.jpg': 'binary content'  # Should be ignored
        }
        
        with patch('os.walk') as mock_walk, \
             patch('builtins.open', mock_open(read_data='file content')):
            
            mock_walk.return_value = [
                ('root', [], list(test_files.keys()))
            ]
            
            result = self.analyzer.extract_text_from_repo()
            
            # Should extract from supported files only
            self.assertEqual(len(result), 4)  # Excluding .jpg
            self.assertTrue(all(content == 'file content' for content in result))

if __name__ == '__main__':
    unittest.main()