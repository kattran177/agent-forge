"""
Tests for tools — search and text processing.
"""

from unittest.mock import patch, MagicMock

from src.tools.search import web_search, format_search_results, SearchResult
from src.tools.text import summarise_text, extract_key_points, format_task_list


class TestWebSearch:
    """Test the DuckDuckGo search tool."""

    @patch("src.tools.search.DDGS")
    def test_web_search_returns_results(self, mock_ddgs_class):
        """Search should return SearchResult objects."""
        mock_instance = MagicMock()
        mock_instance.__enter__ = MagicMock(return_value=mock_instance)
        mock_instance.__exit__ = MagicMock(return_value=False)
        mock_instance.text.return_value = [
            {"title": "Test Result", "href": "https://example.com", "body": "A test snippet"},
        ]
        mock_ddgs_class.return_value = mock_instance

        results = web_search("test query", max_results=1)

        assert len(results) == 1
        assert results[0].title == "Test Result"
        assert results[0].url == "https://example.com"

    @patch("src.tools.search.DDGS")
    def test_web_search_handles_errors(self, mock_ddgs_class):
        """Search should return empty list on error, not crash."""
        mock_ddgs_class.side_effect = Exception("Network error")

        results = web_search("failing query")

        assert results == []


class TestFormatSearchResults:
    """Test formatting search results for LLM consumption."""

    def test_format_results(self):
        results = [
            SearchResult(title="First", url="https://a.com", snippet="Snippet A"),
            SearchResult(title="Second", url="https://b.com", snippet="Snippet B"),
        ]
        formatted = format_search_results(results)

        assert "[1]" in formatted
        assert "[2]" in formatted
        assert "First" in formatted
        assert "https://a.com" in formatted

    def test_format_empty_results(self):
        assert "No search results" in format_search_results([])


class TestSummariseText:
    """Test text truncation."""

    def test_short_text_unchanged(self):
        text = "This is short."
        assert summarise_text(text, max_chars=100) == text

    def test_long_text_truncated(self):
        text = "A" * 3000
        result = summarise_text(text, max_chars=100)
        assert len(result) < 200
        assert "[... truncated]" in result

    def test_truncates_at_sentence_boundary(self):
        text = "First sentence. Second sentence. " + "A" * 2000
        result = summarise_text(text, max_chars=50)
        assert result.endswith("[... truncated]")


class TestExtractKeyPoints:
    """Test key point extraction."""

    def test_extracts_signal_sentences(self):
        text = (
            "Python is a language. "
            "The key advantage of Python is readability. "
            "It was created in 1991. "
            "An important feature is dynamic typing."
        )
        points = extract_key_points(text)
        assert len(points) >= 2
        assert any("key advantage" in p.lower() for p in points)

    def test_empty_text_returns_empty(self):
        assert extract_key_points("") == []


class TestFormatTaskList:
    """Test task list formatting."""

    def test_format_tasks(self):
        tasks = [
            {"description": "Research AI", "status": "completed"},
            {"description": "Write report", "status": "pending"},
        ]
        result = format_task_list(tasks)
        assert "✅" in result
        assert "⏳" in result

    def test_empty_list(self):
        assert "No tasks" in format_task_list([])
