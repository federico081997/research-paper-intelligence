"""Tests for the API bootstrap functions."""

from unittest.mock import Mock, patch

from research_paper_intelligence.api.bootstrap import create_search_service
from research_paper_intelligence.config import Settings


class TestCreateSearchService:
    """Test creation of the search service."""

    def test_loads_resources_and_creates_search_service(
        self,
        simple_settings: Settings,
    ) -> None:
        """Load resources and inject them into the SearchService."""
        paper_data = Mock()
        paper_repository = Mock()
        model = Mock()
        index = Mock()
        vectorizer = Mock()
        tfidf_matrix = Mock()
        search_service = Mock()

        with (
            patch(
                "research_paper_intelligence.api.bootstrap.get_settings",
                return_value=simple_settings,
            ) as get_settings_mock,
            patch(
                "research_paper_intelligence.api.bootstrap.load_data",
                return_value=paper_data,
            ) as load_data_mock,
            patch(
                "research_paper_intelligence.api.bootstrap.PaperRepository",
                return_value=paper_repository,
            ) as paper_repository_mock,
            patch(
                "research_paper_intelligence.api.bootstrap.get_model",
                return_value=model,
            ) as get_model_mock,
            patch(
                "research_paper_intelligence.api.bootstrap.load_faiss_index",
                return_value=index,
            ) as load_faiss_index_mock,
            patch(
                "research_paper_intelligence.api.bootstrap.load_tfidf_index",
                return_value=(vectorizer, tfidf_matrix),
            ) as load_tfidf_index_mock,
            patch(
                "research_paper_intelligence.api.bootstrap.SearchService",
                return_value=search_service,
            ) as search_service_mock,
        ):
            result = create_search_service()

        assert result is search_service

        get_settings_mock.assert_called_once_with()

        load_data_mock.assert_called_once_with(
            simple_settings.processed_papers_path
        )

        paper_repository_mock.assert_called_once_with(paper_data)

        get_model_mock.assert_called_once_with(
            simple_settings.embedding_model_name
        )

        load_faiss_index_mock.assert_called_once_with(
            simple_settings.faiss_index_papers_path
        )

        load_tfidf_index_mock.assert_called_once_with(
            simple_settings.tfidf_vectorizer_path,
            simple_settings.tfidf_matrix_path,
        )

        search_service_mock.assert_called_once_with(
            paper_repository=paper_repository,
            model=model,
            index=index,
            vectorizer=vectorizer,
            tfidf_matrix=tfidf_matrix,
            settings=simple_settings,
        )
