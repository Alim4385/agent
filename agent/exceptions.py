"""Layihəyə xas xəta tipləri — səbəbi konkret ayırd etmək üçün."""


class WebAgentError(Exception):
    """Bütün agent xətalarının əcdadı."""


class SearchError(WebAgentError):
    """Axtarış provideri ilə bağlı problem."""


class FetchError(WebAgentError):
    """Səhifə məzmunu çəkilərkən problem."""


class LLMConnectionError(WebAgentError):
    """LLM serverinə qoşulma problemi."""


class LLMResponseError(WebAgentError):
    """LLM serverindən gözlənilməz cavab formatı."""
