import pytest

from app import lang


@pytest.fixture(autouse=True)
def _indonesian():
    """The existing suite asserts Indonesian text; English tests opt in with the `english` fixture."""
    lang.set_language("id")
    yield
    lang.set_language("id")
