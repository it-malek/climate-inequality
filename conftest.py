"""Explicit synthetic-only output authorization for the repository test suite."""
import pytest

from src.prospective import prospective_execution


@pytest.fixture(scope="session", autouse=True)
def prospective_test_outputs(tmp_path_factory):
    # Includes fixture-created module/session temp directories; never the checkout.
    with prospective_execution(scratch=tmp_path_factory.getbasetemp()):
        yield
