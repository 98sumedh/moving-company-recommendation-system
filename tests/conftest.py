import pytest

from movrec.assumptions import load_assumptions
from movrec.catalog import load_catalog
from movrec.generator.generate import generate

SMALL_N = 4000


@pytest.fixture(scope="session")
def A():
    return load_assumptions()


@pytest.fixture(scope="session")
def catalog():
    return load_catalog()


@pytest.fixture(scope="session")
def small_data():
    return generate(seed=7, n_bookings=SMALL_N)
