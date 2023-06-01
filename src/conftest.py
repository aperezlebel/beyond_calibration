import os
import sys
from os.path import dirname, join, relpath

# import joblib before torch to avoid parallelization issue
# see https://github.com/joblib/joblib/issues/1420
import joblib

# Add root folder to path
path = os.path.dirname(os.path.realpath(__name__))
sys.path.append(path)


def pytest_addoption(parser):
    parser.addoption("--out", action="store", default="img/")
    parser.addoption("--inp", action="store", default="img/")
    parser.addoption("--njobs", action="store", default=1, type=int)
    parser.addoption(
        "--nocache", action="store", default=False, const=True, nargs="?", type=bool
    )


def already_parametrized(metafunc, name):
    """Check if a test has already been parametrized with a given name."""
    markers = metafunc.definition.keywords._markers

    if "pytestmark" not in markers:
        return False

    for mark in markers["pytestmark"]:
        if mark.args[0] == name:
            return True

    return False


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def pytest_generate_tests(metafunc):
    out_value = metafunc.config.option.out
    inp_value = metafunc.config.option.inp

    test_function_name = remove_prefix(metafunc.function.__name__, "test_")
    test_dirpath = dirname(metafunc.module.__file__)
    project_filepath = dirname(__file__)
    relative_test_dirpath = relpath(test_dirpath, project_filepath)

    if "out" in metafunc.fixturenames and out_value is not None:
        out = join(out_value, relative_test_dirpath, test_function_name)
        metafunc.parametrize("out", [out])

    if "inp" in metafunc.fixturenames and inp_value is not None:
        inp = join(inp_value, relative_test_dirpath, test_function_name)
        metafunc.parametrize("inp", [inp])

    if "n_jobs" in metafunc.fixturenames and (
        metafunc.config.option.njobs is not None
        and not already_parametrized(metafunc, "n_jobs")
    ):
        metafunc.parametrize("n_jobs", [metafunc.config.option.njobs])

    if "nocache" in metafunc.fixturenames and (
        metafunc.config.option.njobs is not None
        and not already_parametrized(metafunc, "nocache")
    ):
        metafunc.parametrize("nocache", [metafunc.config.option.nocache])
