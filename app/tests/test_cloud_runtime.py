"""Smoke the committed dashboard bundle without pipeline dependencies.

Run with: python -m unittest discover -s app/tests -v
This suite deliberately does not use the pipeline's pytest conftest.
"""

from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


class CloudRuntimeTest(unittest.TestCase):
    def test_entrypoint_and_all_pages_without_pipeline_packages(self):
        script = textwrap.dedent('''
            import importlib.abc
            import pkgutil
            import sys

            class DashboardOnly(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname.split('.')[0] in {
                        'duckdb', 'kagglehub', 'geopandas', 'pykrige'
                    }:
                        raise ModuleNotFoundError(
                            f'Pipeline dependency imported: {fullname}', name=fullname
                        )

            sys.meta_path.insert(0, DashboardOnly())
            from streamlit.testing.v1 import AppTest
            from app import views

            entry = AppTest.from_file('app/streamlit_app.py', default_timeout=30)
            entry.run()
            assert not entry.exception, entry.exception

            for module in pkgutil.iter_modules(views.__path__):
                name = f'app.views.{module.name}'
                page = AppTest.from_string(
                    f'from importlib import import_module; '
                    f'import_module({name!r}).render()', default_timeout=30
                ).run()
                assert not page.exception, (name, page.exception)
                if module.name == 'vulnerability':
                    assert any('ND-GAIN' in h.value for h in page.header)
                print(f'PASS {name}')
        ''')
        result = subprocess.run(
            [sys.executable, '-c', script],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            timeout=180,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
