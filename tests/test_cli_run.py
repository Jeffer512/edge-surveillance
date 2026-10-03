import subprocess
import sys
import textwrap

# Imports cli.run with fastapi and uvicorn blocked, so a module-scope import of
# either one fails the same way it would on Termux, where pydantic-core has no
# wheel and cannot be built.
_BLOCK = textwrap.dedent(
    """
    import sys

    class Blocker:
        BLOCKED = {"fastapi", "uvicorn", "pydantic", "pydantic_core"}

        def find_module(self, name, path=None):
            return self.find_spec(name, path)

        def find_spec(self, name, path=None, target=None):
            root = name.split(".")[0]
            if root in self.BLOCKED:
                raise ImportError(f"No module named {root!r} (blocked by test)")
            return None

    sys.meta_path.insert(0, Blocker())
    import edge_surveillance.cli.run
    print("IMPORT_OK")
    """
)


def test_cli_run_imports_without_web_deps():
    result = subprocess.run(
        [sys.executable, "-c", _BLOCK],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "IMPORT_OK" in result.stdout

