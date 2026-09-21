from __future__ import annotations

import os
import shutil
import stat
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write_executable(path: Path, text: str) -> None:
    path.write_text(textwrap.dedent(text).lstrip(), encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


class SetupRuntimeTests(unittest.TestCase):
    def make_repo(self, *, envs: tuple[str, ...] = ()) -> tuple[Path, Path, Path]:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        repo = root / "repo"
        fake_bin = root / "bin"
        env_root = root / "envs"
        repo.mkdir()
        fake_bin.mkdir()
        env_root.mkdir()
        shutil.copy2(ROOT / "setup.sh", repo / "setup.sh")
        (repo / "requirements").mkdir()
        shutil.copy2(
            ROOT / "requirements" / "boltz2-cu128-constraints.txt",
            repo / "requirements" / "boltz2-cu128-constraints.txt",
        )
        (repo / "scripts").mkdir()
        (repo / "assets").mkdir()
        (repo / "examples").mkdir()
        (repo / "examples" / "demo_reference_1MEL_AB.cif").write_text("ref\n", encoding="utf-8")
        (repo / "assets" / "report.js").write_text("console.log('report');\n", encoding="utf-8")
        (repo / "scripts" / "runtime_state.py").write_text("# fake runtime_state\n", encoding="utf-8")
        (root / "envs.tsv").write_text(
            "".join(f"{name}\t{env_root / name}\n" for name in envs),
            encoding="utf-8",
        )
        for name in envs:
            self.create_env(env_root / name)
        self.write_fake_commands(root, repo, fake_bin, env_root)
        return repo, fake_bin, env_root

    def create_env(self, prefix: Path) -> None:
        (prefix / "bin").mkdir(parents=True, exist_ok=True)
        write_executable(
            prefix / "bin" / "python",
            """
            #!/usr/bin/env bash
            if [[ "${1:-}" == "--version" ]]; then echo "Python 3.12.9"; exit 0; fi
            if [[ "${1:-}" == "-c" ]]; then
              case "${2:-}" in
                *'torch.__version__'*) echo '2.7.0+cu128' ;;
              esac
              exit 0
            fi
            if [[ "${1:-}" == *"/scripts/runtime_state.py" ]]; then exit 0; fi
            cat >/dev/null
            exit 0
            """,
        )
        write_executable(
            prefix / "bin" / "pip",
            """
            #!/usr/bin/env bash
            prefix="$(cd "$(dirname "$0")/.." && pwd)"
            echo "pip[$prefix] $*" >> "${SETUP_TEST_LOG:?}"
            if [[ -n "${SETUP_FAIL_PIP:-}" && "$1" == "install" ]]; then exit 43; fi
            exit 0
            """,
        )
        write_executable(prefix / "bin" / "boltz", "#!/usr/bin/env bash\nexit 0\n")
        write_executable(prefix / "bin" / "hmmscan", "#!/usr/bin/env bash\nexit 0\n")

    def write_fake_commands(self, root: Path, repo: Path, fake_bin: Path, env_root: Path) -> None:
        template_env = root / "template_env"
        self.create_env(template_env)
        (template_env / "bin" / "hmmscan").unlink()
        create_env_script = root / "create_env.sh"
        write_executable(
            create_env_script,
            f"""
            #!/usr/bin/env bash
            prefix="$1"
            mkdir -p "$prefix/bin"
            cp -a "{template_env / 'bin'}/." "$prefix/bin/"
            """,
        )
        write_executable(
            fake_bin / "mamba",
            f"""
            #!/usr/bin/env bash
            set -eu
            echo "mamba $*" >> "{root / 'calls.log'}"
            state="{root / 'envs.tsv'}"
            env_root="{env_root}"
            create_env="{create_env_script}"
            if [[ "${{1:-}}" == "env" && "${{2:-}}" == "list" ]]; then
              [[ -z "${{SETUP_FAIL_ENV_LIST:-}}" ]] || exit 31
              created_name="$(cat "{root / 'created_name'}" 2>/dev/null || true)"
              if [[ -n "${{SETUP_HIDE_CREATED_FROM_LIST:-}}" && -n "$created_name" ]]; then
                grep -v -F "$created_name" "$state" || true
                exit 0
              fi
              cat "$state"
              exit 0
            fi
            if [[ "${{1:-}}" == "create" ]]; then
              [[ -z "${{SETUP_FAIL_CREATE:-}}" ]] || exit 32
              name=""
              while [[ "$#" -gt 0 ]]; do
                if [[ "$1" == "-n" ]]; then name="$2"; shift 2; continue; fi
                shift
              done
              [[ -n "$name" ]] || exit 33
              prefix="$env_root/$name"
              "$create_env" "$prefix"
              printf '%s' "$name" > "{root / 'created_name'}"
              printf '%s\\t%s\\n' "$name" "$prefix" >> "$state"
              exit 0
            fi
            if [[ "${{1:-}}" == "install" ]]; then
              [[ -z "${{SETUP_FAIL_CONDA_INSTALL:-}}" ]] || exit 34
              name=""
              while [[ "$#" -gt 0 ]]; do
                if [[ "$1" == "-n" ]]; then name="$2"; shift 2; continue; fi
                shift
              done
              if [[ -n "$name" ]]; then
                printf '#!/usr/bin/env bash\\nexit 0\\n' > "$env_root/$name/bin/hmmscan"
                chmod +x "$env_root/$name/bin/hmmscan"
              fi
              exit 0
            fi
            exit 35
            """,
        )
        write_executable(
            fake_bin / "curl",
            """
            #!/usr/bin/env bash
            out=""
            while [[ "$#" -gt 0 ]]; do
              if [[ "$1" == "-o" ]]; then out="$2"; shift 2; continue; fi
              shift
            done
            [[ -n "$out" ]] || exit 2
            : > "$out"
            """,
        )
        write_executable(fake_bin / "nvidia-smi", "#!/usr/bin/env bash\nexit 1\n")
        write_executable(
            repo / "run.sh",
            f"""
            #!/usr/bin/env bash
            echo "${{BOLTZ_ENV:-}}" > "{root / 'verify_env.txt'}"
            [[ -z "${{SETUP_FAIL_VERIFY:-}}" ]] || exit 44
            name=""
            while [[ "$#" -gt 0 ]]; do
              if [[ "$1" == "--name" ]]; then name="$2"; shift 2; continue; fi
              shift
            done
            mkdir -p "outputs/$name/analysis" "outputs/$name/report"
            echo '{{}}' > "outputs/$name/analysis/results.json"
            echo html > "outputs/$name/report/index.html"
            exit 0
            """,
        )

    def run_setup(self, repo: Path, fake_bin: Path, env: dict[str, str] | None = None,
                  *args: str) -> subprocess.CompletedProcess[str]:
        run_env = os.environ.copy()
        for key in tuple(run_env):
            if key in {"ENV_NAME", "REUSE_ENV", "VERIFY"} or key.startswith(("INSTALL_", "SETUP_")):
                run_env.pop(key, None)
        run_env.update({
            "PATH": f"{fake_bin}:{run_env.get('PATH', '')}",
            "SETUP_TEST_LOG": str(repo.parent / "pip.log"),
        })
        if env:
            run_env.update(env)
        return subprocess.run(
            ["bash", "setup.sh", *args],
            cwd=repo,
            env=run_env,
            text=True,
            capture_output=True,
            check=False,
        )

    def assert_pointer(self, repo: Path, prefix: Path) -> None:
        self.assertEqual((repo / ".boltz_env").read_text(encoding="utf-8").strip(), str(prefix))
        self.assertEqual((repo.parent / "verify_env.txt").read_text(encoding="utf-8").strip(), str(prefix))

    def test_fresh_install_creates_default_env_and_uses_exact_prefix(self) -> None:
        repo, fake_bin, env_root = self.make_repo()
        got = self.run_setup(repo, fake_bin, None, "--verify")
        self.assertEqual(got.returncode, 0, got.stderr + got.stdout)
        self.assert_pointer(repo, env_root / "boltz2")
        self.assertIn("boltz2\t" + str(env_root / "boltz2"), (repo.parent / "envs.tsv").read_text())

    def test_existing_default_chooses_numbered_fresh_env(self) -> None:
        repo, fake_bin, env_root = self.make_repo(envs=("boltz2",))
        got = self.run_setup(repo, fake_bin, None, "--verify")
        self.assertEqual(got.returncode, 0, got.stderr + got.stdout)
        self.assert_pointer(repo, env_root / "boltz2_2")
        self.assertTrue((env_root / "boltz2" / "bin" / "python").exists())
        pip_log = (repo.parent / "pip.log").read_text(encoding="utf-8")
        self.assertIn(f"pip[{env_root / 'boltz2_2'}] install", pip_log)
        self.assertNotIn(f"pip[{env_root / 'boltz2'}] install", pip_log)
        calls = (repo.parent / "calls.log").read_text(encoding="utf-8")
        self.assertIn("mamba create -y -n boltz2_2 python=3.12 pip", calls)
        self.assertIn("mamba install -y -n boltz2_2", calls)

    def test_existing_default_and_occupied_suffix_choose_next_suffix(self) -> None:
        repo, fake_bin, env_root = self.make_repo(envs=("boltz2", "boltz2_2"))
        got = self.run_setup(repo, fake_bin, None, "--verify")
        self.assertEqual(got.returncode, 0, got.stderr + got.stdout)
        self.assert_pointer(repo, env_root / "boltz2_3")

    def test_arbitrary_env_name_collision_gets_suffix(self) -> None:
        repo, fake_bin, env_root = self.make_repo(envs=("custom",))
        got = self.run_setup(repo, fake_bin, {"ENV_NAME": "custom"}, "--verify")
        self.assertEqual(got.returncode, 0, got.stderr + got.stdout)
        self.assert_pointer(repo, env_root / "custom_2")

    def test_real_doctor_uses_published_pointer_without_boltz_env(self) -> None:
        repo, fake_bin, env_root = self.make_repo(envs=("boltz2",))
        got = self.run_setup(repo, fake_bin, None, "--verify")
        self.assertEqual(got.returncode, 0, got.stderr + got.stdout)
        chosen = env_root / "boltz2_2"
        self.assert_pointer(repo, chosen)
        shutil.copy2(ROOT / "run.sh", repo / "run.sh")

        run_env = os.environ.copy()
        for key in tuple(run_env):
            if key == "BOLTZ_ENV" or key.startswith("SETUP_"):
                run_env.pop(key, None)
        run_env["PATH"] = f"{fake_bin}:{run_env.get('PATH', '')}"
        doctor = subprocess.run(
            ["bash", "run.sh", "--doctor"],
            cwd=repo,
            env=run_env,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(doctor.returncode, 0, doctor.stderr + doctor.stdout)
        self.assertIn(f"실행 환경: {chosen}", doctor.stdout)

    def test_reuse_env_keeps_legacy_opt_in_without_create(self) -> None:
        repo, fake_bin, env_root = self.make_repo(envs=("boltz2",))
        got = self.run_setup(repo, fake_bin, None, "--reuse-env", "--verify")
        self.assertEqual(got.returncode, 0, got.stderr + got.stdout)
        self.assert_pointer(repo, env_root / "boltz2")
        calls = (repo.parent / "calls.log").read_text(encoding="utf-8")
        self.assertNotIn("mamba create", calls)

    def test_env_list_failure_fails_without_pointer(self) -> None:
        repo, fake_bin, _env_root = self.make_repo()
        got = self.run_setup(repo, fake_bin, {"SETUP_FAIL_ENV_LIST": "1"}, "--verify")
        self.assertNotEqual(got.returncode, 0)
        self.assertFalse((repo / ".boltz_env").exists())

    def test_create_failure_fails_without_pointer_or_verify(self) -> None:
        repo, fake_bin, _env_root = self.make_repo()
        got = self.run_setup(repo, fake_bin, {"SETUP_FAIL_CREATE": "1"}, "--verify")
        self.assertNotEqual(got.returncode, 0)
        self.assertFalse((repo / ".boltz_env").exists())
        self.assertFalse((repo.parent / "verify_env.txt").exists())

    def test_package_install_failure_fails_without_pointer_or_verify(self) -> None:
        repo, fake_bin, _env_root = self.make_repo()
        got = self.run_setup(repo, fake_bin, {"SETUP_FAIL_PIP": "1"}, "--verify")
        self.assertNotEqual(got.returncode, 0)
        self.assertFalse((repo / ".boltz_env").exists())
        self.assertFalse((repo.parent / "verify_env.txt").exists())

    def test_verify_failure_fails_without_pointer_but_uses_chosen_prefix(self) -> None:
        repo, fake_bin, env_root = self.make_repo(envs=("boltz2",))
        got = self.run_setup(repo, fake_bin, {"SETUP_FAIL_VERIFY": "1"}, "--verify")
        self.assertNotEqual(got.returncode, 0)
        self.assertFalse((repo / ".boltz_env").exists())
        self.assertEqual((repo.parent / "verify_env.txt").read_text(encoding="utf-8").strip(),
                         str(env_root / "boltz2_2"))

    def test_verify_failure_preserves_existing_pointer(self) -> None:
        repo, fake_bin, env_root = self.make_repo(envs=("boltz2",))
        old_prefix = env_root / "old"
        (repo / ".boltz_env").write_text(str(old_prefix) + "\n", encoding="utf-8")
        got = self.run_setup(repo, fake_bin, {"SETUP_FAIL_VERIFY": "1"}, "--verify")
        self.assertNotEqual(got.returncode, 0)
        self.assertEqual((repo / ".boltz_env").read_text(encoding="utf-8").strip(), str(old_prefix))

    def test_successful_sequential_setup_rerun_installs_next_suffix(self) -> None:
        repo, fake_bin, env_root = self.make_repo()
        first = self.run_setup(repo, fake_bin, None, "--verify")
        self.assertEqual(first.returncode, 0, first.stderr + first.stdout)
        second = self.run_setup(repo, fake_bin, None, "--verify")
        self.assertEqual(second.returncode, 0, second.stderr + second.stdout)
        self.assert_pointer(repo, env_root / "boltz2_2")
        self.assertIn("boltz2_2\t" + str(env_root / "boltz2_2"),
                      (repo.parent / "envs.tsv").read_text(encoding="utf-8"))

    def test_post_create_missing_from_env_list_fails_without_pointer(self) -> None:
        repo, fake_bin, _env_root = self.make_repo()
        got = self.run_setup(repo, fake_bin, {"SETUP_HIDE_CREATED_FROM_LIST": "1"}, "--verify")
        self.assertNotEqual(got.returncode, 0)
        self.assertFalse((repo / ".boltz_env").exists())
        self.assertFalse((repo.parent / "verify_env.txt").exists())


if __name__ == "__main__":
    unittest.main()
