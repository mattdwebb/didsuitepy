from pathlib import Path
import json,subprocess,sys
import pytest

@pytest.mark.integration
@pytest.mark.parametrize("example",["quickstart","all_rcs","panel_lpdid"])
def test_readme_examples(example):
    root=Path(__file__).resolve().parents[1]
    run=subprocess.run([sys.executable,str(root/"examples"/(example+".py"))],cwd=root,capture_output=True,text=True,check=True)
    expected=(root/"examples"/(example+"_output.txt")).read_text(encoding="utf-8")
    assert run.stdout.strip()==expected.strip()

def test_installer_has_all_nine_unique_backends():
    root=Path(__file__).resolve().parents[1]
    entries=json.loads((root/"estimator-lock.json").read_text(encoding="utf-8"))["estimators"]
    assert len(entries)==len({x["package"] for x in entries})==9
    assert {x["estimator"] for x in entries}=={"twfe","did2s","bjs","flex","csdid","undid","didint","stacked","lpdid"}

def test_csdid_installer_uses_public_version_and_python_subdirectory():
    root=Path(__file__).resolve().parents[1]
    entries=json.loads((root/"estimator-lock.json").read_text(encoding="utf-8"))["estimators"]
    cs=next(e for e in entries if e["estimator"]=="csdid")
    assert cs["repository"]=="mattdwebb/fastcsdid" and cs["version"]=="1.0.1"
    assert cs["subdirectory"]=="python" and cs["sibling_directory"]=="fastcsdid/python"
    run=subprocess.run([sys.executable,str(root/"tools/install_estimators.py"),"--dry-run"],check=True,capture_output=True,text=True)
    commands=json.loads(run.stdout)
    assert f"git+https://github.com/mattdwebb/fastcsdid.git@{cs['commit']}#subdirectory=python" in commands
    assert not any("mattdwebb/fastcsdidpy.git" in item for item in commands)
