"""Install the candidate's pinned estimator set, or matching sibling checkouts."""
from pathlib import Path
import argparse,json,re,subprocess,sys

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-siblings",type=Path,help="Directory containing the nine estimator repositories")
    parser.add_argument("--minimal",action="store_true",help="Install only TWFE, did2s and BJS for the first example")
    parser.add_argument("--dry-run",action="store_true",help="Print the validated pip arguments without installing")
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    entries=json.loads((root/"estimator-lock.json").read_text(encoding="utf-8"))["estimators"]
    if args.minimal: entries=[x for x in entries if x["estimator"] in {"twfe","did2s","bjs"}]
    targets=[]
    for entry in entries:
        if args.from_siblings:
            path=args.from_siblings.resolve()/entry.get("sibling_directory", entry["package"])
            if not (path/"pyproject.toml").is_file():
                raise SystemExit(f"Missing estimator checkout: {path}")
            targets.append(str(path))
        else:
            commit=entry.get("commit") or ""
            if not re.fullmatch(r"[0-9a-f]{40}",commit):
                raise SystemExit("This working candidate is not pinned yet. Use --from-siblings with the complete candidate bundle.")
            suffix = "#subdirectory=" + entry["subdirectory"] if entry.get("subdirectory") else ""
            targets.append(f"git+https://github.com/{entry['repository']}.git@{commit}{suffix}")
    command=[sys.executable,"-m","pip","install",*targets]
    if args.dry_run: print(json.dumps(command,indent=2))
    else: subprocess.run(command,check=True)

if __name__=="__main__": main()
