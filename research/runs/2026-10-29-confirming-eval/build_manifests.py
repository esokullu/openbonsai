import json, sys
sys.path.insert(0, "D:/openbonsai/work/code-gap-snapshot")
from openbonsai.eval import reasoning as R
for name, n, nf in (("humaneval", 164, 50), ("gsm8k", 1319, 100)):
    info, probs = R.load_problems(name, n, local_files_only=True)
    frozen = R.load_frozen(name)["problems"]
    assert [p["row"] for p in probs[:nf]] == [p["row"] for p in frozen], name
    json.dump({"info": info, "problems": probs}, open(f"manifests/{name}-{n}.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(name, n, info["content_sha256"], info["dataset"].get("sha256") if isinstance(info["dataset"], dict) else info["dataset"], "prefix ok")
