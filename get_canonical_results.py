"""Return the canonical cut list for each movie in the project.

Reads canonical/sources.json, which declares what is canonical per movie:
  base       = whole-film reference cut list (usually MovieNet, synced to our copy)
  overrides  = human-verified canonical cuts that REPLACE the base inside a region
               (e.g. Vertigo's Nightmare, which we hand-labelled)

Resolves each movie into a single canonical cut list and writes
canonical/canonical_results.json. Read-only: does NOT process any video / touch the GPU.

  python get_canonical_results.py            # summary
  python get_canonical_results.py --json      # dump full resolved cut lists
"""
import argparse, json, os


def load_cuts(spec):
    if not spec or spec.get("kind") == "none" or not spec.get("path"):
        return None
    p = spec["path"]
    if not os.path.exists(p):
        return None
    return sorted(json.load(open(p)).get("cuts", []))


def resolve(movie):
    base = load_cuts(movie.get("base"))
    if base is None:
        return None, "none"
    cuts = list(base)
    kind = movie["base"]["kind"]
    for ov in movie.get("overrides", []):
        human = load_cuts(ov)
        if human is None:
            continue
        lo, hi = ov["region"]
        cuts = [c for c in cuts if not (lo <= c <= hi)] + [c for c in human if lo <= c <= hi]
        kind = "movienet+human" if kind == "movienet" else "human"
    return sorted(cuts), kind


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="canonical/sources.json")
    ap.add_argument("--out", default="canonical/canonical_results.json")
    ap.add_argument("--json", action="store_true", help="print full resolved cut lists")
    a = ap.parse_args()

    src = json.load(open(a.sources))
    results = {}
    print(f"{'movie':32} {'canonical source':16} {'cuts':>6}")
    print("-" * 58)
    for m in src["movies"]:
        cuts, kind = resolve(m)
        if cuts is None:
            print(f"{m['name'][:32]:32} {'(none yet)':16} {'-':>6}   {m.get('_status','')[:30]}")
            results[m["imdb"]] = {"name": m["name"], "canonical": None, "source": "none",
                                  "status": m.get("_status", "")}
            continue
        print(f"{m['name'][:32]:32} {kind:16} {len(cuts):6d}")
        results[m["imdb"]] = {"name": m["name"], "video": m.get("video"),
                              "source": kind, "n_cuts": len(cuts), "cuts": cuts}

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(results, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")
    print("source key: 'movienet' = machine reference (not gold), "
          "'movienet+human' = MovieNet with hand-verified region(s), 'none' = unlabelled")
    if a.json:
        for imdb, r in results.items():
            if r.get("cuts"):
                print(f"\n{r['name']} ({imdb}): {r['n_cuts']} cuts")
                print(r["cuts"])


if __name__ == "__main__":
    main()
