"""
Part D - Run the fixed 8-question evaluation set from the assignment and
write a markdown results table. "Correct" is judged by you against the
manual (fill in the correct? / notes columns after eyeballing each answer
against the source pages) — this script records the pipeline's output so
that judgment call is documented, not hidden.
"""
import importlib.util
import json
import os

# Python module names can't start with a digit, so "import retrieve_query"
# can never resolve to 04_retrieve_query.py by normal import machinery.
# Load it directly from its file path instead.
_this_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "retrieve_query", os.path.join(_this_dir, "04_retrieve_query.py")
)
_retrieve_query = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_retrieve_query)
answer = _retrieve_query.answer

QUESTIONS = [
    "How many slots does the 1830 PSS-8 shelf provide, and what is its rack-unit (RU) footprint?",
    "What rack-unit footprint does the 1830 PSS-32 shelf have, and how many slots does it provide?",
    "What are the two software load-lines supported by the 1830 PSS system?",
    "Which fan units are supported on the 1830 PSS-32 shelf?",
    "Which fan unit(s) are used on the 1830 PSS-16II shelf?",
    "Name the power filter cards supported on the 1830 PSS-8 shelf.",
    "What is the required horizontal rack aperture for mounting a 1830 PSS-8 shelf, and which common aperture size is explicitly NOT supported?",
    "What is the maximum optical reach, in kilometers, of the 1830 PSS-8 shelf without amplification?",
]


def main(out_md="../eval/results.md", out_json="../eval/results.json"):
    results = []
    for q in QUESTIONS:
        r = answer(q)
        results.append(r)
        print(f"Q: {q}\nA: {r['answer']}\n")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("| # | Question | Retrieved (heading, p.) | Generated answer | Correct? | Notes |\n")
        f.write("|---|---|---|---|---|---|\n")
        for i, r in enumerate(results, 1):
            retrieved = "; ".join(f"{h['heading']} (p.{h['page']})" for h in r["retrieved"][:2])
            ans = r["answer"].replace("\n", " ").replace("|", "/")
            f.write(f"| {i} | {r['question']} | {retrieved} | {ans} | TODO | TODO |\n")

    print(f"Wrote {out_md} and {out_json} — fill in the Correct?/Notes columns by hand.")


if __name__ == "__main__":
    main()