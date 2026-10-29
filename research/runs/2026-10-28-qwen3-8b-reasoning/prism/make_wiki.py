import sys, pyarrow.parquet as pq
src, out = sys.argv[1], sys.argv[2]
rows = pq.read_table(src).column("text").to_pylist()
print(len(rows), repr(rows[:4]))
# llama.cpp's wiki.test.raw: every line of the test split, blank lines as " \n" (HF stores them as "")
open(out + "/wiki.test.raw", "w", encoding="utf-8", newline="").write("".join(r if r else " \n" for r in rows))
open(out + "/wiki.test.join.raw", "w", encoding="utf-8", newline="").write("".join(rows))
