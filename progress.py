import re

log = r"C:\Users\UnboundSB\.local\share\opencode\shell\1c4038c986c464f1f61cab02924727d1183ec643\sh_1e8257a320018MgZxp8rsgWOD3.out"
with open(log, encoding="utf-8", errors="replace") as f:
    text = f.read()

durs = [float(x) for x in re.findall(r"->\s+([0-9.]+)s after trim", text)]
n = len(durs)
total_so_far = sum(durs)
print(f"Chunks generated: {n}/52")
print(f"Audio so far: {total_so_far:.1f}s ({total_so_far/60:.1f} min)")
if n:
    avg = total_so_far / n
    est = avg * 52 + 51 * 0.35
    print(f"Avg per chunk: {avg:.1f}s")
    print(f"Estimated final: {est:.1f}s = {est/60:.1f} min")
    print(f"Remaining: {(est - total_so_far):.0f}s = {(est - total_so_far)/60:.1f} min")
last = re.findall(r"\[(\d+)/52\]", text)
print("Current chunk:", last[-1] if last else "?")