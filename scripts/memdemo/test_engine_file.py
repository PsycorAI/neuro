# scripts/memdemo/test_engine_file.py -- 2026-10-28 (Neuro 85): the append-only memory file. Checks that a reloaded memory
# recalls exactly like the live one (after writes + learning), that a write torn by a crash is dropped and later writes
# still load, that an older npz memory file converts with identical recall, and that a write costs about the same at
# any memory size. Uses throwaway files logs/memdemo/test_*.plasmem. CPU only.
# Usage: CUDA_VISIBLE_DEVICES= python scripts/memdemo/test_engine_file.py
import os, sys, glob, json, time, shutil
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine
fails = []
QS = ['What did Sam fix?', 'What happened at the dentist last week?', 'Where are we going in June?',
      'What will decorate the hallway?', "What is Maya's cat called?"]


def check(name, ok, info=''):
    print(('PASS ' if ok else 'FAIL ') + name + (f'  ({info})' if info else ''))
    ok or fails.append(name)


def acts(e):
    return [e.p.activate(q, e.qemb(q), now=e.today, g_who=engine.G_WHO, **engine.ACT) for q in QS]


def same(a, b):
    return all(x.shape == y.shape and np.array_equal(x, y) for x, y in zip(a, b))


for f in glob.glob(engine.R + 'logs/memdemo/test_*.plasmem*'):
    os.remove(f)
emb = engine.Embedder()
e1 = engine.Engine('test_file', emb=emb)
e1.remember('We booked the cabin at Lake Tahoe for June.', 'Maya', '2026-10-27')
e1.remember('The hallway print is the fog photo.', 'Sam', '2026-10-27')
e1.feedback('What did they see from the top?', next(d['i'] for d in e1.diary() if 'fog over the ocean' in d['text']))
check('reload recalls exactly like the live memory', same(acts(e1), acts(engine.Engine('test_file', emb=emb))))

with open(e1.file, 'a') as f:  # a crash in the middle of a write
    f.write('{"r": ["Maya: half a wri')
e2 = engine.Engine('test_file', emb=emb)
check('torn last write dropped on load', len(e2.p.diary) == len(e1.p.diary))
e2.remember('Lena sent the Lisbon photos.', 'Maya', '2026-10-27')
e3 = engine.Engine('test_file', emb=emb)
check('writes after a torn write still load', len(e3.p.diary) == len(e1.p.diary) + 1 and same(acts(e2), acts(e3)))

old = engine.R + 'logs/memdemo/diary.plasmem.npz-backup'  # an older-format file (kept by the server restart, 10-28)
if os.path.exists(old):
    shutil.copy(old, engine.R + 'logs/memdemo/test_npz.plasmem')
    m1 = engine.Engine('test_npz', emb=emb)  # loads the npz, converts the file
    m2 = engine.Engine('test_npz', emb=emb)  # loads the converted file
    with open(m2.file, 'rb') as f:
        check('npz converted to the append-only format', f.read(1) == b'{')
    check('converted file recalls exactly like the npz', same(acts(m1), acts(m2)))
else:
    print('SKIP npz conversion (no backup file)')

ts = []
for n in range(40):  # write cost should not grow with the memory (it grew linearly before 10-28)
    t0 = time.perf_counter(); e3.remember(f'Note {n}: we tried recipe number {n} and it was fine.', 'Sam', '2026-10-27')
    ts.append(time.perf_counter() - t0)
check('write latency (front end + synapses + file) under 200 ms', np.median(ts) < .2, f'median {1e3 * np.median(ts):.0f} ms')
for f in glob.glob(engine.R + 'logs/memdemo/test_*.plasmem*'):
    os.remove(f)
print('ALL PASS' if not fails else f'{len(fails)} FAILED: {fails}')
sys.exit(1 if fails else 0)
