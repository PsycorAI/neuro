# scripts/memdemo/test_memdemo.py -- 2026-10-27 (Neuro 85): end-to-end checks of a running Plasmem demo server.
# Usage: python scripts/memdemo/test_memdemo.py [port]. Resets the demo memory first and last. Prints PASS / FAIL per check.
import sys, json, urllib.request, urllib.error
BASE = f'http://127.0.0.1:{sys.argv[1] if len(sys.argv) > 1 else 8766}'
fails = []


def call(path, body=None):
    req = urllib.request.Request(BASE + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def check(name, ok, info=''):
    print(('PASS ' if ok else 'FAIL ') + name + (f'  ({info})' if info else ''))
    ok or fails.append(name)


call('/api/reset', {})
st, tour = call('/api/tour')
T = {t['id']: t for t in tour}
check('tour loads', st == 200 and len(tour) == 5)
for tid in ('recall', 'when', 'vector'):
    t = T[tid]
    _, r = call('/api/ask', {'q': t['q'], 'gold': t['gold']})
    check(f'{tid}: gold is #1', r['gold_rank'] == 1, f"rank {r['gold_rank']}, vector {r['gold_vec_rank']}")
_, r = call('/api/ask', {'q': T['when']['q']})
check('when: time window = last week', r['time_window'] == ['2026-10-19', '2026-10-25'], r['time_window'])
_, r = call('/api/ask', {'q': T['idk']['q']})
check('idk: weak confidence + unknown word', r['confidence'] < 0.15 and 'cat' in r['unknown'], f"conf {r['confidence']:.3f}")
_, a = call('/api/ask', {'q': 'Where is Lena moving?'})
check('answerable question: high confidence', a['confidence'] >= 0.3, f"conf {a['confidence']:.3f}")
t = T['learn']
_, r = call('/api/ask', {'q': t['q'], 'gold': t['gold']})
check('learn: gold starts outside top 5', r['gold_rank'] > 5, f"rank {r['gold_rank']}")
_, f = call('/api/feedback', {'q': t['q'], 'span': t['gold'], 'probes': t['probes']})
check('learn: rank -> 1 after one confirm', f['after'][0] == 1, f"{f['before']} -> {f['after']}")
check('learn: a never-asked paraphrase improves', f['after'][1] < f['before'][1], f"{f['before'][1]} -> {f['after'][1]}")
_, d = call('/api/diary')
lena = next(m['i'] for m in d['diary'] if 'moving to Lisbon' in m['text'])
_, r = call('/api/ask', {'q': 'Where is Lena moving?', 'gold': lena})
check('learning did not disturb other recall', r['gold_rank'] == 1, f"rank {r['gold_rank']}")
_, d = call('/api/diary')
n0 = d['stats']['memories']
st, r = call('/api/remember', {'text': 'We booked the cabin at Lake Tahoe for June.', 'who': 'Maya', 'date': 'not a date'})
check('remember with a bad date falls back to today', st == 200)
_, r = call('/api/ask', {'q': 'Where are we going in June?'})
check('new memory is recalled at once', r['hits'][0]['text'].startswith('We booked the cabin'), r['hits'][0]['text'][:40])
st, _ = call('/api/ask', {'q': ''})
check('empty question rejected', st == 400)
st, _ = call('/api/feedback', {'q': 'x', 'span': 10 ** 6})
check('bad memory id rejected', st == 400)
st, r = call('/api/ask', {'q': 'What did Sam fix?', 'gold': 10 ** 6})
check('bad gold ignored', st == 200 and 'gold_rank' not in r)
st, r = call('/api/ask', {'q': '<script>alert(1)</script> & what?'})
check('odd characters handled', st == 200)
_, d = call('/api/reset', {})
check('reset restores the diary', d['stats']['memories'] == n0 and d['stats']['use_synapses'] == 0)
print('ALL PASS' if not fails else f'{len(fails)} FAILED: {fails}')
sys.exit(1 if fails else 0)
