"""scripts/memdemo/server.py -- 2026-10-27 (Neuro 85): the Plasmem demo server (recall only, v1). CPU only, never the GPUs.
Run: CUDA_VISIBLE_DEVICES= python scripts/memdemo/server.py [--port 8766] [--name diary]
Design: docs/n85_plasmem_demo_design_2026_10_27.md. Memory file: logs/memdemo/<name>.plasmem."""
import os, sys, json, argparse, threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine

ap = argparse.ArgumentParser()
ap.add_argument('--port', type=int, default=8766)
ap.add_argument('--host', default='127.0.0.1')
ap.add_argument('--name', default='diary')
args = ap.parse_args()

# Tour: chosen examples on the demo diary (checked by temp/n85_memdemo_probe*.py), not a score. gold = a phrase of the answer.
TOUR = [
    dict(id='recall', label='Recall', q='What did Sam fix?', gold='back brake',
         note='Word neurons for "Sam" and "fix" and the meaning neurons both drive the memory where Sam fixed his bike.'),
    dict(id='when', label='When', q='What happened at the dentist last week?', gold='No cavities',
         note='"Last week" switches on time cells for Oct 19-25. They lift the visit itself above the booking made in September.'),
    dict(id='vector', label='Beats vector search', q='What will decorate the hallway?', gold='fog over the ocean',
         note='One rare word ("hallway") fires one word neuron wired to one memory. Plain vector search ranks it far down.'),
    dict(id='idk', label="I don't know", q="What is Maya's cat called?", gold=None,
         note='No word neuron exists for "cat": nothing stored ever mentioned one. Confidence stays low.'),
    dict(id='learn', label='Learns from use', q='What did they see from the top?', gold='fog over the ocean',
         probes=['What could they see from up there?', 'What did the top of the mountain look like?'],
         note='The answer (glowing in the diary) is not in the top 5. Click "this is the answer" on it, then compare.'),
]
lock = threading.Lock()
E = engine.Engine(args.name)


def gold_id(t):
    return next((d['i'] for d in E.diary() if t['gold'] and t['gold'] in d['text']), None)


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, obj, code=200, ctype='application/json'):
        body = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ('/', '/index.html'):
            return self.send(open(HERE + '/index.html', 'rb').read(), ctype='text/html; charset=utf-8')
        with lock:
            if self.path == '/api/diary':
                return self.send(dict(diary=E.diary(), stats=E.stats(), today=E.today))
            if self.path == '/api/tour':
                return self.send([dict(t, gold=gold_id(t)) for t in TOUR])
        self.send({'error': 'not found'}, 404)

    def do_POST(self):
        try:
            req = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))) or b'{}')
            with lock:
                if self.path == '/api/ask':
                    q = str(req.get('q', '')).strip()[:500]
                    if not q:
                        return self.send({'error': 'empty question'}, 400)
                    g = req.get('gold')
                    return self.send(E.ask(q, now=req.get('now') or None, who=bool(req.get('who', True)),
                                           gold=int(g) if g is not None else None))
                if self.path == '/api/remember':
                    text, who = str(req.get('text', '')).strip()[:1000], str(req.get('who', '')).strip()[:40] or 'Me'
                    if not text:
                        return self.send({'error': 'empty memory'}, 400)
                    i = E.remember(text, who, req.get('date') or E.today)
                    return self.send(dict(i=i, stats=E.stats()))
                if self.path == '/api/feedback':
                    r = E.feedback(str(req['q'])[:500], int(req['span']), now=req.get('now') or None,
                                   who=bool(req.get('who', True)), probes=[str(x)[:500] for x in req.get('probes', [])[:5]])
                    return self.send(dict(r, stats=E.stats()))
                if self.path == '/api/reset':
                    E.reset()
                    return self.send(dict(stats=E.stats()))
            self.send({'error': 'not found'}, 404)
        except Exception as ex:
            self.send({'error': f'{type(ex).__name__}: {ex}'}, 500)


if __name__ == '__main__':
    print(f'Plasmem demo on http://{args.host}:{args.port}  (memory file {E.file})', flush=True)
    ThreadingHTTPServer((args.host, args.port), H).serve_forever()
