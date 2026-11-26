"""run.py PKG MODE WORKERS: MODE old|thin|direct. Prints wall s of the book total, peak RSS MB, value."""
import importlib, resource, sys, threading, time
pkg, mode, workers = sys.argv[1], sys.argv[2], int(sys.argv[3])
m = importlib.import_module(pkg).mx_model
P = m.Projection
keys = P.book()
P.Data if False else None
m.Data.model_point_table()     # inputs read before timing, in every mode
start = time.perf_counter()
if mode == 'old':
    m.collect_workers = workers
    value = m.Book.total()
elif mode == 'thin':
    m._mx_collect_workers = workers
    value = m.Book.total()
else:       # direct: the same items read and deleted by a hand-written loop, in domain order per chunk
    parts = [None] * workers
    def chunk(w):
        total = None
        for k in keys[w::workers]:
            v = P[k].collect_outputs()['sum']['net_cf']
            if mode == 'direct':
                del P[k]
            total = v if total is None else total + v
        parts[w] = total
    threads = [threading.Thread(target=chunk, args=(w,)) for w in range(workers)]
    for t in threads: t.start()
    for t in threads: t.join()
    value = parts[0]
    for p in parts[1:]:
        value = value + p
wall = time.perf_counter() - start
print(f'RESULT {mode} {workers} {wall:.3f} {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.1f} {value!r}')
