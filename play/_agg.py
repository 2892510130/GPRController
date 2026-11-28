import csv, statistics as st, collections
for tag in ('delay3', 'bad_wheel925'):
    rows = list(csv.DictReader(open(f'log/sweep/{tag}_ablation.csv')))
    print(f'=== {tag} ===')
    g = collections.defaultdict(list)
    for r in rows:
        g[r['variant']].append(r)
    for v, rs in g.items():
        ml = [float(r['mean_lat']) * 1000 for r in rs]
        xl = [float(r['max_lat']) * 1000 for r in rs]
        mh = [float(r['mean_head']) for r in rs]
        ms = [float(r['ms_per_step']) for r in rs]
        steps = [int(r['steps']) for r in rs]
        mf = rs[0]['m_final']
        print('{:16s} mean|d|={:6.2f}+-{:.2f} mm  max|d|={:6.2f} mm  mean|th|={:.4f}  {:5.1f} ms/step  steps={}  M={}'.format(
            v, st.mean(ml), st.pstdev(ml), st.mean(xl), st.mean(mh), st.mean(ms), steps, mf))