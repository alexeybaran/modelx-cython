import sys, pandas as pd
src, n, dst = sys.argv[1], int(sys.argv[2]), sys.argv[3]
df = pd.read_csv(src, dtype=str, keep_default_na=False)
t = df.iloc[[i % len(df) for i in range(n)]].copy()
t['point_id'] = [str(i + 1) for i in range(n)]
t['policy_id'] = [f'{p}-{i // len(df)}' for i, p in enumerate(t['policy_id'])]
t.to_csv(dst, index=False)
