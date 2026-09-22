import json

path = r'c:\Users\RAM\ZEPTO\data_pipeline\.ipynb_checkpoints\data_pipeline-checkpoint.ipynb'
nb   = json.load(open(path, encoding='utf-8'))
bugs = []

for i, c in enumerate(nb['cells']):
    src = ''.join(c['source'])
    if 'END AS check' in src:
        bugs.append(f'Cell {i+1}: END AS check  (SQLite reserved word — will crash)')
    if "{'books', 'categories'}" in src and 'NOT LIKE' not in src:
        bugs.append(f'Cell {i+1}: sqlite_master assertion without sqlite_% filter')

print(f'Total cells : {len(nb["cells"])}')
print(f'Bugs found  : {len(bugs)}')
for b in bugs:
    print(f'  FAIL: {b}')
if not bugs:
    print('  PASS: checkpoint is clean — no reserved-word or assertion bugs')
