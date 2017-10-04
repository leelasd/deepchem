import sqlite3

conn = sqlite3.connect('exps.db')

c = conn.cursor()

c.execute('''
INSERT INTO experiment (model_folder, num_epochs, kwargs_json, status)
VALUES ('full_run', 10000, '{}', 'READY')
''')
conn.commit()
conn.close()
