import sqlite3
import json

conn = sqlite3.connect('exps.db')

c = conn.cursor()

# c.execute('''
# INSERT INTO experiment (model_folder, num_epochs, kwargs_json, status)
# VALUES ('full_run', 10000, '{}', 'READY')
# ''')

# d = {
#   'activation': 'gaussian'
# }
# kwargs_json = json.dumps(d)
# c.execute('''
# INSERT INTO experiment (model_folder, num_epochs, kwargs_json, status)
# VALUES ('gaussian_full_run', 10000, ?, 'READY')
# ''', (kwargs_json,))
#
# d = {
#   'activation': 'gaussian',
#   'data_length': 6,
# }
# kwargs_json = json.dumps(d)
# c.execute('''
# INSERT INTO experiment (model_folder, num_epochs, kwargs_json, status)
# VALUES ('gaussian_small_run', 10000, ?, 'READY')
# ''', (kwargs_json,))
#
# d = {
#   'data_length': 6,
# }
# kwargs_json = json.dumps(d)
# c.execute('''
# INSERT INTO experiment (model_folder, num_epochs, kwargs_json, status)
# VALUES ('gaussian_small_run', 10000, ?, 'READY')
# ''', (kwargs_json,))


# d = {
#   'activation': 'relu',
#   'data_length': 4,
#   'batch_norms': [False, True, True]
# }
# kwargs_json = json.dumps(d)
# c.execute('''
# INSERT INTO experiment (model_folder, num_epochs, kwargs_json, status)
# VALUES ('batch_normftt_relu_gdb4', 10000, ?, 'READY')
# ''', (kwargs_json,))
#
# d = {
#   'activation': 'elu',
#   'data_length': 4,
#   'batch_norms': [False, True, True]
# }
# kwargs_json = json.dumps(d)
# c.execute('''
# INSERT INTO experiment (model_folder, num_epochs, kwargs_json, status)
# VALUES ('batch_normftt_elu_4', 10000, ?, 'READY')
# ''', (kwargs_json,))
#
# d = {
#   'activation': 'relu',
#   'data_length': 4,
#   'batch_norms': [True, True, True]
# }
# kwargs_json = json.dumps(d)
# c.execute('''
# INSERT INTO experiment (model_folder, num_epochs, kwargs_json, status)
# VALUES ('batch_normttt_relu_gdb4', 10000, ?, 'READY')
# ''', (kwargs_json,))
#
# d = {
#   'activation': 'relu',
#   'data_length': 4,
#   'batch_norms': [False, False, True]
# }
# kwargs_json = json.dumps(d)
# c.execute('''
# INSERT INTO experiment (model_folder, num_epochs, kwargs_json, status)
# VALUES ('batch_normfft_relu_gdb4', 10000, ?, 'READY')
# ''', (kwargs_json,))

# d = {
#   'activation': 'relu',
#   'data_length': 4,
#   'dropouts': [0.9, 0.5, 0.5, 0.5]
# }
# kwargs_json = json.dumps(d)
# c.execute('''
# INSERT INTO experiment (model_folder, num_epochs, kwargs_json, status)
# VALUES ('dropout_9,5,5,5_relu_gdb4', 10000, ?, 'READY')
# ''', (kwargs_json,))
#
# d = {
#   'activation': 'relu',
#   'data_length': 4,
#   'dropouts': [0.0, 0.5, 0.5, 0.5]
# }
# kwargs_json = json.dumps(d)
# c.execute('''
# INSERT INTO experiment (model_folder, num_epochs, kwargs_json, status)
# VALUES ('dropout_0,5,5,5_relu_gdb4', 10000, ?, 'READY')
# ''', (kwargs_json,))
#
# d = {
#   'activation': 'relu',
#   'data_length': 4,
#   'dropouts': [0.9, 0.5, 0.5, 0.5],
#   'layer_structures': [160, 160, 80]
# }
# kwargs_json = json.dumps(d)
# c.execute('''
# INSERT INTO experiment (model_folder, num_epochs, kwargs_json, status)
# VALUES ('dropout_1.25layer_9,5,5,5_relu_gdb4', 10000, ?, 'READY')
# ''', (kwargs_json,))
#
# d = {
#   'activation': 'relu',
#   'data_length': 4,
#   'dropouts': [0.9, 0.5, 0.5, 0.5],
#   'layer_structures': [192, 192, 128]
# }
# kwargs_json = json.dumps(d)
# c.execute('''
# INSERT INTO experiment (model_folder, num_epochs, kwargs_json, status)
# VALUES ('dropout_1.5layer_9,5,5,5_relu_gdb4', 10000, ?, 'READY')
# ''', (kwargs_json,))
#
# d = {
#   'activation': 'gaussian',
#   'data_length': 4,
#   'dropouts': [0.0, 0.5, 0.5, 0.5]
# }
# kwargs_json = json.dumps(d)
# c.execute('''
# INSERT INTO experiment (model_folder, num_epochs, kwargs_json, status)
# VALUES ('dropout_0,5,5,5_gaussian_gdb4', 10000, ?, 'READY')
# ''', (kwargs_json,))

# d = {
#   'activation': 'gaussian',
#   'data_length': 4,
#   'learning_rate': 0.01
# }
# kwargs_json = json.dumps(d)
# c.execute('''
# INSERT INTO experiment (model_folder, num_epochs, kwargs_json, status)
# VALUES ('learning_rate_1', 10000, ?, 'READY')
# ''', (kwargs_json,))
#
# d = {
#   'activation': 'gaussian',
#   'data_length': 4,
#   'learning_rate': 0.0005
# }
# kwargs_json = json.dumps(d)
# c.execute('''
# INSERT INTO experiment (model_folder, num_epochs, kwargs_json, status)
# VALUES ('learning_rate_2', 10000, ?, 'READY')
# ''', (kwargs_json,))
#
# d = {
#   'activation': 'gaussian',
#   'data_length': 4,
#   'learning_rate': 0.0001
# }
# kwargs_json = json.dumps(d)
# c.execute('''
# INSERT INTO experiment (model_folder, num_epochs, kwargs_json, status)
# VALUES ('learning_rate_3', 10000, ?, 'READY')
# ''', (kwargs_json,))

# d = {
#   'activation': 'gaussian',
#   'data_length': 7
# }
# kwargs_json = json.dumps(d)
# c.execute('''
# INSERT INTO experiment (model_folder, num_epochs, kwargs_json, status)
# VALUES ('gaussian_gdb7', 10000, ?, 'READY')
# ''', (kwargs_json,))
#
# d = {
#   'activation': 'gaussian',
#   'data_length': 8
# }
# kwargs_json = json.dumps(d)
# c.execute('''
# INSERT INTO experiment (model_folder, num_epochs, kwargs_json, status)
# VALUES ('gaussian_gdb8', 10000, ?, 'READY')
# ''', (kwargs_json,))

d = {
  'activation': 'gaussian',
  'data_length': 4,
  'loss_fn': 't_exp'
}
kwargs_json = json.dumps(d)
c.execute('''
INSERT INTO experiment (model_folder, num_epochs, kwargs_json, status)
VALUES ('gaussian_explossfn_gdb8', 10000, ?, 'READY')
''', (kwargs_json,))
#
# d = {
#   'activation': 'gaussian',
#   'data_length': 8,
#   'learning_rate': 0.00003
# }
# kwargs_json = json.dumps(d)
# c.execute('''
# INSERT INTO experiment (model_folder, num_epochs, kwargs_json, status)
# VALUES ('gaussian_lr3_gdb8', 10000, ?, 'READY')
# ''', (kwargs_json,))

conn.commit()
conn.close()
