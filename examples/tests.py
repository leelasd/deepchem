import os
import subprocess
import tempfile
import delegator


def _example_run(path):
  """Checks that the example at the specified path runs corrently.

  Parameters
  ----------
  path: str
    Path to example file.
  Returns
  -------
  result: int 
    Return code. 0 for success, failure otherwise.
  """
  cmd = ["python", path]
  # Will raise a CalledProcessError if fails.
  retval = subprocess.check_output(cmd)
  return retval


def test_adme():
  print("Running test_adme()")
  output = _example_run("./adme/run_benchmarks.py")
  print(output)


def test_tox21_fcnet():
  print("Running tox21_fcnet()")
  output = _example_run("./tox21/tox21_fcnet.py")
  print(output)


if __name__ == "__main__":
  if os.path.exists('example_tests'):
    os.rmdir('example_tests')
  os.mkdir('example_tests')
  folders = list(filter(lambda x: os.path.isdir(x), os.listdir('./')))
  for folder in folders:
    print(folder)
    if folder != 'tox21':
      continue
    os.chdir(folder)
    files = list(filter(lambda x: x.endswith('.py'), os.listdir('./')))
    for f in files:
      cmd = "python %s" % f
      print(cmd)
      c = delegator.run(cmd)

      outfile = os.path.basename(f) + ".out"
      with open(outfile, 'w') as fout:
        fout.write(c.out)

      errfile = os.path.basename(f) + ".err"
      with open(errfile, 'w') as fout:
        fout.write(c.err)
    os.chdir('../')
