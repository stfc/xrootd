from XRootD import client
from XRootD.client.flags import OpenFlags
from threading import Thread
from env import *

class MyThread(Thread):
  def __init__(self, file, id):
    Thread.__init__(self)
    self.file = file
    self.id = id

  def run(self):
    self.file.open(smallfile, OpenFlags.DELETE)
    assert self.file.is_open()

    s, _ = self.file.write(smallbuffer)
    assert s.ok

    print('+++ thread %d says: %s' % (self.id, self.file.read()))

    for line in self.file:
      print('+++ thread %d says: %s' % (self.id, line))

    self.file.close()

def test_threads():
  f = client.File()
  for i in range(16):
    tt = MyThread(f, i)
    tt.start()
    tt.join()
  f.close()
