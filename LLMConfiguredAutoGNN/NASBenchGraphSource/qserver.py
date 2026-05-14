# -*- coding utf-8 -*-

# python qserver.py

from twisted.internet.protocol import Factory, connectionDone
from twisted.internet import reactor, protocol
import qprotocol
import pickle
from hpo import all_archs

class QServer(protocol.Protocol):
    def __init__(self):
        super().__init__()

    def connectionMade(self):
        self.factory.numConnections += 1
        print("Connection established! active: %d" % self.factory.numConnections)

    def dataReceived(self, data):
        msg = qprotocol.parse_message(data)
        if msg['type'] == 'login':
            print("Number of active connections: %d" % self.factory.numConnections)
        elif msg['type'] == 'acc':
            # if put sorting here, all client results are split
            #self.factory.arrange_res(msg['task'], msg['acc'])
            infos = qprotocol.read_info(msg['fname'])
            self.factory.arrange_res(msg['task'], infos)
        mes = self.factory.get_task(msg['sender'])
        self.transport.write(mes)

    def connectionLost(self, reason=connectionDone):
        self.factory.numConnections -= 1
        print("Connection lost! active: %d" % self.factory.numConnections)

class QSFactory(protocol.ServerFactory):
    numConnections = 0

    def __init__(self):
        self.protocol = QServer
        self.archs = all_archs()
        self.architer = iter(self.archs)
        self.res = {}
        self.tick = 1
        self.foname = 'bench.out'

        #self.get_rest("cs0.bench")

    def get_rest(self, fname):
        f = open(fname, "rb")
        self.res = pickle.load(f)
        f.close()

        self.narchs = []
        for arch in self.archs:
            hash = arch.hash_arch()
            info = self.res.get(hash, None)
            if info == None:
                self.narchs.append(arch)

        print(len(self.narchs))
        self.architer = iter(self.narchs)
    
    def get_task(self, sender):
        try:
            task = next(self.architer)
            print(str(task))
            print(str(self.tick))
            print(str(sender))
            mes = qprotocol.construct_task_message(sender, task)
        except StopIteration:
            self.save()
            mes = qprotocol.construct_dismiss_message(sender)
        return mes
    
    def save(self):
        f = open('arxiv0.bench', 'wb')
        pickle.dump(self.res, f)
        f.close()

    def arrange_res(self, task, acc):
        hash = task.hash_arch()
        self.res[hash] = acc

        self.tick += 1
        if self.tick % 100 == 0:
            self.save()

def main():
    reactor.listenTCP(59486, QSFactory())
    print("begin listen")
    reactor.run()

if __name__ == '__main__':
    main()
