import pickle

# Define messages to be sent between server and clients

def parse_message(msg):
    '''takes message with format PROTOCOL and returns a dictionary'''
    return pickle.loads(msg)

def construct_login_message(hostname):
    d = {}
    d['sender'] = hostname
    d['type'] = 'login'
    dstr = pickle.dumps(d)
    return dstr

def construct_dismiss_message(hostname):
    d = {}
    d['sender'] = hostname
    d['type'] = 'dismiss'
    dstr = pickle.dumps(d)
    return dstr

def construct_task_message(hostname, task):
    d = {}
    d['sender'] = hostname
    d['type'] = 'task'
    d['task'] = task
    dstr = pickle.dumps(d)
    return dstr
            
def construct_acc_message(hostname, task, acc, name):
    fname = "pickle/" + name + ".msg"
    fo = open(fname, 'wb')
    d = {}
    d['sender'] = hostname
    d['type'] = 'acc'
    d['task'] = task
    #d['acc'] = acc
    #print(dstr)

    d['fname'] = fname
    pickle.dump(acc, fo)
    fo.close()
    dstr = pickle.dumps(d)

    return dstr
            
def read_info(fname):
    fi = open(fname, 'rb')
    info = pickle.load(fi)  
    return info