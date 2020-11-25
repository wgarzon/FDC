import zmq
import time
import sys

port = "5556"
NoSites = 1

#python3 mainControl.py port #sites
if len(sys.argv) > 1:
    port =  int(sys.argv[1])

if len(sys.argv) > 2:
    NoSites =  int(sys.argv[2])

STEPS = {"First":"Second","Third":"Fourth", "Fifth", "Sixth"}

def main():
    context = zmq.Context()
    sock = context.socket(zmq.REP)
    sock.bind("tcp://*:%s" % port)

    sites = {}
    print("MainController listening all sites:", NoSites)
    time.sleep(2)

    for ss in STEPS:
        print("*******"*20)
        print("Processing Step:",ss)
        sendStep(sock,ss)

    print("The main mainController has processed all sites:", str(NoSites))
    print("Successful process!!!")



def sendStep(sock,lblStep):
    sites = {}
    while True:
        msg = sock.recv_json()
        step, site = msg['msg'].split(",")
        sites[site]=lblStep

        print("aaa step",step,"lblStep",lblStep, "site", site)
        if step == lblStep:
            print("step",step,"lblStep",lblStep)
            print("First message received from Site:", site)
            if step == "Fifth":
                sock.send(b"STOP")
            else:
                send_next_step(sock,{"Central":STEPS[lblStep]})

            if valSites(sites,step):
                print("sites",sites,"step",step)
                break

def send_next_step(sock, step):
    try:
        sock.send_json(step)
    except StopIteration:
        sock.send_json({})

def valSites(sites, step):
    newDict = dict(filter(lambda elem: elem[1] == step, sites.items()))
    if len(newDict) == NoSites:
        print("Messages received from all Sites:", str(NoSites), ",Step:", step)
        return True
    return False

if __name__ == "__main__":
    main()
