import zmq
import sys
import time

port = "5556"
site = 1
ip = "localhost"

#python3 siteControl.py port siteid ip
if len(sys.argv) > 1:
    port =  int(sys.argv[1])

if len(sys.argv) > 2:
    site =  int(sys.argv[2])

if len(sys.argv) > 3:
    ip =  (sys.argv[3])


def main():
    context = zmq.Context()
    sock = context.socket(zmq.REQ)
    sock.connect("tcp://%s:%s" % (ip,port))
    print("Site " + str(site) + " is waiting message from the MainController")

    while True:
        sock.send_json({ "msg": "First,"+str(site)})
        work = sock.recv_json()
        if work != {}: print("First message sent!!!")
        if work == {}: continue
        step = work['Central']
        print("Message received from Central Site, Step:%s" % (step))
        print("Doing local Processing")
        time.sleep(10)

        sock.send_json({ "msg": "Third,"+str(site)})
        work = sock.recv_json()
        if work != {}: print("Second message sent!!!")
        if work == {}: continue
        step = work['Central']
        print("Message received from Central Site, Step:%s" % (step))
        print("Evaluating collaborative model")
        time.sleep(10)

        sock.send_json({ "msg": "Fifth,"+str(site)})
        print("Fifth message sent!!!")
        rta = sock.recv()
        if rta == b"STOP": break

    print("The site", str(site), "has finished the process!!!")


if __name__ == "__main__":
    main()
