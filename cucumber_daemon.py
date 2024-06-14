from time import sleep
from datetime import datetime
import cucumber_tools as ct

UPDATE_TIME = 60  # Wait between database updates
SNAPSHOT_TIME = 12*60*60 # Wait between snapshots

stop_flag = False  # Global flag to control the main loop

def main_loop():
    global stop_flag
    print("Starting cucumber daemon")
    cucumber = ct.Cucumber()
    i=0
    while not stop_flag:
        sleep(UPDATE_TIME)
        dt = datetime.now()
        print(f"{dt} Updating database...")
        cucumber.update_db()
        i += 1
        if i*UPDATE_TIME >= SNAPSHOT_TIME:
            print(f"{dt} - Snapshotting...")
            cucumber.snapshot_all()
            i = 0

def stop_loop():
    global stop_flag
    stop_flag = True

if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        print("Killing cucumber daemon")
        stop_loop()