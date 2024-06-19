from time import sleep
from datetime import datetime, timedelta
import cucumber_tools as ct
from cucumber_analysis import plot_all_samples, plot_batches_from_file
import logging

logging.basicConfig(
    filename='cucumber_daemon.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

UPDATE_TIME = 60  # Wait in seconds between database updates

SNAPSHOT_TIME = ['00:00', '06:00']  # Time to snapshot the database

stop_flag = False  # Global flag to control the main loop

snapshot_times = [datetime.strptime(t, '%H:%M').time() for t in SNAPSHOT_TIME]
now = datetime.now().time()
next_run_time = next((t for t in snapshot_times if t > now), snapshot_times[0])

cucumber=ct.Cucumber()
while not stop_flag:
    sleep(UPDATE_TIME)
    dt = datetime.now()
    logging.info(f"{dt} Updating database...")
    cucumber.update_db()

    if dt.time() >= next_run_time:
        logging.info(f"{dt} Snapshotting database...")
        try:
            cucumber.snapshot_all()
            plot_all_samples()
            plot_batches_from_file()
        except Exception as e:
            logging.critical(f"Error snapshotting and graphing: {e}")
        else:
            logging.info(f"{dt} Snapshotting and graphing complete")
        
        next_run_time = next((t for t in snapshot_times if t > dt.time()), snapshot_times[0])

        # If next_run_time is earlier than the current time, add one day to it
        if next_run_time <= dt.time():
            next_run_time = (datetime.combine(dt, next_run_time) + timedelta(days=1)).time()

def stop_loop():
    global stop_flag
    stop_flag = True

if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        logging.info("Killing cucumber daemon")
        stop_loop()
