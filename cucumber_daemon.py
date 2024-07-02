""" Daemon to update database, snapshot jobs and plots graphs

Updates database regularly and snapshots all jobs then analyses and plots graphs
at specified times each day. Change the update time and snapshot times in the
main block to suit your needs.
"""
from sys import stdout
from time import sleep
from datetime import datetime, timedelta
import logging
import traceback
import cucumber_tools as ct
from cucumber_analysis import plot_all_samples, plot_batches_from_file

STOP_FLAG = False

def daemon_loop(update_time: float = None, snapshot_times: list = None):
    """ Main loop for updating, snapshotting and plotting.
    
    Args:
        update_time: Time in seconds between database updates, default 300
        snapshot_times: List of times to snapshot the database each day
            specified in 24-hour format as a string, e.g. ['00:00', '12:00']
            default ['02:00']
    """
    logging.basicConfig(
        filename='cucumber_daemon.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    # Add a stream handler to also log to the console
    console_handler = logging.StreamHandler(stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)
    logging.getLogger("paramiko").setLevel(logging.WARNING)
    logging.getLogger("scp").setLevel(logging.WARNING)

    if not update_time:
        update_time = 300
        logging.warning("No update time specified, defaulting to 5 minutes")
    else:
        logging.info("Sleeping for %s seconds between database updates", update_time)
    if not snapshot_times:
        snapshot_times = ['02:00']
        logging.warning("No snapshot times specified, defaulting to 2am")
    else:
        logging.info("Snapshotting and plotting at %s each day", snapshot_times)

    now = datetime.now()
    snapshot_datetimes = [datetime.combine(now, datetime.strptime(t, '%H:%M').time()) for t in snapshot_times]
    snapshot_datetimes = [t if t > now else t + timedelta(days=1) for t in snapshot_datetimes]
    next_run_time = min(snapshot_datetimes)  # Find the earliest next run time
    logging.info("Next snapshot at %s", next_run_time)

    cucumber=ct.Cucumber()
    logging.info("Cucumber complete, entering main loop...")
    while not STOP_FLAG:
        sleep(update_time)
        now = datetime.now()
        logging.info("Updating database...")
        try:
            cucumber.update_db()
        except Exception as e:
            logging.critical("Error updating database: %s", e)
            logging.debug(traceback.format_exc())
        if now >= next_run_time:
            logging.info("Snapshotting database...")
            try:
                cucumber.snapshot_all()
            except Exception as e:
                logging.critical("Error snapshotting: %s", e)
                logging.debug(traceback.format_exc())
            else:
                logging.info("Snapshotting complete")
            try:
                plot_all_samples()
                plot_batches_from_file()
            except Exception as e:
                logging.critical("Error plotting graphs: %s", e)
                logging.debug(traceback.format_exc())
            else:
                logging.info("Plotting complete")

            # Calculate the next run time for the snapshot
            next_run_time = min(t + timedelta(days=1) for t in snapshot_datetimes if t + timedelta(days=1) > now)
            logging.info("Next snapshot at %s", next_run_time)

if __name__ == "__main__":
    try:
        daemon_loop(update_time = 300, snapshot_times = ['02:00'])
    except KeyboardInterrupt:
        logging.info("Killing cucumber daemon")
        STOP_FLAG = True
