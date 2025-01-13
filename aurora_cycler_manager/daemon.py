"""Copyright © 2025, Empa, Graham Kimbell, Enea Svaluto-Ferro, Ruben Kuhnel, Corsin Battaglia.

Daemon to update database, snapshot jobs and plots graphs

Updates database regularly and snapshots all jobs then analyses and plots graphs
at specified times each day. Change the update time and snapshot times in the
main block to suit your needs.
"""
import logging
import sys
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from time import sleep

root_dir = Path(__file__).resolve().parents[1]
if root_dir not in sys.path:
    sys.path.append(str(root_dir))
import aurora_cycler_manager.server_manager as server_manager
from aurora_cycler_manager.analysis import analyse_all_batches, analyse_all_samples
from aurora_cycler_manager.eclab_harvester import convert_mpr, get_all_mprs
from aurora_cycler_manager.neware_harvester import convert_neware_data, harvest_all_neware_files

STOP_FLAG = False

def daemon_loop(update_time: float | None = None, snapshot_times: list | None = None):
    """Run main loop for updating, snapshotting and plotting.

    Args:
        update_time: Time in seconds between database updates, default 300
        snapshot_times: List of times to snapshot the database each day
            specified in 24-hour format as a string, e.g. ['00:00', '12:00']
            default ['02:00']

    """
    logging.basicConfig(
        filename="daemon.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    # Add a stream handler to also log to the console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
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
        snapshot_times = ["02:00"]
        logging.warning("No snapshot times specified, defaulting to 2am")
    else:
        logging.info("Snapshotting and plotting at %s each day", snapshot_times)

    now = datetime.now()
    snapshot_datetimes = [datetime.combine(now, datetime.strptime(t, "%H:%M").time()) for t in snapshot_times]
    snapshot_datetimes = [t if t > now else t + timedelta(days=1) for t in snapshot_datetimes]
    next_run_time = min(snapshot_datetimes)  # Find the earliest next run time
    logging.info("Next snapshot at %s", next_run_time)

    sm=server_manager.ServerManager()
    try:
        sm.update_db()
    except Exception as e:
        logging.critical("Error updating database: %s", e)
        logging.debug(traceback.format_exc())
    logging.info("Server manager initialised, entering main loop...")
    while not STOP_FLAG:
        sleep(update_time)
        now = datetime.now()
        logging.info("Updating database...")
        try:
            sm.update_db()
        except Exception as e:
            logging.critical("Error updating database: %s", e)
            logging.debug(traceback.format_exc())
        if now >= next_run_time:
            logging.info("Snapshotting database...")
            try:
                sm.snapshot_all()
            except Exception as e:
                logging.critical("Error snapshotting: %s", e)
                logging.debug(traceback.format_exc())
            else:
                logging.info("Snapshotting complete")
            try:
                new_files = get_all_mprs()
                for mpr_path in new_files:
                    convert_mpr(mpr_path)
            except Exception as e:
                logging.critical("Error converting mprs: %s", e)
                logging.debug(traceback.format_exc())
            else:
                logging.info("mprs downloaded and converted")
            try:
                new_files = harvest_all_neware_files()
                for file in new_files:
                    convert_neware_data(file)
            except Exception as e:
                logging.critical("Error converting neware files: %s", e)
                logging.debug(traceback.format_exc())
            else:
                logging.info("neware files downloaded and converted")
            try:
                analyse_all_samples()
                analyse_all_batches()
            except Exception as e:
                logging.critical("Error analysing and plotting: %s", e)
                logging.debug(traceback.format_exc())
            else:
                logging.info("Plotting complete")

            # Calculate the next run time for the snapshot
            now = datetime.now()
            snapshot_datetimes = [datetime.combine(now, datetime.strptime(t, "%H:%M").time()) for t in snapshot_times]
            snapshot_datetimes = [t if t > now else t + timedelta(days=1) for t in snapshot_datetimes]
            next_run_time = min(snapshot_datetimes)
            logging.info("Next snapshot at %s", next_run_time)

if __name__ == "__main__":
    try:
        daemon_loop(update_time = 300, snapshot_times = ["02:00"])
    except KeyboardInterrupt:
        logging.info("Killing daemon")
        STOP_FLAG = True
