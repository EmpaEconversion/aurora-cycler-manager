"""Copyright © 2025, Empa, Graham Kimbell, Enea Svaluto-Ferro, Ruben Kuhnel, Corsin Battaglia.

Notification system for the Aurora cycler manager app.

To send notification in a callback, Output to notifications-container.

To send asynchronous notifications during a long callback is more complicated.
There is no built-in or third-party asynchronous notification system for Dash.
Here we use a global queue to store notifications, and an interval callback to
periodically check the queue and display new notifications.

The interval is set to 1 second while some loop is happening (e.g. waiting for
multiple samples to upload), and it is 'idle' and set to 1 minute otherwise.

"""
from time import sleep

from dash import Input, Output, html
from dash.dcc import Interval
from dash_mantine_components import NotificationProvider

notification_queue = []
idle_time = 1000*60 # 1 minute
active_time = 1000 # 1 second

def queue_notification(notification):
    """Add a notification to the queue."""
    global notification_queue
    notification_queue.append(notification)
    sleep(0.01) # HACK so the notification is added BEFORE the check notifications callback runs

notifications_layout = html.Div([
    html.Div([],id="notifications-container"),
    NotificationProvider(),
    Interval(id="notify-interval", interval=1000*60), # Check for new notifications every second
])

# When in a 'listening' state, a function will set the interval to e.g. 1 second
# Otherwise in 'idle' it will be set to 1 minute
def register_notifications_callbacks(app):
    @app.callback(
        Output("notifications-container", "children"),
        Input("notify-interval", "n_intervals"),
        Input("notify-interval", "interval"),
        prevent_initial_call=True,
    )
    def check_notifications(n_intervals, interval):
        # return notification list and clear it
        global notification_queue
        if not notification_queue:
            return []
        notifications = notification_queue
        notification_queue = []
        return notifications
