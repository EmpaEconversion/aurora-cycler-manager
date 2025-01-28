"""Copyright © 2025, Empa, Graham Kimbell, Enea Svaluto-Ferro, Ruben Kuhnel, Corsin Battaglia.

Notification system for the Aurora cycler manager app.

To send notification in a callback, Output to notifications-container.

To send asynchronous notifications during a long callback is more complicated.
There is no built-in or third-party asynchronous notification system for Dash.
Here we use a global queue to store notifications, and an interval callback to
periodically check the queue and display new notifications.

The interval is set to 500 ms while some loop is happening (e.g. waiting for
multiple samples to upload), and it is 'idle' and set to 1 minute otherwise.

An additional interval is used to check one last time after switching to 'idle'.
This is because there is a race condition where the interval may switch to
'idle' before the final notification is displayed.

"""
from dash import Dash, Input, Output, html
from dash.dcc import Interval
from dash_mantine_components import Notification, NotificationProvider

notification_queue = []
idle_time = 1000*60  # Time to check for notifications when 'idle'
active_time = 500  # Time to check for notifications when 'active'
trigger_time = 600 # Delay to check one final time after switching to 'idle'

def queue_notification(notification: Notification) -> None:
    """Add a notification to the queue."""
    global notification_queue
    notification_queue.append(notification)

notifications_layout = html.Div([
    html.Div([],id="notifications-container"),
    NotificationProvider(),
    Interval(id="notify-interval", interval=idle_time),
    Interval(id="trigger-interval", interval=trigger_time, n_intervals=0, disabled=True),
])

# When in a 'listening' state, a function will set the interval to e.g. 1 second
# Otherwise in 'idle' it will be set to 1 minute
def register_notifications_callbacks(app: Dash) -> None:
    # When the notify-interval time changes, change trigger-interval one second later
    @app.callback(
        Output("trigger-interval", "disabled", allow_duplicate=True),
        Input("notify-interval", "interval"),
        prevent_initial_call=True,
    )
    def update_interval_changed(interval: int) -> bool:
        return False

    # Check for notifications whenever notify or trigger interval changes
    @app.callback(
        Output("notifications-container", "children"),
        Output("trigger-interval", "disabled", allow_duplicate=True),
        Input("notify-interval", "n_intervals"),
        Input("trigger-interval", "n_intervals"),
        prevent_initial_call=True,
    )
    def check_notifications(n_notify: int, n_trigger: int) -> tuple[list[Notification], bool]:
        # return notification list and clear it
        global notification_queue
        if not notification_queue:
            return [], bool(n_trigger)
        notifications = notification_queue
        notification_queue = []
        return notifications, bool(n_trigger)

