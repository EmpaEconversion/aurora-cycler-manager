"""Copyright © 2025, Empa.

Notification system and loading messages for the Aurora cycler manager app.

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

Use with a 'running' keyword in callback wrapper:
    @app.callback(
        ...
        running=[
            (Output("loading-message-store", "data"), "Some message...", ""),
            (Output("notify-interval", "interval"), active_time, idle_time),
        ],
    )
This shows a loading message and sets notifications to listening while the
callback is running.

"""

from dash import Dash, Input, Output, html
from dash.dcc import Interval
from dash_mantine_components import Notification, NotificationProvider

notification_queue = []
idle_time = 1000 * 60  # Time to check for notifications when 'idle'
active_time = 500  # Time to check for notifications when 'active'
trigger_time = 600  # Delay to check one final time after switching to 'idle'


def queue_notification(notification: Notification) -> None:
    """Add a notification to the queue."""
    global notification_queue
    notification_queue.append(notification)


def success_notification(title: str, message: str, queue: bool = False) -> Notification:
    """Create a success notification."""
    notification = Notification(
        title=title,
        message=message,
        color="green",
        action="show",
        icon=html.I(className="bi bi-check-circle"),
    )
    if queue:
        queue_notification(notification)
    return notification


def info_notification(title: str, message: str, queue: bool = False) -> Notification:
    """Create an info notification."""
    notification = Notification(
        title=title,
        message=message,
        color="blue",
        action="show",
        icon=html.I(className="bi bi-info-circle"),
    )
    if queue:
        queue_notification(notification)
    return notification


def warning_notification(title: str, message: str, queue: bool = False) -> Notification:
    """Create a warning notification."""
    notification = Notification(
        title=title,
        message=message,
        color="yellow",
        action="show",
        icon=html.I(className="bi bi-exclamation-triangle"),
    )
    if queue:
        queue_notification(notification)
    return notification


def error_notification(title: str, message: str, queue: bool = False) -> Notification:
    """Create an error notification."""
    notification = Notification(
        title=title,
        message=message,
        color="red",
        action="show",
        icon=html.I(className="bi bi-x-circle"),
    )
    if queue:
        queue_notification(notification)
    return notification


notifications_layout = html.Div(
    [
        html.Div([], id="notifications-container"),
        NotificationProvider(),
        Interval(id="notify-interval", interval=idle_time),
        Interval(id="trigger-interval", interval=trigger_time, n_intervals=0, disabled=True),
    ],
)


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
        Output("notifications-container", "children", allow_duplicate=True),
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


# Loading spinner
custom_spinner = html.Div(
    style={
        "position": "absolute",
        "top": "50%",
        "left": "50%",
        "transform": "translate(-50%, -50%)",
        "width": "100px",
        "height": "100px",
    },
    children=[
        html.Img(
            src="/assets/spinner-spin.svg",
            className="spinner-spin",
            style={"width": "100px", "height": "100px"},
        ),
        html.Img(
            src="/assets/spinner-stationary.svg",
            style={
                "position": "absolute",
                "top": "0",
                "left": "0",
                "width": "100px",
                "height": "100px",
                "color": "white",
            },
        ),
    ],
)

# Loading messages
loading_message = html.Div(
    "",
    id="loading-message",
    style={
        "position": "absolute",
        "left": "50%",
        "top": "50%",
        "transform": "translate(-50%, 50px)",
        "fontSize": "20px",
        "color": "#000000",
        "textAlign": "center",
        "textGlow": "0 0 20px blue",
        "opacity": 1,
        "transition": "opacity 0.5s ease-in-out",
    },
)
