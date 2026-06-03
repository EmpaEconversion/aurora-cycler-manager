# Single-user, local

The simplest set up is a single user installing the app locally and keeping all data locally. This is the default configuration if creating a project locally and running the app.

# One `aurora-cycler-manager`, multiple users

You can also serve the app from a server machine which users then connect to. Users will only have access to the features through the GUI, and cannot directly access data or write scripts to control cyclers. You can set the host and port, and skip opening in a browser with e.g. `aurora-app --host="0.0.0.0" --port=8050 --no-browser"`. You must set up your firewalls to allow user to connect.

# Multiple `aurora-cycler-manager`s

Multiple users can install `aurora-cycler-manager` and access the same project if it is in a network accessible location. Then multiple users are able to run complex scripts. It is possible to store an sqlite database on a network drive, but it can become slow if it the database gets large. It is better to set up postgresql.
