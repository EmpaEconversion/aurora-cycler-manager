# Set up SSH

Interacting with cyclers remotely (submitting jobs, pulling data, etc.) works with **OpenSSH**. Servers must have OpenSSH server installed and running, and users must have OpenSSH client (usually installed by default) and password-less access to the cycler PCs.

## Install OpenSSH on cycler PC

On Windows, go to **Settings** -> **System** -> **Optional Features**. Check that you have **OpenSSH Server** installed, if not click **Add an optional feature**. It can take a very long time (up to an hour) to add OpenSSH, be patient.

In **Services**, make sure OpenSSH server is running and set to automatic start up.


## Put your public key on cycler PC

On your PC running aurora-cycler-manager, run `ssh-keygen`, and copy your public key to `~/.ssh/authorized_keys` on the cycler PC.

The cycler PC must also be in your `known_hosts`, the easiest way is to just connect using `ssh user@host` and say yes to adding to known hosts. If your public key is on the cycler PC, you should not be asked for a password. If you are asked for a password, it is probably a Windows permission issue on the cycler PC.
