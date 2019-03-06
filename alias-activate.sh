# Be sure to run this by sourcing. Trying to run it from the command line creates a new shell
# that terminates immediately.
# E.g. if the ALIAS_NAME is act then run as follows: 
# $ source alias-activate.sh ; act

# This is a little utility script to make it easier to activate the virtualenv
# Change the env vars to match your own directory structure and preference for alias name.

VIRTUALENVS=~/.virtualenvs
VENV_NAME=neural-networks-and-deep-learning
ALIAS_NAME=nn
alias $ALIAS_NAME="source $VIRTUALENVS/$VENV_NAME/bin/activate"

echo "You now have an alias that can be used to access this virtual env from a prompt like so:"
echo "$ $ALIAS_NAME"
# My own bash failure--can't figure out why the next line doesn't invoke the alias.

# $ALIAS_NAME
