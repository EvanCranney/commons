import argparse

# boilerplate setup
parser = argparse.ArgumentParser()
parser.add_argument("arg01") # store first arg in parser.parse_args().echo
parser.add_argument("arg02", help="second argument") # help info
parser.add_argument("parameter01", type=int) # specify type


# to retrieve the named args
args = parser.parse_args()


print(args)

