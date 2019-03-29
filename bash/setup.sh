#!/usr/bin/env bash

# (1) build and git setup everything
echo "build & git_setup..."
(./astroboys_dumbe/git_setup.py ; cc-env ./astroboys_dumbe/build.py) &
# repeat for astroboys_proxy
# repeat for astroboys_simml
echo "...done"

# (2) start mock services
echo "start mock services..."
konsole -e "mock-ids --port 21192" --no-close
konsole -e "mock-limits --ports 1234" --no-close
echo "...done"

# (3) start proxy
echo "start proxy..."
konsole -e "./astroboys_proxy/build/debug/product/"
echo "...done"

# (4) start simml
echo "start simml..."
echo "...done"

# (5) start dumbe
echo "start dumbe"
echo "...done"
