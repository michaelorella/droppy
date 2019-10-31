#!/bin/bash
# Simple bash script that runs contact angle analysis on all files in a folder

# Check that a target directory was specified
if [ "$1" != "" ]; then
	target="$1"
else
	echo "No target folder provided! Make sure to specify the pathname"
	exit 2
fi

# Check if any different parameters for the python script were specified
if [ "${*:2}" != "" ]; then
	params="${@:2}"
else
	params=""
fi


# Loop through each file for which results have not yet been calculated
for file in "$target"/*.avi; do
	f=$(basename "$file")
	d=$(dirname "$file")

	if [ ! -f "$d/results_$f".csv ]; then
		echo "Running on $file"
		python analysis.py "$file" $params
	fi
done