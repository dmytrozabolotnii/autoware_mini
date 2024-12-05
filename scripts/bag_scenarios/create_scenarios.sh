#!/bin/bash

# Ensure script requires two arguments
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <path_to_csv_file>"
    echo " "
    echo "Example: $0 ~/autoware_mini_ws/src/autoware_mini/data/bag_scenarios/tartu_large/pedestrian_crossing.csv"
    echo "This script will rerecord the bag with new detection (cluster and sfa) and then convert it to a scenario."
    echo "  - bags must be present in the ...autoware_mini/data/bags/ folder"
    echo "  - final scenarios will be saved in the same folder as the csv file"
    echo " "
    exit 1
fi

CSV_FILE=$1
# Derive parameters from the CSV filename (remove the path and extension)
SCENARIO_TYPE=$(basename "$CSV_FILE" .csv)
SCENARIO_DIR=$(dirname "$CSV_FILE")
SCENARIO_MAP=$(basename $SCENARIO_DIR)

# Location of the autoware_mini package
AUTOWARE_MINI_DIR="$(rospack find autoware_mini)"


# Function to read parameters from the CSV file and call the processing function
read_params() {
    # Open the CSV file on a different file descriptor (e.g., 3) to avoid stdin issues
    exec 3< "$CSV_FILE"

    # Loop through each line in the CSV file
    while IFS=", " read -r BAG_FILE START DURATION SCENARIO_NUMBER <&3; do

        # Skip lines that are empty or start with a '#'
        [[ -z "$BAG_FILE" || "$BAG_FILE" == \#* ]] && continue

        echo "Processing BAG_FILE: $BAG_FILE with map: $SCENARIO_MAP, START: $START, DURATION: $DURATION, SCENARIO_NUMBER: $SCENARIO_NUMBER"

        # Loop through the values: 'cluster' and 'sfa'
        for DETECTOR in "cluster" "sfa"; do
            # put together output filename
            OUTPUT_FILE="${SCENARIO_TYPE}_${DETECTOR}_${SCENARIO_NUMBER}.bag"

            # Call the processing function with the parameters
            process_bag "$BAG_FILE" "$START" "$DURATION" "$SCENARIO_NUMBER" "$DETECTOR" "$OUTPUT_FILE" &
            wait
        done

    done

    # Close the file descriptor
    exec 3<&-
}

process_bag() {
    local BAG_FILE="$1"
    local START="$2"
    local DURATION="$3"
    local SCENARIO_NUMBER="$4"
    local DETECTOR="$5"
    local OUTPUT_FILE="$6"

    # Step 1: Launch the ROS bag with specified parameters in the background
    echo "  Rerecord bag using ${DETECTOR} detector"

    roslaunch autoware_mini start_bag.launch \
        bag_file:=${BAG_FILE} \
        map_name:=$SCENARIO_MAP \
        detector:=lidar_${DETECTOR} \
        start:=${START} \
        duration:=${DURATION} \
        record_bag:=${OUTPUT_FILE} > /dev/null 2>&1 < /dev/null
    wait

    # Step 2: Run the scenario creation script
    echo "  Creating scenario ${OUTPUT_FILE}"
    ./create_scenario_bag.py $AUTOWARE_MINI_DIR/data/bags/${OUTPUT_FILE} ${OUTPUT_FILE} < /dev/null

    # Step 3: Remove the rerecorded bag file after processing
    rm $AUTOWARE_MINI_DIR/data/bags/${OUTPUT_FILE}

    # Step 4: Move the scenario file to the bag_scenarios directory
    mv ./${OUTPUT_FILE} ${SCENARIO_DIR}
    echo "  Ready!"
}

# Start reading parameters and processing bags
read_params

echo "All commands executed successfully."
