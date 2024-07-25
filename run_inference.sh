#!/bin/bash

# An array of scripts and their corresponding virtual environments
declare -A scripts
scripts["discourse_segmenter/segmenter.py"]=".venv"
scripts["discourse_parser/src/preprocess.py"]=".venv"
scripts["discourse_parser/src/main.py"]=".venv"
scripts["inference/relation_labelling.py"]=".venv"
scripts["inference/question_generation.py"]=".venv"
scripts["inference/distractor_generation.py"]=".venv"

# An array for arguments
declare -A args
args["discourse_parser/src/main.py"]="--inference"

# Function to activate virtual environment
activate_env() {
    echo ""
    echo "Activating environment '$1'"
    source $1/bin/activate
}

# Function to deactivate virtual environment
deactivate_env() {
    echo ""
    echo "Deactivating environment"
    deactivate
}

# Function to run a script with arguments
run_script() {
    local script_path=$1
    local env=${scripts[$script_path]}
    local script_name=$(basename "$script_path")
    local script_dir=$(dirname "$script_path")
    local arguments=${args[$script_path]}

    # Check and adjust script_dir
    if [[ "$script_dir" == *"discourse_parser/"* ]]; then
        script_dir="discourse_parser"
        script_name="${script_path#*$script_dir/}"
    fi

    # Change directory to where the script is located
    cd "$script_dir" || exit
    
    # Activate the virtual environment
    activate_env "$env"

    echo ""
    echo "Running '$script_name' in virtual environment $env from '$script_dir' with args '$arguments'"
    # Run the script and capture any errors
    if python "$script_name" $arguments; then
        echo "'$script_name' completed successfully"
    else
        echo "Error running '$script_name'"
    fi

    # Deactivate the virtual environment
    deactivate_env

    # Return to the original diractory
    cd - > /dev/null
}

# Run scripts in defined order
run_script "discourse_segmenter/segmenter.py"
run_script "discourse_parser/src/preprocess.py"
run_script "discourse_parser/src/main.py"
run_script "inference/relation_labelling.py"
run_script "inference/question_generation.py"
run_script "inference/distractor_generation.py"

echo ""
echo "All files have been run!"
