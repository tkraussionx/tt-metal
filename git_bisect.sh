# Check if start and end commit hashes are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <start_commit> <end_commit>"
    exit 1
fi
echo "Start commit: $1"
echo "End commit: $2"

START_COMMIT=$1
END_COMMIT=$2

git bisect start
git bisect bad $END_COMMIT
git bisect good $START_COMMIT

# Tell git bisect to use the run_test function for testing
git bisect run ./git_run.sh

# Finish git bisect
git bisect reset
