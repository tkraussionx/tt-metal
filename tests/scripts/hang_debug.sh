#/bin/bash

for i in $(seq 1 10);
do
    echo -e "Running unit tests $i... "
    /bin/bash /home/faraz/tt-metal/tests/scripts/run_python_api_unit_tests.sh
    echo -e "Unit tests completed for iteration $i! "

done

echo -e "All unit tests completed!!!"

for i in $(seq 1 10);
do

    echo -e "Running model $i... "
    /bin/bash /home/faraz/tt-metal/tests/scripts/run_models.sh
    echo -e "Models completed for iteration $i! "

done

echo -e "All model tests completed!!!"
