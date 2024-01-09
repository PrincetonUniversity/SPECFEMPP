## Set OpenMP environment variables
export OMP_PROC_BIND=spread
export OMP_THREADS=places

generate_help_message=false
verbose=false
results_file=/dev/null
## command line argumentes
while getopts d:i:r:e:vh flag
do
    case "${flag}" in
        d) device=${OPTARG};;
        i) base_dir=${OPTARG};;
        r) results_file=${OPTARG};;
        e) executable=${OPTARG};;
        h) generate_help_message=true;;
        v) verbose=true;;
    esac
done

if [ "$generate_help_message" = true ]; then
    echo "Running regression tests - help"
    echo "  -h      :   Print this help message"
    echo "  -d      :   Which tests to run cpu/gpu"
    echo "  -i      :   Location of base directory for the tests."
    echo "              The test folder is structured as base-dir/<testname>/(<cpu> or <gpu>)"
    echo "  -r      :   path to where the results will be written"
    echo "  -v      :   Print the output of the test in this file"
    echo "  -e      :   Path to specfem executible"
    exit 0
fi

# overrite previous results file
echo "results :" > ${results_file}

echo "------------------------------------------"
echo "          Running regression tests        "
echo "------------------------------------------"

itest=1
ntests=$(ls -l ${base_dir}/*/${device}/specfem_config.yaml | wc -l)

echo "Total number of tests = ${ntests}"

echo ""
for test_config in ${base_dir}/*/${device}/specfem_config.yaml ; do
    echo -n "Test ${itest}/${ntests} ..... "
    output=$(${executable} -p ${test_config})
    if [ $? -eq 0 ]; then
        echo "Done"
        test_name=$(echo "$output" | grep "Title :" | cut -d ' ' -f 3-)
        simulation_time=$(echo "$output" | grep "Total solver time " | awk '{print $7}')
        echo "  ${test_name} : ${simulation_time}" >> ${results_file}
    else
        echo "Failed"
    fi

    if [ "$verbose" = true ]; then
        echo "Printing test output :"
        echo "$output"
    fi

    ((itest++))
done

echo "------------------------------------------"
echo "    Finished running regression tests     "
echo "------------------------------------------"
