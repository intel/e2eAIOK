echo "Splitting the last day into 2 parts of test and validation..."

last_day=$1/day_23
temp_test=$1/test
temp_validation=$1/validation
mkdir -p $temp_test $temp_validation

lines=`wc -l $last_day | awk '{print $1}'`
former=89137319
latter=89137318
head -n $former $last_day > $temp_test/day_23
tail -n $latter $last_day > $temp_validation/day_23
