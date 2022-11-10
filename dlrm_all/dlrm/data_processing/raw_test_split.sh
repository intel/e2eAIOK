#!/bin/bash
echo "Splitting the last day into 2 parts of test and validation..."
HADOOP_PATH="/home/hadoop-3.3.1"
last_day=$1/day_23
temp_test=$1/test
temp_validation=$1/validation

former=89137319
latter=89137318

if [[ "$last_day" =~ ^hdfs.* ]]
then
    echo "write to hdfs"
    $HADOOP_PATH/bin/hdfs dfs -test -e $temp_test
    if [ $? -eq 0 ] ;then
        $HADOOP_PATH/bin/hdfs dfs -rm -r $temp_test
    fi
    $HADOOP_PATH/bin/hdfs dfs -test -e $temp_validation
    if [ $? -eq 0 ] ;then
        $HADOOP_PATH/bin/hdfs dfs -rm -r $temp_validation
    fi
    $HADOOP_PATH/bin/hdfs dfs -cat $last_day | head -$former | $HADOOP_PATH/bin/hdfs dfs -appendToFile - $temp_test/day_23
    $HADOOP_PATH/bin/hdfs dfs -cat $last_day | tail -$latter | $HADOOP_PATH/bin/hdfs dfs -appendToFile - $temp_validation/day_23
else
    echo "write to local"
    temp_test="${temp_test#file://}"
    temp_validation="${temp_validation#file://}"
    last_day="${last_day#file://}"
    mkdir -p $temp_test $temp_validation
    head -n $former $last_day > $temp_test/day_23
    tail -n $latter $last_day > $temp_validation/day_23
fi