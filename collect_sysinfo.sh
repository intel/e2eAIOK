sar 3 -P ALL > sar.`date +'%Y%m%d%H%M'`.txt &
pid=$!
turbostat -S > turbostat.`date +'%Y%m%d%H%M'`.txt
kill $pid
