cat $1 | awk '{i+=1; sum+=$1;if (i%100==0){print sum;sum=0}}'
