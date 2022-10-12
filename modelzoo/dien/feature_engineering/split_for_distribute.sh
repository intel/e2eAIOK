num_copy=$2
num_total=`wc -l < $1/local_train_splitByUser`
echo ${num_copy}
echo ${num_total}
slice=$((${num_total}/${num_copy}))
echo split -l${slice} $1/local_train_splitByUser $1/local_train_splitByUser.slice -da 2
split -l${slice} $1/local_train_splitByUser $1/local_train_splitByUser.slice -da 2
