# !/bin/bash
source_dir="/home/vmagent/app/dataset/amazon_reviews/"
dest_dir="/home/vmagent/app/dataset/amazon_reviews_distributed"
conf="/home/vmagent/app/e2eaiok/conf/e2eaiok_defaults_dien_example.conf"

# load hosts from yaml
yaml_hosts() {
    python3 -c "import yaml;print(' '.join(yaml.safe_load(open('${conf}'))['hosts']))"
}

yaml_ppn() {
    python3 -c "import yaml;print(len(yaml.safe_load(open('${conf}'))['hosts']))"
}

split_func() {
    num_copy=$2
    num_total=`wc -l < $1/local_train_splitByUser`
    slice=$((${num_total}/${num_copy}))
    echo split -l${slice} $1/local_train_splitByUser $1/local_train_splitByUser.slice -da 2
    split -l${slice} $1/local_train_splitByUser $1/local_train_splitByUser.slice -da 2
}

HOSTS=$(yaml_hosts)
num_copy=$(yaml_ppn)
echo will distributed data to $HOSTS, total ${num_copy} nodes

# split data
split_func ${source_dir}/train ${num_copy}

# prepare distributed meta
echo "uid_voc: ${dest_dir}/uid_voc.pkl" > ${source_dir}/meta_d.yaml
echo "mid_voc: ${dest_dir}/mid_voc.pkl" >> ${source_dir}/meta_d.yaml
echo "cat_voc: ${dest_dir}/cat_voc.pkl" >> ${source_dir}/meta_d.yaml

# scp data
i=0
for node in $HOSTS; do
    ssh ${node} mkdir -p ${dest_dir} ${dest_dir}/train
    scp ${source_dir}/train/local_train_splitByUser.slice0${i} ${node}:/${dest_dir}/train/local_train_splitByUser
    scp ${source_dir}/*pkl ${node}:/${dest_dir}
    scp -r ${source_dir}/valid/ ${node}:/${dest_dir}
    scp ${source_dir}/meta_d.yaml ${node}:/${dest_dir}/meta.yaml
    i=$((i+1))
done
rm ${source_dir}/train/local_train_splitByUser.slice0*