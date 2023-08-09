# !/bin/bash
timeout 5 ssh $1 hostname
if [ $? -eq 0 ]; then
    exit
fi
mkdir -p ~/.ssh
cd ~/.ssh
if [ ! -f ~/.ssh/id_rsa ]
  then
    ssh-keygen -t rsa -N "" -f id_rsa;ssh-add -k id_rsa
fi
sshpass -p docker ssh -o StrictHostKeyChecking=no $1 mkdir -p /root/.ssh/
sshpass -p docker scp -o StrictHostKeyChecking=no ~/.ssh/id_rsa.pub $1://root/.ssh/authorized_keys
sshpass -p docker ssh -o StrictHostKeyChecking=no $1 eval $(ssh-agent)
