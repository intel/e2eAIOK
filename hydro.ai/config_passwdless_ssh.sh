if [ $# -eq 0 ]
  then
    echo "bash" $0 "hostname"
    exit
  else
    echo "bash" $0 $1
fi
mkdir -p ~/.ssh
cd ~/.ssh
if [ ! -f ~/.ssh/id_rsa ]
  then
    ssh-keygen -t rsa -N "" -f id_rsa;ssh-add -k id_rsa
fi
scp id_rsa.pub $1://root/.ssh/authorized_keys
ssh $1 eval $(ssh-agent)
