echo "========Start creating namespace fate-10000========"
cd kubefate
sudo kubectl create namespace fate-10000
sudo kubefate cluster install -f examples/party-10000/cluster-spark-pulsar.yaml
echo "========ALL DONE, use [ watch -n 1 -d sudo kubectl get po -n fate-10000 ] to check pod works========="
echo "Notebook should be http://party10000.notebook.example.com:<PORT_NAME>"
echo "Done - go to local and type ssh -i <YOURS_CREDENTIAL> -N -p 22 <USER_NAME>@<EXTERNAL_IP> -L 127.0.0.1:<PORT_NAME>:192.168.49.2:<PORT_NAME>"
cd ../
