cd kubefate
echo "========Start minikube & Docker========"
sudo systemctl start docker
sudo minikube start --memory 8192 --cpus 16 start --driver=docker --force --kubernetes-version v1.19.0 --cni=flannel
echo "========Start ingress & apply yaml========"
sudo minikube addons enable ingress
sudo kubectl apply -f ./rbac-config.yaml
sudo kubectl delete -A ValidatingWebhookConfiguration ingress-nginx-admission
sudo kubectl apply -f ./kubefate.yaml
watch -n 1 -d sudo kubectl get all,ingress -n kube-fate
echo "========Use [ watch -n 1 -d sudo kubectl get all,ingress -n kube-fate ] to check if ingress ready========"
echo "========If ingress ready, run [ source kubefate_create_service_<party_id>]========"
cd ../
