echo "========Sync Previous Data========"
sudo kubectl cp persistence fate-9999/client-0:/data/projects/fate
sudo kubectl cp persistence fate-10000/client-0:/data/projects/fate
echo "========Use command to check pod inside [ sudo kubectl exec -it client-0 -n fate-9999 bash ]========="
