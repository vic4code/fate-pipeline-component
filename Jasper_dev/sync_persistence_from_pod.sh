echo "========Sync Previous Data========"
sudo rm -r persistence
sudo kubectl cp fate-9999/client-0:/data/projects/fate/persistence persistence
echo "========Sync Done========="
