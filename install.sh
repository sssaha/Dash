sudo apt-get update -y
sudo apt-get upgrade -y
sudo apt-get install -y libkrb5-dev gcc



sudo curl -sSL -O https://packages.microsoft.com/config/debian/$(grep VERSION_ID /etc/os-release | cut -d '"' -f 2 | cut -d '.' -f 1)/packages-microsoft-prod.deb
sudo dpkg -i packages-microsoft-prod.deb
sudo rm packages-microsoft-prod.deb
sudo apt-get update
sudo ACCEPT_EULA=Y apt-get install -y msodbcsql17
sudo ACCEPT_EULA=Y apt-get install -y mssql-tools
sudo echo 'export PATH="$PATH:/opt/mssql-tools/bin"' >> ~/.bashrc
sudo . ~/.bashrc
sudo apt-get install -y unixodbc-dev
sudo apt-get install -y libgssapi-krb5-2


# sudo apt-get install lsb-release curl gpg
# curl -fsSL https://packages.redis.io/gpg | sudo gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg
# sudo chmod 644 /usr/share/keyrings/redis-archive-keyring.gpg
# echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/redis.list
# sudo apt-get update
# sudo apt-get install redis

# sudo systemctl enable redis-server
# sudo systemctl start redis-server
