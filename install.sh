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