# Yoga AI

## System Configuration 

    - python3.11
    - nginx
    - certbot
    - daphne
    - supervisor

### Installing python 3.11

```bash
sudo apt update
sudo apt upgrade
sudo apt-get install build-essential cmake pkg-config libx11-dev libatlas-base-dev libgtk-3-dev libboost-python-dev

# Useful for installing venv and other libraries

# python venv
sudo apt install python3.10-venv

# Setup to install python3.11

sudo add-apt-repository ppa:deadsnakes/ppa

sudo apt update

sudo apt-get install python3.11

python3 --version

python3.11 --version

sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 2

sudo update-alternatives --config python3

ython3 --version

sudo apt install python-is-python3

python --version

cd /usr/lib/python3/dist-packages/

sudo cp apt_pkg.cpython-310-x86_64-linux-gnu.so apt_pkg.so

cd

curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py

python get-pip.py

python -m pip --version

# You can also do this for ~/.zshrc
nano ~/.bashrc

# add this line to the end of file, save and exit the file
alias pip='python -m pip'

pip --version
pip install --upgrade pip

# Installing venv 3.11
sudo apt install python3.11-venv


```


### Installing nginx cerbot supervisor

```bash
# Installing nginx and supervisor
sudo apt-get install nginx supervisor

# Installing Cerbot

sudo snap install core; sudo snap refresh core

sudo snap install --classic certbot


sudo ln -s /snap/bin/certbot /usr/bin/certbot

```


### Cloning git repo

```bash
git clone https://github.com/jaisPank/yoga.git

```

### Django Configuration 

```bash
# Creating a venv

python -m venv env

# Activating env
source env/bin/activate

# Installing Requirements.txt
cd yoga
pip install -r requirements.txt --no-cache

# Checking django code
python manage.py check
python manage.py makemigrations
python manage.py migrate

## Testing with daphne , run this command from where manage.py is there.

daphne -b 0.0.0.0 -p 8001 myproject.asgi:application

## Once it is running browse url with ip address of manchine
```

### Setting up daphne service

```bash
sudo nano /etc/supervisor/conf.d/daphne.conf 
sudo mkdir /var/log/daphne/
sudo mkdir -p /run/daphne
sudo chown yoga:yoga /run/daphne
sudo chmod 775 /run/daphne
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start daphne
sudo supervisorctl status



# daphne.conf, change command and dir as per your configuration

[program:daphne]
command=/home/yoga/env/bin/daphne -u /run/daphne/daphne%(process_num)s.sock -b 0.0.0.0 -p 8001 myproject.asgi:application
directory=/home/yoga/yoga
user=yoga
group=www-data
autostart=true
autorestart=true
stdout_logfile=/var/log/daphne/daphne.log
stderr_logfile=/var/log/daphne/daphne_error.log
numprocs=2  ; Adjust this value based on the number of CPU cores
process_name=%(program_name)s_%(process_num)02d

```


#### Setting Up nginx

```bash

# cerbot step have already done

sudo nano /etc/nginx/sites-available/yoga.xpponet.in


## config details

map $http_upgrade $connection_upgrade {
    default upgrade;
    '' close;
}

upstream websocket {

    server unix:/run/daphne/daphne0.sock;
    server unix:/run/daphne/daphne1.sock;

}

server {
    server_name yogix.xpponet.in;
    client_body_buffer_size 512k;
    client_header_buffer_size 4k;
    client_max_body_size 100M;
    large_client_header_buffers 4 8k;
    proxy_headers_hash_max_size 512;
    proxy_headers_hash_bucket_size 64;

    location / {
 
        proxy_pass http://websocket;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /static/ {
        alias /home/yoga/yoga/myapp/static/;
        expires 30d;  # Cache static files for 30 days
        access_log off;  # Disable logging for static files
    }

    location /media/ {
        alias /home/yoga/yoga/myapp/media/;
        expires 30d;  # Cache media files for 30 days
        access_log off;  # Disable logging for media files
    }

    location /ws/ {
        # proxy_pass http://127.0.0.1:8001;
        proxy_pass http://websocket;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection $connection_upgrade;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 6000s;
        proxy_send_timeout 6000s;
        proxy_buffering off;  # Disable proxy buffering
        proxy_cache_bypass $http_upgrade;  # Added
        proxy_no_cache $http_upgrade;      # Added
        proxy_set_header X-NginX-Proxy true;
        proxy_ssl_certificate /etc/letsencrypt/live/yogix.xpponet.in/fullchain.pem;
        proxy_ssl_certificate_key /etc/letsencrypt/live/yogix.xpponet.in/privkey.pem;
    }

    location ~* ^/media/wp-includes/ {
        return 444;
    }

    listen 443 ssl; # managed by Certbot
    ssl_certificate /etc/letsencrypt/live/yogix.xpponet.in/fullchain.pem; # managed by Certbot
    ssl_certificate_key /etc/letsencrypt/live/yogix.xpponet.in/privkey.pem; # managed by Certbot
    include /etc/letsencrypt/options-ssl-nginx.conf; # managed by Certbot
    ssl_dhparam /etc/letsencrypt/ssl-dhparams.pem; # managed by Certbot

    gzip on;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;
    gzip_min_length 1024;
    gzip_comp_level 5;
    gzip_vary on;
    gzip_disable "msie6";
}

server {
    if ($host = yogix.xpponet.in) {
        return 301 https://$host$request_uri;
    } # managed by Certbot


        listen 80 ;
        listen [::]:80 ;
    server_name yogix.xpponet.in;
    return 404; # managed by Certbot

}




## nginx restart
sudo nginx -t
sudo systemctl reload nginx
sudo systemctl restart nginx


## Delete default nginx file from sites-enabled and restart nginx
sudo rm /etc/nginx/sites-enabled/default
sudo systemctl restart nginx
sudo systemctl status ngin
```

