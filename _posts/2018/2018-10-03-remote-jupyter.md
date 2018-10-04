---
layout: post
title: How to set up a remote jupyter notebook/ lab
category: tutorial
tags: [jupyter Notebook, jupyter Lab, ssh, server]
---


## Setting up a Jupyter Notebook remote server
------


[Setting up a Jupyter Lab remote server](https://agent-jay.github.io/2018/03/jupyterserver/) is the tutorial I referenced from.

1. Firt of all, start on the server:

```
linyange@nellodee:~$ jupyter notebook --generate-config
Writing default config to: /home/you/.jupyter/jupyter_notebook_config.py
jupyter notebook password
Enter password: ****
Verify password: ****
[NotebookPasswordApp] Wrote hashed password to /home/you/.jupyter/jupyter_notebook_config.json
```

```
$ openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout mycert.pem -out mycert.pem
```

You will need to answer a couple of basic question which is needed to generate a hash file.
The file will be named after *mycert.pem* which you will need later. You will find it in the current path.

Open the `/home/you/.jupyter/jupyter_notebook_config.py` file and then change thins below:(remember to remove all the sharp sign from the biginning of these line.)

```
  # Set options for certfile, ip, password, and toggle off
  # browser auto-opening
  c.NotebookApp.certfile = u'/absolute/path/to/your/certificate/mycert.pem'
  c.NotebookApp.keyfile = u'/absolute/path/to/your/certificate/mycert.pem'
  # Set ip to '*' to bind on all interfaces (ips) for the public server
  c.NotebookApp.ip = '*'
  c.NotebookApp.password = u'sha1:bcd259ccf...<your hashed password here>'
  c.NotebookApp.open_browser = False

  # It is a good idea to set a known, fixed port for server access
  c.NotebookApp.port = 9999
```

Then just run `jupyter notebook` on the server.
Then type the line below on your local laptop:

```
ssh -N -f -L 8888:localhost:9999 user@domain.com
```
Then open a web browser on local and type https://localhost:8888/
(do not do http)

Then you are done!
