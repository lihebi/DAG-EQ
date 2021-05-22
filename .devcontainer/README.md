This is a docker container for development. It contains:
- jupyter
- julia kernel
- python kernel

The default CMD for this container is a jupyter instance running on port 8888.
So you probably just need to run the docker-compose. It will set the nvidia
runtime, and bind the port 8888.

```
docker-compose up -d
```

Now go to http://0.0.0.0 and enter the token from the logs of the running container.