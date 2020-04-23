# Robot Startup Outside VSCode
There are two components of this to be run on this piece of hardware.
1. Model Serving
2. Object Detector Robot Software

Each of these startups are described below.

## Model Serving
Details on this are on another project related project.
```
docker run --restart unless-stopped --privileged -d \
    --name modelserving \
    --memory 2g \
    --memory-swap 3g \
    --network host \
    modelserving:latest
```

## Object Detector

### Environment Variables
You need to define a few environment variables for credentials in some file... (`.env.list` or other, see `docker` command below).
```
export AZURE_TENANT_ID=...
export AZURE_CLIENT_ID=...
export AZURE_CLIENT_SECRET=...
```
Docker command to run this. IT can be in `.xsessionrc` as this needs to be run with the session started.

```
export DISPLAY=:0;
xhost +si:localuser:root;
docker run --restart unless-stopped --privileged -d \
    --name objectdetector \
    --network host \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --env-file ~/proj/objectdetector/.env.list \
    --memory 2g \
    --memory-swap 3g \
    --cpus=2.5 \
    objectdetector:latest
```

