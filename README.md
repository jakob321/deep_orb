# How to run 🚀:

1. Update submodules ```git submodule update --recursive --remote```
2. Build dockerfile ```docker build -t deep_orb .```
3. Start container ```docker compose run deep_orb```

### ORB_SLAM3 setup
4. Build ORB_SLAM3 by running the build script in the container ```cd ORB_SLAM3 && ./build.sh```
5. Verify ORB_SLAM3 it works with: TODO:FILL IN EXAMPLE COMMAND

### ml-depth-pro setup
6. Download weights by ```cd ml-depth-pro && chmod +x get_pretrained_models.sh && ./get_pretrained_models.sh```