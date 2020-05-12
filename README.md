# NVIDIA Jetson TX2 Object Detection
This is a subset of a (larger) project ([here, the jetson-robot Lego project](https://github.com/omartin2010/jetson-robot)) to do object detection (toy detection, really) on a Jetson TX2 powered Lego Robot. For example, here are samples of the two classes of objects the model included in this repo can detect :<br>
<br>
Insert IMAGE 1 - class = Similar to lego blocks <br>
Insert IMAGE 2 - class = Rubik Cube <br>

See [here](https://www.youtube.com/watch?v=w8ZtLOhuymo) for an actual example of a video recording of what the system is currently doing. The idea is that this piece of software is to provide a robot the coordinates so that the robot can actually pick up and move these objects around. Bounding boxes are not simply boxes, but tied to Python objects that are tracked as things move around.

### Features

Main features comprise :
- [ ] Ability to record videos and photos of what the camera is seeing (the system is intended to be headless)
- [ ] Uses Microsoft's Kinect for Azure camera and SDK (using this [Python binding](https://github.com/etiennedub/pyk4a))
- [ ] Combines with a EV3 Lego robot (built with [ev3dev](https://www.ev3dev.org/)) to give instructions to actually pick up objects
- [ ] Uses MQTT to receive "orders". Current orders support : 
    - [ ] Video and picture upload to cloud (to Azure Blob) from the Kinect Camera
    - [ ] More to come - some are there for debugging purposes, or changing object properties on the fly without restarting container
- [ ] Run fully containerized
- [ ] Relies on Model Serving from [this project](https://github.com/omartin2010/jetson-model-serving). The model is built using Tensorflow Object Detection API. 


| Nb Objects    |   Cpu Usage (from docker stats)   |
|:--------------|----------------------------------:|
| 1             |   200 %                           |
| 2             |   210 %                           |
| 3             |   210 %                           |
| 4             |   210 %                           |



