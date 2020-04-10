# jetson_object_detection
Object Detection on Jetson TX2
A subsent of a project to do object detection (toy detection really) on a jetson TX2 Lego Robot

- [X] ~~MQTT~~
- [X] ~~Async event loop~~
- [X] ~~modify code to read images for scoring via the K4A drivers~~
- [x] ~~Need to fix to get boxes > threshold... and return info to caller (via MQTT queues)~~
- [X] ~~Add support for K4A (Kinect for Azure - with Submodule)~~
- [ ] Need to fifx bounding boxes not following (following in reverse for the Y axis)
- [ ] Need to understand why it's tracking more than 1 object when we tell it not to.
- [ ] Check CPU usage during the operation