# jetson_object_detection
Object Detection on Jetson TX2
A subsent of a project to do object detection (toy detection really) on a jetson TX2 Lego Robot

- [X] ~~MQTT~~
- [X] ~~Async event loop~~
- [X] ~~modify code to read images for scoring via the K4A drivers~~
- [x] ~~Need to fix to get boxes > threshold... and return info to caller (via MQTT queues)~~
- [X] ~~Add support for K4A (Kinect for Azure - with Submodule)~~
- [X] ~~Need to fifx bounding boxes not following (following in reverse for the Y axis)~~
- [ ] Need to understand why it's tracking more than 1 object when we tell it not to.
- [ ] Check CPU usage during the operation
- [ ] Moving things around, sometimes there is an exception unable to remove a specific UUID from the list of tracked object. Investigate...
- [ ] Add that the objects that are kept monitored are ordered by scores... so if max_item = 1, we don't keep score = 60% if one has score = 80%