# jetson_object_detection
Object Detection on Jetson TX2
A subsent of a project to do object detection (toy detection really) on a jetson TX2 Lego Robot. For example, here are samples of the two classes of objects the model included in this repo can detect :<br>
<br>
Insert IMAGE 1 - class = similar to lego blocks <br>
Insert IMAGE 2 - class = Rubik Cube <br>


Issues to deal with soon:

- [X] ~~MQTT~~
- [X] ~~Async event loop~~
- [X] ~~modify code to read images for scoring via the K4A drivers~~
- [x] ~~Need to fix to get boxes > threshold... and return info to caller (via MQTT queues)~~
- [X] ~~Add support for K4A (Kinect for Azure - with Submodule)~~
- [X] ~~Need to fifx bounding boxes not following (following in reverse for the Y axis)~~
- [X] ~~Check CPU usage during the operation~~
- [X] ~~Moving things around, sometimes there is an exception unable to remove a specific UUID from the list of tracked object. Investigate...~~
- [X] ~~Add that the objects that are kept monitored are ordered by scores... so if max_item = 1, we don't keep score = 60% if one has score = 80%~~
- [X] Need to understand why it's tracking more than 1 object when we tell it not to.
- [X] ~~Remove Tensorflow (remove-tensorflow branch) : ~~
    - [X] ~~almost done! - need to add actual class name to object and get rid of the self.category thing...~~
- [X] ~~Run development from container (vscode-remote-container)~~
- [ ] Find a way to remove dangling processes (maybe garbage collecting some unused queues...???)
- [ ] Figure out distance to object... and propagate it.
- [ ] Tracking Stability :
    - [ ] Need to debug processes when there are 4-5 objects... seems to get confused.
    - [ ] Abiltity to track N objects
    - [ ] Validate that we can remove all objects and they dissapear
    - [ ] Check that K4A still works all the time when removing objects.
- [ ] Updating not only tracking position but score as the TF scoring varies
- [X] CPU usage :

| Nb Objects    |   Cpu Usage (from docker stats)   |
|:--------------|----------------------------------:|
| 1             |   200 %                           |
| 2             |   210 %                           |
| 3             |   210 %                           |
| 4             |   210 %                           |
|


