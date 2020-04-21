# NVIDIA Jetson Object Detection
Object Detection on Jetson TX2
A subset of a project to do object detection (toy detection really) on a Jetson TX2 powered Lego Robot. For example, here are samples of the two classes of objects the model included in this repo can detect :<br>
<br>
Insert IMAGE 1 - class = Similar to lego blocks <br>
Insert IMAGE 2 - class = Rubik Cube <br>

See [here](https://www.youtube.com/watch?v=w8ZtLOhuymo) for an actual example of a video recording of what it's doing. The idea is that this piece of software is to provide a robot the coordinates so that the robot can actually pick up and move these objects around.

Outstanding issues:

- [ ] Tracking Stability :
    - [ ] Find a way to remove dangling processes (maybe garbage collecting some unused queues...???)
- [X] Ability to record video:
    - [X] ~~Save to temp file~~
    - [ ] Upload to blob in proper structure (/date/...)
    - [ ] Use logic app to send to onedrive if need be
- [ ] Ability to save image (for future trainings):
    - [ ] Save to temp file
    - [ ] Upload to blob
- [ ] Add a system watchdog to monitor threads and processes and shutdown if important threads are failing... (like video capture)
- [X] Move video display to its own task
- [ ] Improve model - take more pictures
- [ ] Modify the mqtt processing for setting logging level so that dict can be:</br>
```
{
    'logger_id': level,
    'logger_id_2': level
}
```
- [ ] Fix this error when quitting application</br>
```
2020-04-20 11:13:33.720:PID1:WARNING:MainThread:obj_detector_proc_mqtt:Cancelled the MQTT dequeing task.
Traceback (most recent call last):
  File "src/detector.py", line 552, in async_display_video
    await asyncio.sleep(sleep_time)
  File "/usr/lib/python3.6/asyncio/tasks.py", line 482, in sleep
    return (yield from future)
```


Solved issues:

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
- [X] ~~Run development from container (vscode-remote-container)~~
- [X] ~~Figure out distance to object... and propagate it.~~
- [X] ~~Updating not only tracking position but score as the TF scoring varies~~
- [X] Stability : 
    - [X] Distance can't be computed... need to understand why.
    - [X] ~~Need to debug processes when there are 4-5 objects... seems to get confused.~~
    - [X] ~~Abiltity to track N objects~~
    - [X] ~~Validate that we can remove all objects and they dissapear~~
    - [X] ~~Check that K4A still works all the time when removing objects.~~
    - [X] ~~Fix traceback on exit: ~~
                    ```Traceback (most recent call last):</br>
                File "src/detector.py", line 818, in </br>
                remove_tracked_objects</br>
                    self.image_queue.unregister(name=str(obj_id))</br>
                File "src/publish_queues.py", line 12, in inner</br>
                    return func(self, *args, **kwargs)</br>
                File "src/publish_queues.py", line 41, in unregister</br>
                    self._queues.pop(name)</br>
KeyError: 'e7506c84-f9ab-44dc-9999-8018e76f7fd9'</br>```</br>
            ~~that happens when exiting the process.~~
- [X] ~~Investigate Distance issue resetting at every loop:~~
    - [x] ~~Try finding why it comes back to -12 (or constructor's value)~~
    - [X] ~~seems to be recreating a new object at every loop - check other threads... we may well be doing that~~
    - [X] ~~Problem with sometimes bounding box coordinates being negative from object_tracking_processes.py (assert issue)~~
- [X] CPU usage :

| Nb Objects    |   Cpu Usage (from docker stats)   |
|:--------------|----------------------------------:|
| 1             |   200 %                           |
| 2             |   210 %                           |
| 3             |   210 %                           |
| 4             |   210 %                           |



