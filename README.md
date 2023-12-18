<div align="center">
 <h1>Class Contagion</h1>
</div>

<br>

[Class Contagion](https://github.com/Yamilhb/yolo-class-contagion) is an excercise (not finished) with yoloV8 that aims to explore the tracking capabilities of the model.

The goal of the project is to find, in real time, the person who made the action that we are looking for (the event). To do that, we have to detect the person involved in the event, and for simplicity, I will work with objects (we can think that the actions must be done with objects, for example, the action of *shooting* must be done with a *gun*).

Therefore, what I am doing here is associate the ***target*** object with the person who has it, changing its bounding box in a proper way. This is the process I called ***Class Contagion***:

The next example shows an *Android* with its bounding box, but whe 'he' take the *cup*, his state changes and his label says *Contagion by a cup* instead of *Android* (onwards i will refer to the android as a man):

<div align="center">
<img src="https://github.com/Yamilhb/yolo-class-contagion/blob/master/class_contagion/outputs/gif20%3A23%3A24.gif">
</div>

This new label will persist until the man goes beyond the limits of the image. In this case, we can go to the 'resultados' folder and see the fragment of the video where the event happens. Indeed, the video is saved automatically seconds before the desired event. It can be checked in the gif or in [class_contagion/outputs/2023-11-26 20:23:05.mp4](https://github.com/Yamilhb/yolo-class-contagion/blob/master/class_contagion/outputs/2023-11-26%2020%3A23%3A05.mp4) or [class_contagion/outputs/2023-11-26 20:23:24.mp4](https://github.com/Yamilhb/yolo-class-contagion/blob/master/class_contagion/outputs/2023-11-26%2020%3A23%3A24.mp4) (the file with the word *prueba*... is the whole test, and the other videos are the automatic savings).


