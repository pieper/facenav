"""
#not needed:
#pip_install("opencv-python")

pip_install("open3d")
pip_install("tensorflow") ;# install twice if it fails at first
pip_install("mtcnn")

rename "/c/Program\ Files/Azure\ Kinect\ SDK\ v1.4.0/tools/k4a.dll" by changing 1.4 to 1.2

#/c/sq5r/Slicer-build/Slicer.exe --python-script c:/pieper/facenav/facenav.py

/c/sr/Slicer-build/Slicer.exe --python-script c:/pieper/facenav/facenav.py

path="c:/pieper/facenav/facenav.py"
exec(open(path).read())



"""

#subject = "manny"
subject = "MRHead"

from mtcnn import MTCNN
import numpy
import open3d
import requests

import SampleData

class FaceFrames:
    def __init__(self):
        self.markups = {"faceFrame": None, "keypoints": None}
        self.faceDetector = MTCNN()
        self.keypoints = ['left_eye', 'right_eye','nose','mouth_left','mouth_right',]
        self.keypointFunctions = {}
        self.faceFunction = vtk.vtkImplicitBoolean()
        self.faceFunction.SetOperationTypeToUnion()
        self.faceExtractor = vtk.vtkExtractPolyDataGeometry()
        self.faceExtractor.SetImplicitFunction(self.faceFunction)

    def update(self, kinect):
        faceImage = kinect.colorArray[0]
        self.faces = self.faceDetector.detect_faces(faceImage)
        if not self.markups["faceFrame"]:
            self.markups["faceFrame"] = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsClosedCurveNode")
            self.markups["faceFrame"].SetName("faceFrame")
            self.markups["faceFrame"].SetCurveTypeToLinear()
            self.markups["faceFrame"].CreateDefaultDisplayNodes()
            self.markups["faceFrame"].GetDisplayNode().SetVisibility(False)
            for point in range(4):
                self.markups["faceFrame"].AddControlPoint(vtk.vtkVector3d(0,0,0))
            self.markups["faceFrame"].SetAndObserveTransformNodeID(kinect.toRAS.GetID())
            self.markups["keypoints"] = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
            self.markups["keypoints"].SetName("keypoints")
            self.markups["keypoints"].CreateDefaultDisplayNodes()
            self.markups["keypoints"].GetDisplayNode().SetVisibility(False)
            for keypoint in self.keypoints:
                self.markups["keypoints"].AddControlPoint(vtk.vtkVector3d(0,0,0), keypoint)
            self.markups["keypoints"].SetAndObserveTransformNodeID(kinect.toRAS.GetID())
            for keypoint in self.keypoints:
                self.keypointFunctions[keypoint] = vtk.vtkSphere()
                self.faceFunction.AddFunction(self.keypointFunctions[keypoint])

        if len(self.faces) >= 1:
            # R is row (y), A is column (x), indexing is [row,column]
            (column,row,w,h) = self.faces[0]['box']
            column2 = min(column+w, kinect.depthArray.shape[2]-1)
            row2 = min(row+h, kinect.depthArray.shape[1]-1)
            cornerIndices = [[row,column], [row,column2], [row2,column2], [row2,column]]
            depth = kinect.depthArray[0][int((row+row2)/2), int((column+column2)/2)]
            for corner in range(4):
                self.markups["faceFrame"].SetNthControlPointPosition(corner, *(cornerIndices[corner]), depth)
            for keypointIndex in range(len(self.keypoints)):
                column,row = self.faces[0]['keypoints'][self.keypoints[keypointIndex]]
                column = min(column, kinect.depthArray.shape[2]-1)
                row = min(row, kinect.depthArray.shape[1]-1)
                depth = kinect.depthArray[0][row, column]
                self.markups["keypoints"].SetNthControlPointPosition(keypointIndex, row, column, depth)
                keypoint = self.keypoints[keypointIndex]
                self.keypointFunctions[keypoint].SetCenter(row, column, depth)
            leftEyeCenter = numpy.array(self.keypointFunctions['left_eye'].GetCenter())
            rightEyeCenter = numpy.array(self.keypointFunctions['right_eye'].GetCenter())
            interpupilaryDistance = numpy.linalg.norm(leftEyeCenter-rightEyeCenter)
            for keypoint in self.keypoints:
                self.keypointFunctions[keypoint].SetRadius(interpupilaryDistance/3)



class KinectStream:

    def __init__(self, config, deviceIndex, align_depth_to_color, update_period=10):
        self.deviceIndex = deviceIndex
        self.deviceString = f"_{self.deviceIndex}"
        self.align_depth_to_color = align_depth_to_color
        self.update_period = update_period
        self.near = 10
        self.far = 1000
        self.running = False
        self.depthVolume = None
        self.colorVolume = None
        self.model = None
        self.toRAS = None
        self.faceFrames = FaceFrames()
        self.faceMesh = None

        self.sensor = open3d.io.AzureKinectSensor(config)
        if not self.sensor.connect(self.deviceIndex):
            raise RuntimeError('Failed to connect to sensor')
        self.cameraAspect = 1280 / 720

        self.perspectiveTransform = vtk.vtkPerspectiveTransform()
        self.perspectiveTransform.Perspective(90, self.cameraAspect, self.near, self.far)

    def createModel(self):
        # Create model node
        planeSource = vtk.vtkPlaneSource()
        shape = self.depthArray.shape[1:]
        planeSource.SetResolution(shape[1]-1, shape[0]-1)
        planeSource.SetOrigin(0.0, 0.0, 0.0)
        planeSource.SetPoint2(shape[0], 0.0, 0.0)
        planeSource.SetPoint1(0.0, shape[1], 0.0)
        self.model = slicer.modules.models.logic().AddModel(planeSource.GetOutputPort())
        self.model.SetName("kinect"+self.deviceString)

        # Tune display properties
        modelDisplay = self.model.GetDisplayNode()
        modelDisplay.SetColor(0.5,0.5,0.5)
        modelDisplay.SetBackfaceCulling(0)
        modelDisplay.SetAmbient(1)
        modelDisplay.SetDiffuse(0)

        colorDataArray = vtk.vtkUnsignedCharArray()
        colorDataArray.SetName("Color")
        colorDataArray.SetNumberOfComponents(3)
        colorDataArray.SetNumberOfTuples(shape[0]*shape[1])
        self.model.GetPolyData().GetPointData().AddArray(colorDataArray)

        self.model.GetDisplayNode().SetActiveScalarName("Color")
        self.model.GetDisplayNode().SetScalarVisibility(True)
        self.model.GetDisplayNode().SetScalarRangeFlag(self.model.GetDisplayNode().UseDirectMapping)

    def updateModel(self):
        colorArray = slicer.util.arrayFromModelPointData(self.model, "Color")
        colorArray.reshape(self.colorArray.shape)[:] = self.colorArray
        slicer.util.arrayFromModelPointDataModified(self.model, "Color")
        points = slicer.util.arrayFromModelPoints(self.model)
        points[:,2] = numpy.clip(self.depthArray.flatten(), self.near, self.far)
        zPoints = points[:,2]
        zPoints[zPoints==self.near] = self.far
        slicer.util.arrayFromModelPointsModified(self.model)

    def step(self):
        if not self.running:
            return
        self.rgbd = self.sensor.capture_frame(self.align_depth_to_color)
        if self.rgbd is not None:
            self.depthArray = numpy.asarray(self.rgbd.depth)
            newshape = list(self.depthArray.shape)
            newshape.insert(0,1)
            self.depthArray = self.depthArray.reshape(newshape)
            self.colorArray = numpy.asarray(self.rgbd.color)
            newshape = list(self.colorArray.shape)
            newshape.insert(0,1)
            self.colorArray = self.colorArray.reshape(newshape)
            if not self.depthVolume or not self.colorVolume:
                self.depthVolume = slicer.util.addVolumeFromArray(self.depthArray, name="depth"+self.deviceString)
                self.colorVolume = slicer.util.addVolumeFromArray(self.colorArray, name="color"+self.deviceString, nodeClassName="vtkMRMLVectorVolumeNode")
                self.createModel()
                self.toRAS = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode")
                self.toRAS.SetName("kinectToRAS"+self.deviceString)
                toRASMatrix = numpy.array( [[0, 1, 0, -self.depthArray.shape[2]/2],
                                            [0, 0, -1, self.near],
                                            [-1, 0, 0, self.depthArray.shape[1]/2],
                                            [0, 0, 0, 1]] )
                slicer.util.updateTransformMatrixFromArray(self.toRAS, toRASMatrix)
                for transformable in [self.model, self.depthVolume, self.colorVolume]:
                    transformable.SetAndObserveTransformNodeID(self.toRAS.GetID())

            slicer.util.updateVolumeFromArray(self.depthVolume, self.depthArray)
            slicer.util.updateVolumeFromArray(self.colorVolume, self.colorArray)
            self.updateModel()
            self.faceFrames.update(self)

            if self.faceFrames.markups["faceFrame"] is not None:
                facePoints = vtk.vtkPoints()
                self.faceFrames.markups["keypoints"].GetControlPointPositionsWorld(facePoints)
                landmarkTransform = vtk.vtkLandmarkTransform()
                # landmarkTransform.SetModeToRigidBody()
                landmarkTransform.SetModeToSimilarity()
                landmarkTransform.SetSourceLandmarks(scanPoints)
                landmarkTransform.SetTargetLandmarks(facePoints)
                landmarkTransform.Update()
                scanTransform.SetMatrixTransformToParent(landmarkTransform.GetMatrix())

            if self.faceMesh is None:
                self.faceMesh = slicer.modules.models.logic().AddModel(vtk.vtkPolyData())
                self.faceMesh.SetName("faceMesh"+self.deviceString)
                self.faceMesh.SetAndObserveTransformNodeID(self.toRAS.GetID())
            self.faceFrames.faceExtractor.SetInputData(self.model.GetPolyData())
            self.faceFrames.faceExtractor.Update()
            self.faceMesh.SetAndObservePolyData(self.faceFrames.faceExtractor.GetOutputDataObject(0))

        qt.QTimer.singleShot(self.update_period, self.step)

    def go(self):
        self.running = True
        self.step()

    def stop(self):
        self.running = False

try:
    kinects
except NameError:
    exectedDeviceCount = 1
    kinects = {}
    azureConfig = { "color_format": "K4A_IMAGE_FORMAT_COLOR_MJPG",    # defaults
                    "color_resolution": "K4A_COLOR_RESOLUTION_720P",
                    "depth_mode": "K4A_DEPTH_MODE_WFOV_2X2BINNED",
                    "camera_fps": "K4A_FRAMES_PER_SECOND_30",
                    "synchronized_images_only": "false",
                    "depth_delay_off_color_usec": "0",
                    "wired_sync_mode": "K4A_WIRED_SYNC_MODE_STANDALONE",
                    "subordinate_delay_off_master_usec": "0",
                    "disable_streaming_indicator": "false" }
    azureConfig['depth_mode'] = "K4A_DEPTH_MODE_NFOV_UNBINNED"
    config = open3d.io.AzureKinectSensorConfig(azureConfig)
    align_depth_to_color = True
    for deviceIndex in range(exectedDeviceCount):
        kinects[deviceIndex] = KinectStream(config, deviceIndex, align_depth_to_color)

try:
    scan = slicer.util.getNode(subject)
except slicer.util.MRMLNodeNotFoundException:
    renderLogic = slicer.modules.volumerendering.logic()
    if subject == "MRHead":
        scanFaceMesh = slicer.util.loadModel("d:/data/facenav/MRHead-face/Segment_1.vtk")
        scanFaceMesh.SetName("scanFaceMesh")
        scan = SampleData.SampleDataLogic().downloadMRHead()
        renderPreset = renderLogic.GetPresetByName('MR-Default')
        scanKeypointPositions = [ [-32.62567138671875, 93.67095184326172, -24.12122917175293],
                                [36.235939025878906, 91.43257141113281, -25.009567260742188],
                                [2.620298559948708, 136.70169243604, -55.57851810061035],
                                [-33.98469543457031, 97.54539489746094, -81.3556900024414],
                                [37.931026458740234, 96.51762390136719, -82.2235336303711] ]
    else:
        #scanFaceMesh = slicer.util.loadModel("d:/data/facenav/manny-face/manny-face.vtk")
        scanFaceMesh = slicer.util.loadModel("d:/data/facenav/manny-face/manny.vtk")
        scanFaceMesh.SetName("scanFaceMesh")
        scan = slicer.util.loadVolume("d:/data/manny.nrrd")
        scan.SetName("scan")
        renderPreset = renderLogic.GetPresetByName('CT-Chest-Contrast-Enhanced')
        scanKeypointPositions = [ [-31.546663284301758, 268.36932373046875, -154.3578338623047],
                                  [30.530109405517578, 266.26531982421875, -155.9859161376953],
                                  [-2.1105196475982666, 290.3841247558594, -186.3923797607422],
                                  [-23.16329002380371, 261.33074951171875, -215.99192810058594],
                                  [22.74981689453125, 257.8655700683594, -215.7384490966797] ]

    keypointFunctions = {}
    faceFunction = vtk.vtkImplicitBoolean()
    faceFunction.SetOperationTypeToUnion()
    faceExtractor = vtk.vtkExtractPolyDataGeometry()
    faceExtractor.SetImplicitFunction(faceFunction)
    for keypoint,position in zip(kinects[deviceIndex].faceFrames.keypoints,scanKeypointPositions):
        keypointFunctions[keypoint] = vtk.vtkSphere()
        keypointFunctions[keypoint].SetCenter(position)
        faceFunction.AddFunction(keypointFunctions[keypoint])
    leftEyeCenter = numpy.array(keypointFunctions['left_eye'].GetCenter())
    rightEyeCenter = numpy.array(keypointFunctions['right_eye'].GetCenter())
    interpupilaryDistance = numpy.linalg.norm(leftEyeCenter-rightEyeCenter)
    for keypoint,position in zip(kinects[deviceIndex].faceFrames.keypoints,scanKeypointPositions):
        keypointFunctions[keypoint].SetRadius(interpupilaryDistance/3)
    faceExtractor.SetInputData(scanFaceMesh.GetPolyData())
    faceExtractor.Update()
    scanFaceMesh.SetAndObservePolyData(faceExtractor.GetOutputDataObject(0))


    displayNode = renderLogic.CreateDefaultVolumeRenderingNodes(scan)
    displayNode.SetVisibility(True)
    displayNode.GetVolumePropertyNode().Copy(renderPreset)
    scanTransform = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode")
    scanTransform.SetName("ScanToFace")
    scan.SetAndObserveTransformNodeID(scanTransform.GetID())
    scanFaceMesh.SetAndObserveTransformNodeID(scanTransform.GetID())

scanBounds = [0]*6
scan.GetRASBounds(scanBounds)
scanMarkups = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsClosedCurveNode")
scanMarkups.SetName("scanFrame")
scanMarkups.SetAndObserveTransformNodeID(scanTransform.GetID())
scanMarkups.SetCurveTypeToLinear()
scanMarkups.CreateDefaultDisplayNodes()
scanMarkups.GetDisplayNode().SetVisibility(False)
scanFrameVertices = [ [scanBounds[0], scanBounds[3], scanBounds[5]],
                    [scanBounds[1], scanBounds[3], scanBounds[5]],
                    [scanBounds[1], scanBounds[3], scanBounds[4]],
                    [scanBounds[0], scanBounds[3], scanBounds[4]] ]

for point in range(4):
    scanMarkups.AddControlPoint(vtk.vtkVector3d(*scanFrameVertices[point]))


layoutManager = slicer.app.layoutManager()
layoutManager.threeDWidget(0).mrmlViewNode().SetBoxVisible(False)
layoutManager.threeDWidget(0).mrmlViewNode().SetAxisLabelsVisible(False)

scanKeypoints = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
scanKeypoints.SetName("scanKeypoints")
scanKeypoints.CreateDefaultDisplayNodes()
scanKeypoints.GetDisplayNode().SetVisibility(False)
for keypoint,position in zip(kinects[deviceIndex].faceFrames.keypoints,scanKeypointPositions):
    index = scanKeypoints.AddControlPoint(vtk.vtkVector3d(*position), "scan_" + keypoint)
    scanKeypoints.UnsetNthControlPointPosition(index)
scanKeypoints.SetAndObserveTransformNodeID(scanTransform.GetID())
scanPoints = vtk.vtkPoints()
scanKeypoints.GetControlPointPositionsWorld(scanPoints)


# TODO:
# - use keypoint functions and extract geometry on scan skin mesh
#   - vary extraction based on angle of keypoints (head pose)?
# - harden meshes so ICP gives relative transform and make transform tree
#   - make interactive ICP transform testing?
#   - try open3d ICP?
#   - make vtkAddon of ICP with initializer?
# - run icp on scan and between each kinect model (treat 0 as reference frame)
# - perspective divide x and y values of kinect model
# - add SH folders for kinects and scan for convenience
icp = vtk.vtkIterativeClosestPointTransform()
icp.SetMaximumNumberOfIterations(500)
icp.SetMaximumNumberOfLandmarks(1000)
def scanFaceMeshToKinect():
    faceMesh_0 = slicer.util.getNode("faceMesh_0")
    icp.GetLandmarkTransform().SetModeToRigidBody()
    icp.SetTarget(faceMesh_0.GetPolyData())
    icp.SetSource(scanFaceMesh.GetPolyData())
    icp.Update()
    scanTransform.SetMatrixTransformToParent(icp.GetMatrix())



def go(deviceIndex=None):
    if deviceIndex is not None:
        kinects[deviceIndex].go()
    else:
        for deviceIndex in range(exectedDeviceCount):
            kinects[deviceIndex].go()

def stop():
    for deviceIndex in range(exectedDeviceCount):
        kinects[deviceIndex].stop()

go(0)
