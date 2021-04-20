using System;
using UnityEngine;
using UnityEngine.UI;

[Serializable]
public class MSACC_CameraType {
	[Tooltip("A camera must be associated with this variable. The camera that is associated here, will receive the settings of this index.")]
	public Camera _camera;
	public enum TipoRotac{LookAtThePlayer, FirstPerson, FollowPlayer, Orbital, Stop, StraightStop, OrbitalThatFollows, ETS_StyleCamera, FlyCamera_OnlyWindows, PlaneX_Z}
	[Tooltip("Here you must select the type of rotation and movement that camera will possess.")]
	public TipoRotac rotationType = TipoRotac.LookAtThePlayer;
	[Range(0.01f,1.0f)][Tooltip("Here you must adjust the volume that the camera attached to this element can perceive. In this way, each camera can perceive a different volume.")]
	public float volume = 1.0f;
}
[Serializable]
public class MSACC_CameraSetting {
	[Header("Configure Inputs")]
	[Tooltip("The input that will define the horizontal movement of the cameras.")]
	public string inputMouseX = "Mouse X";
	[Tooltip("The input that will define the vertical movement of the cameras.")]
	public string inputMouseY = "Mouse Y";
	[Tooltip("The input that allows you to zoom in or out of the camera.")]
	public string inputMouseScrollWheel = "Mouse ScrollWheel";
	[Tooltip("In this variable you can configure the key responsible for switching cameras.")]
	public KeyCode cameraSwitchKey = KeyCode.C;

	public enum UpdateMode {Update, FixedUpdate, LateUpdate};
	[Header("Update mode")]
	[Tooltip("Here it is possible to decide whether the motion of the cameras will be processed in the void Update, FixedUpdate or LateUpdate. The mode that best suits most situations is the 'LateUpdate'.")]
	public UpdateMode camerasUpdateMode = UpdateMode.LateUpdate;

	[Header("General settings")]
	[Tooltip("If this variable is checked, the script will automatically place the 'IgnoreRaycast' layer on the player when needed.")]
	public bool ajustTheLayers = true;
	[Tooltip("In this class you can configure the 'FirstPerson' style cameras.")]
	public MSACC_SettingsCameraFirstPerson firstPerson;
	[Tooltip("In this class you can configure the 'FollowPlayer' style cameras.")]
	public MSACC_SettingsCameraFollow followPlayer;
	[Tooltip("In this class you can configure the 'Orbital' style cameras.")]
	public MSACC_SettingsCameraOrbital orbital;
	[Tooltip("In this class you can configure the 'OrbitalThatFollows' style cameras.")]
	public MSACC_SettingsCameraOrbitalThatFollows OrbitalThatFollows;
	[Tooltip("In this class you can configure the 'ETS_StyleCamera' style cameras.")]
	public MSACC_SettingsCameraETS_StyleCamera ETS_StyleCamera;
	[Tooltip("In this class you can configure the 'FlyCamera' style cameras.")]
	public MSACC_SettingsFlyCamera FlyCamera_OnlyWindows;
    [Tooltip("In this class you can configure the 'PlaneX_Y' style cameras.")]
    public MSACC_SettingsCameraPlaneX_Z PlaneX_Z;
}
[Serializable]
public class MSACC_SettingsCameraFirstPerson {
	[Header("Sensibility")]
	[Range(1,20)][Tooltip("Horizontal camera rotation sensitivity.")]
	public float sensibilityX = 10.0f;
	[Range(1,20)][Tooltip("Vertical camera rotation sensitivity.")]
	public float sensibilityY = 10.0f;
	[Range(0,1)][Tooltip("The speed with which the camera can approach your vision through the mouseScrool.")]
	public float speedScroolZoom = 0.5f;
	[Header("Limits")]
	[Range(0,360)][Tooltip("The highest horizontal angle that camera style 'FistPerson' camera can achieve.")]
	public float horizontalAngle = 65.0f;
	[Range(0,85)][Tooltip("The highest vertical angle that camera style 'FistPerson' camera can achieve.")]
	public float verticalAngle = 20.0f;
	[Range(0,40)][Tooltip("The maximum the camera can approximate your vision.")]
	public float maxScroolZoom = 30.0f;

	[Header("Custom Rotation Input")]
	[Tooltip("If this variable is true, the camera will only rotate when the key selected in the 'KeyToRotate' variable is pressed. If this variable is false, the camera can rotate freely, even without pressing any key.")]
	public bool rotateWhenClick = false;
	[Tooltip("Here you can select the button that must be pressed in order to rotate the camera.")]
	public string keyToRotate = "mouse 0";

	[Space(7)]
	[Tooltip("If this variable is true, the X-axis input will be inverted.")]
	public bool invertXInput = false;
	[Tooltip("If this variable is true, the X-axis input will be inverted.")]
	public bool invertYInput = false;
}
[Serializable]
public class MSACC_SettingsCameraFollow {
	[Header("Collision")]
	[Tooltip("If this variable is true, the camera ignores the colliders and crosses the walls freely.")]
	public bool ignoreCollision = false;

	[Header("Movement")]
	[Range(1,20)][Tooltip("The speed at which the camera can follow the player.")]
	public float displacementSpeed = 3.0f;

	[Header("Rotation")]
	[Tooltip("If this variable is true, the code makes a lookAt using quaternions.")]
	public bool customLookAt = false;
	[Range(1,30)][Tooltip("The speed at which the camera rotates as it follows and looks at the player.")]
	public float spinSpeedCustomLookAt = 15.0f;

	[Header("Use Scrool")]
	[Tooltip("If this variable is true, the 'FollowPlayer' camera style will have the player's distance affected by the mouse scrool. This will allow the player to zoom in or out of the camera.")]
	public bool useScrool = false;
	[Range(0.01f,2.0f)][Tooltip("The speed at which the player can zoom in and out of the camera.")]
	public float scroolSpeed = 1.0f;
	[Range(1,30)][Tooltip("The minimum distance the camera can be relative to the player.")]
	public float minDistance = 7.0f;
	[Range(1,200)][Tooltip("The maximum distance the camera can be relative to the player.")]
	public float maxDistance = 40.0f;
}
[Serializable]
public class MSACC_SettingsCameraOrbital {
	[Header("Settings")]
	[Range(0.01f,2.0f)][Tooltip("In this variable you can configure the sensitivity with which the script will perceive the movement of the X and Y inputs. ")]
	public float sensibility = 0.8f;
	[Range(0.01f,2.0f)][Tooltip("In this variable, you can configure the speed at which the orbital camera will approach or distance itself from the player when the mouse scrool is used.")]
	public float speedScrool = 1.0f;
	[Range(0.01f,2.0f)][Tooltip("In this variable, you can configure the speed at which the orbital camera moves up or down.")]
	public float speedYAxis = 0.5f;

	[Header("Limits")]
	[Range(3.0f,20.0f)][Tooltip("In this variable, you can set the minimum distance that the orbital camera can stay from the player.")]
	public float minDistance = 5.0f;
	[Range(20.0f,1000.0f)][Tooltip("In this variable, you can set the maximum distance that the orbital camera can stay from the player.")]
	public float maxDistance = 50.0f;
	[Range(-85,0)][Tooltip("In this variable it is possible to define the minimum angle that the camera can reach on the Y axis")]
	public float minAngleY = 0.0f;
	[Range(0,85)][Tooltip("In this variable it is possible to define the maximum angle that the camera can reach on the Y axis")]
	public float maxAngleY = 80.0f;
	[Tooltip("If this variable is true, the camera ignores the colliders and crosses the walls freely.")]
	public bool ignoreCollision = false;

	[Header("Custom Rotation Input")]
	[Tooltip("If this variable is true, the camera will only rotate when the key selected in the 'KeyToRotate' variable is pressed. If this variable is false, the camera can rotate freely, even without pressing any key.")]
	public bool rotateWhenClick = false;
	[Tooltip("Here you can select the button that must be pressed in order to rotate the camera.")]
	public string keyToRotate = "mouse 0";

	[Space(7)]
	[Tooltip("If this variable is true, the X-axis input will be inverted.")]
	public bool invertXInput = false;
	[Tooltip("If this variable is true, the X-axis input will be inverted.")]
	public bool invertYInput = false;
}
[Serializable]
public class MSACC_SettingsCameraOrbitalThatFollows {
	[Header("Settings(Follow)")]
	[Range(1,20)][Tooltip("The speed at which the camera can follow the player.")]
	public float displacementSpeed = 5.0f;
	[Tooltip("If this variable is true, the code makes a lookAt using quaternions.")]
	public bool customLookAt = false;
	[Range(1,30)][Tooltip("The speed at which the camera rotates as it follows and looks at the player.")]
	public float spinSpeedCustomLookAt = 15.0f;

	[Header("Settings(Orbital)")]
	[Range(0.01f,2.0f)][Tooltip("In this variable you can configure the sensitivity with which the script will perceive the movement of the X and Y inputs. ")]
	public float sensibility = 0.8f;
	[Range(0.01f,2.0f)][Tooltip("In this variable, you can configure the speed at which the orbital camera will approach or distance itself from the player when the mouse scrool is used.")]
	public float speedScrool = 1.0f;
	[Range(0.01f,2.0f)][Tooltip("In this variable, you can configure the speed at which the orbital camera moves up or down.")]
	public float speedYAxis = 0.5f;
	[Range(3.0f,20.0f)][Tooltip("In this variable, you can set the minimum distance that the orbital camera can stay from the player.")]
	public float minDistance = 5.0f;
	[Range(20.0f,1000.0f)][Tooltip("In this variable, you can set the maximum distance that the orbital camera can stay from the player.")]
	public float maxDistance = 50.0f;
	[Range(-85,0)][Tooltip("In this variable it is possible to define the minimum angle that the camera can reach on the Y axis")]
	public float minAngleY = 0.0f;
	[Range(0,85)][Tooltip("In this variable it is possible to define the maximum angle that the camera can reach on the Y axis")]
	public float maxAngleY = 80.0f;
	[Space(7)]
	[Tooltip("If this variable is true, the camera will only rotate when the key selected in the 'KeyToRotate' variable is pressed. If this variable is false, the camera can rotate freely, even without pressing any key.")]
	public bool rotateWhenClick = false;
	[Tooltip("Here you can select the button that must be pressed in order to rotate the camera.")]
	public string keyToRotate = "mouse 0";
	[Tooltip("If this variable is true, the X-axis input will be inverted.")]
	public bool invertXInput = false;
	[Tooltip("If this variable is true, the X-axis input will be inverted.")]
	public bool invertYInput = false;
	//
	public enum ResetTimeType { Time, Input_OnlyWindows }
	[Header("Settings(General)")]
	[Tooltip("In this variable it is possible to define how the control will be redefined for the camera that follows the player, through input or through a time.")]
	public ResetTimeType ResetControlType = ResetTimeType.Time;
	[Tooltip("If 'ResetControlType' is set to 'Input_OnlyWindows', the key that must be pressed to reset the control will be set by this variable.")]
	public KeyCode resetKey = KeyCode.Z;
	[Range(1.0f,50.0f)][Tooltip("If 'ResetControlType' is set to 'Time', the wait time for the camera to reset the control will be set by this variable.")]
	public float timeToReset = 8.0f;
	[Tooltip("If this variable is true, the camera ignores the colliders and crosses the walls freely.")]
	public bool ignoreCollision = false;
}
[Serializable]
public class MSACC_SettingsCameraETS_StyleCamera {
	[Header("Settings")]
	[Range(1,20)][Tooltip("Horizontal camera rotation sensitivity.")]
	public float sensibilityX = 10.0f;
	[Range(1,20)][Tooltip("Vertical camera rotation sensitivity.")]
	public float sensibilityY = 10.0f;
	[Range(0.5f,3.0f)][Tooltip("The distance the camera will move to the left when the mouse is also shifted to the left. This option applies only to cameras that have the 'ETS_StyleCamera' option selected.")]
	public float ETS_CameraShift = 1.0f;
	[Range(0,40)][Tooltip("The maximum the camera can approximate your vision.")]
	public float maxScroolZoom = 30.0f;
	[Range(0,1)][Tooltip("The speed with which the camera can approach your vision through the mouseScrool.")]
	public float speedScroolZoom = 0.5f;

	[Header("Custom Rotation Input")]
	[Tooltip("If this variable is true, the camera will only rotate when the key selected in the 'KeyToRotate' variable is pressed. If this variable is false, the camera can rotate freely, even without pressing any key.")]
	public bool rotateWhenClick = false;
	[Tooltip("Here you can select the button that must be pressed in order to rotate the camera.")]
	public string keyToRotate = "mouse 0";
	[Space(7)]
	[Tooltip("If this variable is true, the X-axis input will be inverted.")]
	public bool invertXInput = false;
	[Tooltip("If this variable is true, the X-axis input will be inverted.")]
	public bool invertYInput = false;
}
[Serializable]
public class MSACC_SettingsFlyCamera {
	[Header("Inputs")]
	[Tooltip("Here you can configure the 'Horizontal' inputs that should be used to move the camera 'CameraFly'.")]
	public string horizontalMove = "Horizontal";
	[Tooltip("Here you can configure the 'Vertical' inputs that should be used to move the camera 'CameraFly'.")]
	public string verticalMove = "Vertical";
	[Tooltip("Here you can configure the keys that must be pressed to accelerate the movement of the camera 'CameraFly'.")]
	public KeyCode speedKeyCode = KeyCode.LeftShift;
	[Tooltip("Here you can configure the key that must be pressed to move the camera 'CameraFly' up.")]
	public KeyCode moveUp = KeyCode.E;
	[Tooltip("Here you can configure the key that must be pressed to move the camera 'CameraFly' down.")]
	public KeyCode moveDown = KeyCode.Q;
	//
	[Header("Settings")]
	[Range(1,20)][Tooltip("Horizontal camera rotation sensitivity.")]
	public float sensibilityX = 10.0f;
	[Range(1,20)][Tooltip("Vertical camera rotation sensitivity.")]
	public float sensibilityY = 10.0f;
	[Range(1,100)][Tooltip("The speed of movement of this camera.")]
	public float movementSpeed = 20.0f;
	[Space(7)]
	[Tooltip("If this variable is true, the X-axis input will be inverted.")]
	public bool invertXInput = false;
	[Tooltip("If this variable is true, the X-axis input will be inverted.")]
	public bool invertYInput = false;
}
[Serializable]
public class MSACC_SettingsCameraPlaneX_Z {

    [Header("Limits")]
    [Tooltip("The smallest position on the X axis that the camera can reach.")]
    public float minXPosition = -100;
    [Tooltip("The largest position on the X axis that the camera can reach.")]
    public float maxXPosition = 100;
    [Space(5)]
    [Tooltip("The smallest position on the Z axis that the camera can reach.")]
    public float minZPosition = -100;
    [Tooltip("The largest position on the Z axis that the camera can reach.")]
    public float maxZPosition = 100;

    [Header("Camera Height")]
    [Tooltip("The normal position of the camera on the Y axis, when the player is far from the edges of the scene.")]
    public float normalYPosition = 40;
    [Tooltip("The lowest position the camera can reach on the Y axis when the player is close to an edge.")]
    public float limitYPosition = 22;
    [Tooltip("The distance of the player in relation to any edge of movement of the camera, so that the camera begins to descend in relation to the ground.")]
    public float edgeDistanceToStartDescending = 50;

    [Header("Movement")]
    [Range(0.5f, 20)]
    [Tooltip("The speed at which the camera can follow the player.")]
    public float displacementSpeed = 2.0f;

    

    public enum PlaneXZRotationType {
        KeepTheRotationFixed, LookAt, OptimizedLookAt
    }
    [Header("Rotation")]
    [Tooltip("Here it is possible to define the type of rotation that the camera will have.")]
    public PlaneXZRotationType SelectRotation = PlaneXZRotationType.OptimizedLookAt;
    [Range(0.1f, 20)]
    [Tooltip("The speed at which the camera rotates as it follows and looks at the player. This variable only has an effect on the 'Optimized LookAt' option.")]
    public float spinSpeedCustomLookAt = 1.0f;
}


public class MSCameraController : MonoBehaviour {

	[Tooltip("Here you must associate the object that the cameras will follow. If you leave this variable empty, the cameras will follow the object in which this script was placed.")]
	public Transform target;
    [Tooltip("In this variable, it is possible to define which will be the first camera used by the player, in case several cameras are being used.")]
    public int startCameraIndex = 0;

	[Space(7)][Tooltip("Here you must associate all the cameras that you want to control by this script, associating each one with an index and selecting your preferences.")]
	public MSACC_CameraType[] cameras = new MSACC_CameraType[0];
	[Tooltip("Here you can configure the cameras, deciding their speed of movement, rotation, zoom, among other options.")]
	public MSACC_CameraSetting cameraSettings;

	bool orbitalAtiv;
	bool orbital_AtivTemp;
	float rotacX = 0.0f;
	float rotacY = 0.0f;
	float tempoOrbit = 0.0f;
	float rotacXETS = 0.0f;
	float rotacYETS = 0.0f;

	Vector2 cameraRotationFly;
	bool changeCam;

	GameObject[] objPosicStopCameras;
	Quaternion[] originalRotation;
	GameObject[] originalPosition;
	Vector3[] originalPositionETS;
	float[] xOrbit;
	float[] yOrbit;
	float[] distanceFromOrbitalCamera;
	float[] initialFieldOfView;
	float[] camFollowPlayerDistance;

	int index = 0;
	int lastIndex = 0;

	Transform targetTransform;
	GameObject playerCamsObj;


	//global inputs
	[HideInInspector]
	public float _horizontalInputMSACC;
	[HideInInspector]
	public float _verticalInputMSACC;
	[HideInInspector]
	public float _scrollInputMSACC;
	[HideInInspector]
	public bool _enableMobileInputs;
	[HideInInspector]
	public int _mobileInputsIndex; // 0 = off,   1 = all,   2 = scroll buttons only

	void OnValidate (){
        //clamp CameraIndex
        startCameraIndex = Mathf.Clamp(startCameraIndex, 0, (cameras.Length - 1));

        //clamp volume
        if (cameras != null) {
            for (int x = 0; x < cameras.Length; x++) {
                if (cameras[x].volume == 0) {
                    cameras[x].volume = 1;
                }
            }
        }

        //clamp limits
        if (cameraSettings != null) {
            if (cameraSettings.PlaneX_Z != null) {
                cameraSettings.PlaneX_Z.minXPosition = Mathf.Clamp(cameraSettings.PlaneX_Z.minXPosition, -99999, cameraSettings.PlaneX_Z.maxXPosition -10);
                cameraSettings.PlaneX_Z.maxXPosition = Mathf.Clamp(cameraSettings.PlaneX_Z.maxXPosition, cameraSettings.PlaneX_Z.minXPosition +10, +99999);
                //
                cameraSettings.PlaneX_Z.minZPosition = Mathf.Clamp(cameraSettings.PlaneX_Z.minZPosition, -99999, cameraSettings.PlaneX_Z.maxZPosition -10);
                cameraSettings.PlaneX_Z.maxZPosition = Mathf.Clamp(cameraSettings.PlaneX_Z.maxZPosition, cameraSettings.PlaneX_Z.minZPosition +10, +99999);
                //
                cameraSettings.PlaneX_Z.normalYPosition = Mathf.Clamp(cameraSettings.PlaneX_Z.normalYPosition, cameraSettings.PlaneX_Z.limitYPosition, +99999);
                cameraSettings.PlaneX_Z.limitYPosition = Mathf.Clamp(cameraSettings.PlaneX_Z.limitYPosition, -99999, cameraSettings.PlaneX_Z.normalYPosition);
                //
                float maxDistance = 99999;
                float minDistX = (cameraSettings.PlaneX_Z.maxXPosition - cameraSettings.PlaneX_Z.minXPosition) * 0.25f;
                float minDistZ = (cameraSettings.PlaneX_Z.maxZPosition - cameraSettings.PlaneX_Z.minZPosition) * 0.25f;
                if ((minDistX) < maxDistance) {
                    maxDistance = minDistX;
                }
                if ((minDistZ) < maxDistance) {
                    maxDistance = minDistZ;
                }
                cameraSettings.PlaneX_Z.edgeDistanceToStartDescending = Mathf.Clamp(cameraSettings.PlaneX_Z.edgeDistanceToStartDescending, 1, maxDistance);
            }
            //
            if (cameraSettings.followPlayer != null) {
                cameraSettings.followPlayer.minDistance = Mathf.Clamp(cameraSettings.followPlayer.minDistance, 1, cameraSettings.followPlayer.maxDistance);
                cameraSettings.followPlayer.maxDistance = Mathf.Clamp(cameraSettings.followPlayer.maxDistance, cameraSettings.followPlayer.minDistance, 200);
            }
        }
	}

    private void OnDrawGizmosSelected() {
        bool drawCamPlaneX_Y = false;
        if (cameras != null) {
            for (int x = 0; x < cameras.Length; x++) {
                if (cameras[x].rotationType == MSACC_CameraType.TipoRotac.PlaneX_Z) {
                    drawCamPlaneX_Y = true;
                }                
            }
        }
        if (drawCamPlaneX_Y) {
            Gizmos.color = Color.red;
            Vector3 bottomLeftCorner = new Vector3(cameraSettings.PlaneX_Z.minXPosition, cameraSettings.PlaneX_Z.limitYPosition, cameraSettings.PlaneX_Z.minZPosition);
            Vector3 bottomRightCorner = new Vector3(cameraSettings.PlaneX_Z.maxXPosition, cameraSettings.PlaneX_Z.limitYPosition, cameraSettings.PlaneX_Z.minZPosition);
            Vector3 topLeftCorner = new Vector3(cameraSettings.PlaneX_Z.minXPosition, cameraSettings.PlaneX_Z.limitYPosition, cameraSettings.PlaneX_Z.maxZPosition);
            Vector3 topRightCorner = new Vector3(cameraSettings.PlaneX_Z.maxXPosition, cameraSettings.PlaneX_Z.limitYPosition, cameraSettings.PlaneX_Z.maxZPosition);
            Gizmos.DrawLine(bottomLeftCorner, bottomRightCorner);
            Gizmos.DrawLine(bottomLeftCorner, topLeftCorner);
            Gizmos.DrawLine(topLeftCorner, topRightCorner);
            Gizmos.DrawLine(topRightCorner, bottomRightCorner);
            //
            Gizmos.color = Color.green;
            float limit = cameraSettings.PlaneX_Z.edgeDistanceToStartDescending;
            Vector3 bottomLeftCornerUP = new Vector3(cameraSettings.PlaneX_Z.minXPosition + limit, cameraSettings.PlaneX_Z.normalYPosition, cameraSettings.PlaneX_Z.minZPosition + limit);
            Vector3 bottomRightCornerUP = new Vector3(cameraSettings.PlaneX_Z.maxXPosition - limit, cameraSettings.PlaneX_Z.normalYPosition, cameraSettings.PlaneX_Z.minZPosition + limit);
            Vector3 topLeftCornerUP = new Vector3(cameraSettings.PlaneX_Z.minXPosition + limit, cameraSettings.PlaneX_Z.normalYPosition, cameraSettings.PlaneX_Z.maxZPosition - limit);
            Vector3 topRightCornerUP = new Vector3(cameraSettings.PlaneX_Z.maxXPosition - limit, cameraSettings.PlaneX_Z.normalYPosition, cameraSettings.PlaneX_Z.maxZPosition - limit);
            Gizmos.DrawLine(bottomLeftCornerUP, bottomRightCornerUP);
            Gizmos.DrawLine(bottomLeftCornerUP, topLeftCornerUP);
            Gizmos.DrawLine(topLeftCornerUP, topRightCornerUP);
            Gizmos.DrawLine(topRightCornerUP, bottomRightCornerUP);
            //
            Gizmos.color = Color.red;
            Gizmos.DrawLine(bottomLeftCorner, bottomLeftCornerUP);
            Gizmos.DrawLine(bottomRightCorner, bottomRightCornerUP);
            Gizmos.DrawLine(topLeftCorner, topLeftCornerUP);
            Gizmos.DrawLine(topRightCorner, topRightCornerUP);
        }
    }

    void Awake(){

		if (target) {
			targetTransform = target;
		} else {
			targetTransform = transform;
		}

		GameObject temp = new GameObject ("PlayerCams");
		temp.transform.parent = targetTransform;
		objPosicStopCameras = new GameObject[cameras.Length];
		originalRotation = new Quaternion[cameras.Length];
		originalPosition = new GameObject[cameras.Length];
		originalPositionETS = new Vector3[cameras.Length];
        xOrbit = new float[cameras.Length];
		yOrbit = new float[cameras.Length];

		distanceFromOrbitalCamera = new float[cameras.Length];
		initialFieldOfView = new float[cameras.Length];
		camFollowPlayerDistance = new float[cameras.Length];

		changeCam = false;
		orbitalAtiv = false;
		orbital_AtivTemp = false;

		for (int x = 0; x < cameras.Length; x++) {
			if (cameras [x]._camera) {
				if (cameras [x].volume == 0) {
					cameras [x].volume = 1;
				}
				initialFieldOfView [x] = cameras [x]._camera.fieldOfView;

				if (cameras [x].rotationType == MSACC_CameraType.TipoRotac.FirstPerson) {
					cameras [x]._camera.transform.parent = temp.transform;
					originalRotation [x] = cameras [x]._camera.transform.localRotation;
				}


				if (cameras [x].rotationType == MSACC_CameraType.TipoRotac.FollowPlayer) {
					cameras [x]._camera.transform.parent = temp.transform;
					originalPosition [x] = new GameObject ("positionFollowPlayerCamera" + x);
					originalPosition [x].transform.parent = temp.transform;
					originalPosition [x].transform.position = cameras [x]._camera.transform.position;
					if (cameraSettings.ajustTheLayers) {
						targetTransform.gameObject.layer = 2;
						foreach (Transform trans in targetTransform.gameObject.GetComponentsInChildren<Transform>(true)) {
							trans.gameObject.layer = 2;
						}
					}
					camFollowPlayerDistance [x] = Vector3.Distance (cameras [x]._camera.transform.position, targetTransform.position);
				}

				if (cameras [x].rotationType == MSACC_CameraType.TipoRotac.Orbital) {
					cameras [x]._camera.transform.parent = temp.transform;
					cameras [x]._camera.transform.LookAt (target);
					xOrbit [x] = cameras [x]._camera.transform.eulerAngles.y;
					yOrbit [x] = cameras [x]._camera.transform.eulerAngles.x;
					if (cameraSettings.ajustTheLayers) {
						targetTransform.gameObject.layer = 2;
						foreach (Transform trans in targetTransform.gameObject.GetComponentsInChildren<Transform>(true)) {
							trans.gameObject.layer = 2;
						}
					}
				}
				distanceFromOrbitalCamera [x] = Vector3.Distance (cameras [x]._camera.transform.position, targetTransform.position);
				distanceFromOrbitalCamera [x] = Mathf.Clamp (distanceFromOrbitalCamera [x], cameraSettings.orbital.minDistance, cameraSettings.orbital.maxDistance);

				if (cameras [x].rotationType == MSACC_CameraType.TipoRotac.Stop) {
					cameras [x]._camera.transform.parent = temp.transform;
				}

				if (cameras [x].rotationType == MSACC_CameraType.TipoRotac.StraightStop) {
					cameras [x]._camera.transform.parent = temp.transform;
					objPosicStopCameras [x] = new GameObject ("positionStraightStopCamera" + x);
					objPosicStopCameras [x].transform.parent = cameras [x]._camera.transform;
					objPosicStopCameras [x].transform.localPosition = new Vector3 (0, 0, 1.0f);
					objPosicStopCameras [x].transform.parent = temp.transform;
				}

				if (cameras [x].rotationType == MSACC_CameraType.TipoRotac.OrbitalThatFollows) {
					cameras [x]._camera.transform.parent = temp.transform;
					xOrbit [x] = cameras [x]._camera.transform.eulerAngles.x;
					yOrbit [x] = cameras [x]._camera.transform.eulerAngles.y;

					originalPosition [x] = new GameObject ("positionCameraFollowPlayer" + x);
					originalPosition [x].transform.parent = temp.transform;
					originalPosition [x].transform.position = cameras [x]._camera.transform.position;

					if (cameraSettings.ajustTheLayers) {
						targetTransform.gameObject.layer = 2;
						foreach (Transform trans in targetTransform.gameObject.GetComponentsInChildren<Transform>(true)) {
							trans.gameObject.layer = 2;
						}
					}
				}

				if (cameras [x].rotationType == MSACC_CameraType.TipoRotac.ETS_StyleCamera) {
					cameras [x]._camera.transform.parent = temp.transform;
					originalRotation [x] = cameras [x]._camera.transform.localRotation;
					originalPositionETS [x] = cameras [x]._camera.transform.localPosition;
				}

				if (cameras [x].rotationType == MSACC_CameraType.TipoRotac.FlyCamera_OnlyWindows) {
					cameras [x]._camera.transform.parent = null;
				}

				if (cameras [x].rotationType == MSACC_CameraType.TipoRotac.LookAtThePlayer) {
					cameras [x]._camera.transform.parent = null;
				}

                if (cameras[x].rotationType == MSACC_CameraType.TipoRotac.PlaneX_Z) {
                    Vector3 newRot = new Vector3(90, cameras[x]._camera.transform.eulerAngles.y, cameras[x]._camera.transform.eulerAngles.z);
                    cameras[x]._camera.transform.rotation = Quaternion.Euler(newRot);
                    cameras[x]._camera.transform.parent = temp.transform;
                    cameras[x]._camera.transform.position = new Vector3(targetTransform.position.x, cameraSettings.PlaneX_Z.normalYPosition, targetTransform.position.z);
                }

				AudioListener audListner = cameras [x]._camera.GetComponent<AudioListener> ();
				if (audListner == null) {
					cameras [x]._camera.transform.gameObject.AddComponent (typeof(AudioListener));
				}

			} else {
				Debug.LogWarning ("There is no camera associated with the index " + x);
			}
		}
		playerCamsObj = temp;
	}

	void Start(){
		index = startCameraIndex;
		lastIndex = startCameraIndex;
		_enableMobileInputs = false;
		EnableCameras (index);
	}

	void EnableCameras (int index){
		if (cameras.Length > 0) {
			changeCam = true;
			for (int x = 0; x < cameras.Length; x++) {
				if (cameras [x]._camera) {
					if (x == index) {
						cameras [x]._camera.gameObject.SetActive (true);
						//
						if (cameras [x].rotationType == MSACC_CameraType.TipoRotac.FirstPerson) {
							rotacX = 0.0f;
							rotacY = 0.0f;
						}
						if (cameras [x].rotationType == MSACC_CameraType.TipoRotac.OrbitalThatFollows) {
							tempoOrbit = 0.0f;
						}
						if (cameras [x].rotationType == MSACC_CameraType.TipoRotac.ETS_StyleCamera) {
							rotacXETS = 0.0f;
							rotacYETS = 0.0f;
						}
						if (cameras [x].rotationType == MSACC_CameraType.TipoRotac.FlyCamera_OnlyWindows) {
							cameras [x]._camera.transform.position = cameras [lastIndex]._camera.transform.position;
							cameraRotationFly = new Vector2(cameras [lastIndex]._camera.transform.eulerAngles.y, 0);
						}
					} else {
						cameras [x]._camera.gameObject.SetActive (false);
					}
				}
			}
		}
	}

	void ManageCameras(){
		if (changeCam) {
			changeCam = false;
			for (int x = 0; x < cameras.Length; x++) {
				if (cameras [x].rotationType == MSACC_CameraType.TipoRotac.FollowPlayer || cameras[x].rotationType == MSACC_CameraType.TipoRotac.PlaneX_Z) {
					if (cameras [x]._camera.isActiveAndEnabled) {
						cameras [x]._camera.transform.parent = null;
					} else {
						cameras [x]._camera.transform.parent = playerCamsObj.transform;
					}
				}
				//
				if (cameras [x].rotationType == MSACC_CameraType.TipoRotac.Orbital || cameras [x].rotationType == MSACC_CameraType.TipoRotac.OrbitalThatFollows) {
					cameras [x]._camera.transform.LookAt (target);
					xOrbit [x] = cameras [x]._camera.transform.eulerAngles.y;
					yOrbit [x] = cameras [x]._camera.transform.eulerAngles.x;
					distanceFromOrbitalCamera [x] = Vector3.Distance (cameras [x]._camera.transform.position, targetTransform.position);
					distanceFromOrbitalCamera [x] = Mathf.Clamp (distanceFromOrbitalCamera [x], cameraSettings.orbital.minDistance, cameraSettings.orbital.maxDistance);
				}
                if (cameras[x].rotationType == MSACC_CameraType.TipoRotac.PlaneX_Z) {
                    float clampXPos = Mathf.Clamp(targetTransform.position.x, cameraSettings.PlaneX_Z.minXPosition, cameraSettings.PlaneX_Z.maxXPosition);
                    float clampZPos = Mathf.Clamp(targetTransform.position.z, cameraSettings.PlaneX_Z.minZPosition, cameraSettings.PlaneX_Z.maxZPosition);
                    Vector3 newPos = new Vector3(clampXPos, cameraSettings.PlaneX_Z.limitYPosition, clampZPos);
                    cameras[x]._camera.transform.position = newPos;
                }
            }

			//set on/off mobile inputs
			if (cameras [index].rotationType == MSACC_CameraType.TipoRotac.Stop || cameras [index].rotationType == MSACC_CameraType.TipoRotac.StraightStop || 
				cameras [index].rotationType == MSACC_CameraType.TipoRotac.LookAtThePlayer || cameras [index].rotationType == MSACC_CameraType.TipoRotac.FlyCamera_OnlyWindows ||
                cameras [index].rotationType == MSACC_CameraType.TipoRotac.PlaneX_Z) {
				_mobileInputsIndex = 0; // 0 = all mobile inputs off
			}
			if (cameras [index].rotationType == MSACC_CameraType.TipoRotac.FirstPerson || cameras [index].rotationType == MSACC_CameraType.TipoRotac.ETS_StyleCamera ||
				cameras [index].rotationType == MSACC_CameraType.TipoRotac.Orbital || cameras [index].rotationType == MSACC_CameraType.TipoRotac.OrbitalThatFollows) {
				_mobileInputsIndex = 1; // 1 = all mobile inputs on
			}
			if (cameras [index].rotationType == MSACC_CameraType.TipoRotac.FollowPlayer) {
                if (cameraSettings.followPlayer.useScrool) {
                    _mobileInputsIndex = 2; // 2 = scroll buttons only
                } else {
                    _mobileInputsIndex = 0; // 0 = all mobile inputs off
                }
			}
		}

		AudioListener.volume = cameras [index].volume;
		float timeScaleSpeed = Mathf.Clamp (1.0f / Time.timeScale, 0.01f, 1);
		switch (cameras[index].rotationType ) {
		    case MSACC_CameraType.TipoRotac.Stop:
			    //stop camera
			    break;
		    case MSACC_CameraType.TipoRotac.StraightStop:
			    Quaternion linearRotation = Quaternion.LookRotation(objPosicStopCameras[index].transform.position - cameras [index]._camera.transform.position, Vector3.up);
			    cameras [index]._camera.transform.rotation = Quaternion.Slerp(cameras [index]._camera.transform.rotation, linearRotation, Time.deltaTime * 15.0f);
			    break;
		    case MSACC_CameraType.TipoRotac.LookAtThePlayer:
			    cameras [index]._camera.transform.LookAt (targetTransform.position);
			    break;
		    case MSACC_CameraType.TipoRotac.FirstPerson:
			    //getInputs
			    float xInput = _horizontalInputMSACC;
			    float yInput = _verticalInputMSACC;
			    if (cameraSettings.firstPerson.invertXInput) {
				    xInput = -_horizontalInputMSACC;
			    }
			    if (cameraSettings.firstPerson.invertYInput) {
				    yInput = -_verticalInputMSACC;
			    }
			    if (cameraSettings.firstPerson.rotateWhenClick) {
				    if (Input.GetKey (cameraSettings.firstPerson.keyToRotate) || _enableMobileInputs) {
					    rotacX += xInput * cameraSettings.firstPerson.sensibilityX;
					    rotacY += yInput * cameraSettings.firstPerson.sensibilityY;
				    }
			    } else {
				    rotacX += xInput * cameraSettings.firstPerson.sensibilityX;
				    rotacY += yInput * cameraSettings.firstPerson.sensibilityY;
			    }
			    //
			    rotacX = MSADCCClampAngle (rotacX, -cameraSettings.firstPerson.horizontalAngle, cameraSettings.firstPerson.horizontalAngle);
			    rotacY = MSADCCClampAngle (rotacY, -cameraSettings.firstPerson.verticalAngle, cameraSettings.firstPerson.verticalAngle);
			    Quaternion xQuaternion = Quaternion.AngleAxis (rotacX, Vector3.up);
			    Quaternion yQuaternion = Quaternion.AngleAxis (rotacY, -Vector3.right);
			    Quaternion _nextRot = originalRotation [index] * xQuaternion * yQuaternion;
			    cameras [index]._camera.transform.localRotation = Quaternion.Lerp (cameras [index]._camera.transform.localRotation, _nextRot, Time.deltaTime * 10.0f * timeScaleSpeed);
			    //fieldOfView
			    cameras [index]._camera.fieldOfView -= _scrollInputMSACC * cameraSettings.firstPerson.speedScroolZoom * 50.0f;
			    if (cameras [index]._camera.fieldOfView < (initialFieldOfView [index] - cameraSettings.firstPerson.maxScroolZoom)) {
				    cameras [index]._camera.fieldOfView = (initialFieldOfView [index] - cameraSettings.firstPerson.maxScroolZoom);
			    }
			    if (cameras [index]._camera.fieldOfView > initialFieldOfView [index]) {
				    cameras [index]._camera.fieldOfView = (initialFieldOfView [index]);
			    }
			    break;
		    case MSACC_CameraType.TipoRotac.FollowPlayer:
			    //move
			    RaycastHit hitCamFollow;
			    if (cameraSettings.followPlayer.useScrool) {
				    float camLerpSpeed = Time.deltaTime * cameraSettings.followPlayer.displacementSpeed * (camFollowPlayerDistance [index] * 0.1f);
				    camFollowPlayerDistance [index] = camFollowPlayerDistance [index] - _scrollInputMSACC * (cameraSettings.followPlayer.scroolSpeed * 50.0f);
				    camFollowPlayerDistance [index] = Mathf.Clamp (camFollowPlayerDistance [index], cameraSettings.followPlayer.minDistance, cameraSettings.followPlayer.maxDistance);
				    Vector3 direction = (targetTransform.position - originalPosition [index].transform.position).normalized;
				    Vector3 finalPos = targetTransform.position - direction * camFollowPlayerDistance [index];
				    //
				    if (!Physics.Linecast (targetTransform.position, finalPos)) {
					    cameras [index]._camera.transform.position = Vector3.Lerp (cameras [index]._camera.transform.position, finalPos, camLerpSpeed);
				    } else if (Physics.Linecast (targetTransform.position, finalPos, out hitCamFollow)) {
					    if (cameraSettings.followPlayer.ignoreCollision) {
						    cameras [index]._camera.transform.position = Vector3.Lerp (cameras [index]._camera.transform.position, finalPos, camLerpSpeed);
					    } else {
						    cameras [index]._camera.transform.position = Vector3.Lerp (cameras [index]._camera.transform.position, hitCamFollow.point, camLerpSpeed);
					    }
				    }
			    } else {
				    if (!Physics.Linecast (targetTransform.position, originalPosition [index].transform.position)) {
					    cameras [index]._camera.transform.position = Vector3.Lerp (cameras [index]._camera.transform.position, originalPosition [index].transform.position, Time.deltaTime * cameraSettings.followPlayer.displacementSpeed);
				    } else if (Physics.Linecast (transform.position, originalPosition [index].transform.position, out hitCamFollow)) {
					    if (cameraSettings.followPlayer.ignoreCollision) {
						    cameras [index]._camera.transform.position = Vector3.Lerp (cameras [index]._camera.transform.position, originalPosition [index].transform.position, Time.deltaTime * cameraSettings.followPlayer.displacementSpeed);
					    } else {
						    cameras [index]._camera.transform.position = Vector3.Lerp (cameras [index]._camera.transform.position, hitCamFollow.point, Time.deltaTime * cameraSettings.followPlayer.displacementSpeed);
					    }
				    }
			    }
			    //rotation
			    if (cameraSettings.followPlayer.customLookAt) {
				    Quaternion nextRotation = Quaternion.LookRotation (targetTransform.position - cameras [index]._camera.transform.position, Vector3.up);
				    cameras [index]._camera.transform.rotation = Quaternion.Slerp (cameras [index]._camera.transform.rotation, nextRotation, Time.deltaTime * cameraSettings.followPlayer.spinSpeedCustomLookAt);
			    } else {
				    cameras [index]._camera.transform.LookAt (targetTransform.position);
			    }
			    break;
		    case MSACC_CameraType.TipoRotac.Orbital:
			    //raycast hit
			    RaycastHit hitCamOrbital;
			    float minDistance = cameraSettings.orbital.minDistance;
			    if (Physics.Linecast (targetTransform.position, cameras [index]._camera.transform.position, out hitCamOrbital)) {
				    if (!cameraSettings.orbital.ignoreCollision) {
					    distanceFromOrbitalCamera [index] = Vector3.Distance (targetTransform.position, hitCamOrbital.point);
					    minDistance = Mathf.Clamp (distanceFromOrbitalCamera [index], minDistance * 0.5f, cameraSettings.orbital.maxDistance);
				    }
			    }
			    //getInputs
			    float xInputOrb = _horizontalInputMSACC;
			    float yInputOrb = _verticalInputMSACC;
			    if (cameraSettings.orbital.invertXInput) {
				    xInputOrb = -_horizontalInputMSACC;
			    }
			    if (cameraSettings.orbital.invertYInput) {
				    yInputOrb = -_verticalInputMSACC;
			    }
			    if (cameraSettings.orbital.rotateWhenClick) {
				    if (Input.GetKey (cameraSettings.orbital.keyToRotate) || _enableMobileInputs) {
					    xOrbit [index] += xInputOrb * (cameraSettings.orbital.sensibility * distanceFromOrbitalCamera [index]) / (distanceFromOrbitalCamera [index] * 0.5f);
					    yOrbit [index] -= yInputOrb * cameraSettings.orbital.sensibility * (cameraSettings.orbital.speedYAxis * 10.0f);
				    }
			    } else {
				    xOrbit [index] += xInputOrb * (cameraSettings.orbital.sensibility * distanceFromOrbitalCamera [index]) / (distanceFromOrbitalCamera [index] * 0.5f);
				    yOrbit [index] -= yInputOrb * cameraSettings.orbital.sensibility * (cameraSettings.orbital.speedYAxis * 10.0f);
			    }
			    //move - rotation
			    yOrbit [index] = MSADCCClampAngle (yOrbit [index], cameraSettings.orbital.minAngleY, cameraSettings.orbital.maxAngleY);
			    Quaternion quatToEuler = Quaternion.Euler (yOrbit [index], xOrbit [index], 0);
			    distanceFromOrbitalCamera [index] = Mathf.Clamp (distanceFromOrbitalCamera [index] - _scrollInputMSACC * (cameraSettings.orbital.speedScrool * 50.0f), minDistance, cameraSettings.orbital.maxDistance);
			    Vector3 zPosition = new Vector3 (0.0f, 0.0f, -distanceFromOrbitalCamera [index]);
			    Vector3 nextPosCam = quatToEuler * zPosition + targetTransform.position;
			    Vector3 currentPosCam = cameras [index]._camera.transform.position;
			    Quaternion camRotation = cameras [index]._camera.transform.rotation;
			    cameras [index]._camera.transform.rotation = Quaternion.Lerp(camRotation, quatToEuler, Time.deltaTime * 5.0f * timeScaleSpeed);
			    cameras [index]._camera.transform.position = Vector3.Lerp(currentPosCam, nextPosCam, Time.deltaTime * 5.0f * timeScaleSpeed);
			    break;
		    case MSACC_CameraType.TipoRotac.OrbitalThatFollows:
			    float movXInput = 0.0f;
			    float movYInput = 0.0f;
			    float movZInput = _scrollInputMSACC;
			    if (cameraSettings.OrbitalThatFollows.rotateWhenClick) {
				    if (Input.GetKey (cameraSettings.OrbitalThatFollows.keyToRotate) || _enableMobileInputs) {
					    movXInput = _horizontalInputMSACC;
					    movYInput = _verticalInputMSACC;
					    if (cameraSettings.OrbitalThatFollows.invertXInput) {
						    movXInput = -_horizontalInputMSACC;
					    }
					    if (cameraSettings.OrbitalThatFollows.invertYInput) {
						    movYInput = -_verticalInputMSACC;
					    }
				    }
			    } else {
				    movXInput = _horizontalInputMSACC;
				    movYInput = _verticalInputMSACC;
				    if (cameraSettings.OrbitalThatFollows.invertXInput) {
					    movXInput = -_horizontalInputMSACC;
				    }
				    if (cameraSettings.OrbitalThatFollows.invertYInput) {
					    movYInput = -_verticalInputMSACC;
				    }
			    }
			    //
			    if (movXInput > 0.0f || movYInput > 0.0f || movZInput > 0.0f) {
				    orbitalAtiv = true;
				    tempoOrbit = 0.0f;
				    if (!orbital_AtivTemp) {
					    orbital_AtivTemp = true;
					    xOrbit [index] = cameras [index]._camera.transform.eulerAngles.y;
					    yOrbit [index] = cameras [index]._camera.transform.eulerAngles.x;
				    }
			    } else {
				    tempoOrbit += Time.deltaTime;
				    if (tempoOrbit > cameraSettings.OrbitalThatFollows.timeToReset) {
					    tempoOrbit = cameraSettings.OrbitalThatFollows.timeToReset + 0.1f;
				    }
			    }
			    //
			    switch (cameraSettings.OrbitalThatFollows.ResetControlType) {
			        case MSACC_SettingsCameraOrbitalThatFollows.ResetTimeType.Time:
				        if (tempoOrbit > cameraSettings.OrbitalThatFollows.timeToReset) {
					        orbitalAtiv = false;
					        orbital_AtivTemp = false;
				        }
				        break;
			        case MSACC_SettingsCameraOrbitalThatFollows.ResetTimeType.Input_OnlyWindows:
				        if (Input.GetKeyDown(cameraSettings.OrbitalThatFollows.resetKey)) {
					        orbitalAtiv = false;
					        orbital_AtivTemp = false;
				        }
				        break;
			    }
			    //
			    RaycastHit hitCamOTS;
			    if(orbitalAtiv == true){
				    float _minDistance = cameraSettings.OrbitalThatFollows.minDistance;
				    if (Physics.Linecast (targetTransform.position, cameras [index]._camera.transform.position, out hitCamOTS)) {
					    if (!cameraSettings.OrbitalThatFollows.ignoreCollision) {
						    distanceFromOrbitalCamera [index] = Vector3.Distance (targetTransform.position, hitCamOTS.point);
						    _minDistance = Mathf.Clamp (distanceFromOrbitalCamera [index], _minDistance * 0.5f, cameraSettings.OrbitalThatFollows.maxDistance);
					    }
				    }
				    xOrbit [index] += movXInput * (cameraSettings.OrbitalThatFollows.sensibility * distanceFromOrbitalCamera [index]) / (distanceFromOrbitalCamera [index] * 0.5f);
				    yOrbit [index] -= movYInput * cameraSettings.OrbitalThatFollows.sensibility * (cameraSettings.OrbitalThatFollows.speedYAxis * 10.0f);
				    yOrbit [index] = MSADCCClampAngle (yOrbit [index], cameraSettings.OrbitalThatFollows.minAngleY, cameraSettings.OrbitalThatFollows.maxAngleY);
				    Quaternion quaterToEuler = Quaternion.Euler (yOrbit [index], xOrbit [index], 0);
				    distanceFromOrbitalCamera [index] = Mathf.Clamp (distanceFromOrbitalCamera [index] - movZInput * (cameraSettings.OrbitalThatFollows.speedScrool * 50.0f), _minDistance, cameraSettings.OrbitalThatFollows.maxDistance);
				    Vector3 _zPosition = new Vector3 (0.0f, 0.0f, -distanceFromOrbitalCamera [index]);
				    Vector3 _camNewPos = quaterToEuler * _zPosition + targetTransform.position;
				    Vector3 _camCurrentPos = cameras [index]._camera.transform.position;
				    Quaternion camRot = cameras [index]._camera.transform.rotation;
				    cameras [index]._camera.transform.rotation = Quaternion.Lerp (camRot, quaterToEuler, Time.deltaTime * 5.0f * timeScaleSpeed);
				    cameras [index]._camera.transform.position = Vector3.Lerp (_camCurrentPos, _camNewPos, Time.deltaTime * 5.0f * timeScaleSpeed);
			    } else {
				    if (!Physics.Linecast (targetTransform.position, originalPosition [index].transform.position)) {
					    cameras [index]._camera.transform.position = Vector3.Lerp (cameras [index]._camera.transform.position, originalPosition [index].transform.position, Time.deltaTime * cameraSettings.OrbitalThatFollows.displacementSpeed);
				    }
				    else if(Physics.Linecast(targetTransform.position, originalPosition [index].transform.position,out hitCamOTS)){
					    if (cameraSettings.OrbitalThatFollows.ignoreCollision) {
						    cameras [index]._camera.transform.position = Vector3.Lerp (cameras [index]._camera.transform.position, originalPosition [index].transform.position, Time.deltaTime * cameraSettings.OrbitalThatFollows.displacementSpeed);
					    } 
					    else {
						    cameras [index]._camera.transform.position = Vector3.Lerp(cameras [index]._camera.transform.position, hitCamOTS.point,Time.deltaTime * cameraSettings.OrbitalThatFollows.displacementSpeed);
					    }
				    }
				    //
				    if (cameraSettings.OrbitalThatFollows.customLookAt) {
					    Quaternion quatLookRot = Quaternion.LookRotation (targetTransform.position - cameras [index]._camera.transform.position, Vector3.up);
					    cameras [index]._camera.transform.rotation = Quaternion.Slerp (cameras [index]._camera.transform.rotation, quatLookRot, Time.deltaTime * cameraSettings.OrbitalThatFollows.spinSpeedCustomLookAt);
				    } else {
					    cameras [index]._camera.transform.LookAt (targetTransform.position);
				    }
			    }
			    break;
		    case MSACC_CameraType.TipoRotac.ETS_StyleCamera:
			    float xInputEts = _horizontalInputMSACC;
			    float yInputEts = _verticalInputMSACC; 
			    if (cameraSettings.ETS_StyleCamera.invertXInput) {
				    xInputEts = -_horizontalInputMSACC;
			    }
			    if (cameraSettings.ETS_StyleCamera.invertYInput) {
				    yInputEts = -_verticalInputMSACC;
			    }
			    if (cameraSettings.ETS_StyleCamera.rotateWhenClick) {
				    if (Input.GetKey (cameraSettings.ETS_StyleCamera.keyToRotate) || _enableMobileInputs) {
					    rotacXETS += xInputEts * cameraSettings.ETS_StyleCamera.sensibilityX;
					    rotacYETS += yInputEts * cameraSettings.ETS_StyleCamera.sensibilityY;
				    }
			    } else {
				    rotacXETS += xInputEts * cameraSettings.ETS_StyleCamera.sensibilityX;
				    rotacYETS += yInputEts * cameraSettings.ETS_StyleCamera.sensibilityY;
			    }
			    Vector3 newPositionETS = new Vector3 (originalPositionETS [index].x + Mathf.Clamp (rotacXETS / 50 + (cameraSettings.ETS_StyleCamera.ETS_CameraShift/3.0f), -cameraSettings.ETS_StyleCamera.ETS_CameraShift, 0), originalPositionETS [index].y, originalPositionETS [index].z);
			    cameras [index]._camera.transform.localPosition = Vector3.Lerp (cameras [index]._camera.transform.localPosition, newPositionETS, Time.deltaTime * 10.0f);
			    rotacXETS = MSADCCClampAngle (rotacXETS, -180, 80);
			    rotacYETS = MSADCCClampAngle (rotacYETS, -60, 60);
			    Quaternion _xQuaternion = Quaternion.AngleAxis (rotacXETS, Vector3.up);
			    Quaternion _yQuaternion = Quaternion.AngleAxis (rotacYETS, -Vector3.right);
			    Quaternion nextRot = originalRotation [index] * _xQuaternion * _yQuaternion;
			    cameras [index]._camera.transform.localRotation = Quaternion.Lerp (cameras [index]._camera.transform.localRotation, nextRot, Time.deltaTime * 10.0f * timeScaleSpeed);
			    //fieldOfView
			    cameras [index]._camera.fieldOfView -= _scrollInputMSACC * cameraSettings.ETS_StyleCamera.speedScroolZoom * 50.0f;
			    if (cameras [index]._camera.fieldOfView < (initialFieldOfView [index] - cameraSettings.ETS_StyleCamera.maxScroolZoom)) {
				    cameras [index]._camera.fieldOfView = (initialFieldOfView [index] - cameraSettings.ETS_StyleCamera.maxScroolZoom);
			    }
			    if (cameras [index]._camera.fieldOfView > initialFieldOfView [index]) {
				    cameras [index]._camera.fieldOfView = (initialFieldOfView [index]);
			    }
			    break;
		    case MSACC_CameraType.TipoRotac.FlyCamera_OnlyWindows:
			    float xInputFly = _horizontalInputMSACC;
			    float yInputFly = _verticalInputMSACC; 
			    if (cameraSettings.FlyCamera_OnlyWindows.invertXInput) {
				    xInputFly = -_horizontalInputMSACC;
			    }
			    if (cameraSettings.FlyCamera_OnlyWindows.invertYInput) {
				    yInputFly = -_verticalInputMSACC;
			    }
			    //
			    cameraRotationFly.x += xInputFly * cameraSettings.FlyCamera_OnlyWindows.sensibilityX * 15 * Time.deltaTime;
			    cameraRotationFly.y += yInputFly * cameraSettings.FlyCamera_OnlyWindows.sensibilityY * 15 * Time.deltaTime;
			    cameraRotationFly.y = Mathf.Clamp (cameraRotationFly.y, -90, 90);
			    cameras [index]._camera.transform.rotation = Quaternion.AngleAxis (cameraRotationFly.x, Vector3.up);
			    cameras [index]._camera.transform.rotation *= Quaternion.AngleAxis (cameraRotationFly.y, Vector3.left);
			    //
			    float speedCamFly = cameraSettings.FlyCamera_OnlyWindows.movementSpeed;
			    if (Input.GetKey (cameraSettings.FlyCamera_OnlyWindows.speedKeyCode)) {
				    speedCamFly *= 3.0f;
			    }
			    cameras [index]._camera.transform.position += cameras [index]._camera.transform.right * speedCamFly * Input.GetAxis(cameraSettings.FlyCamera_OnlyWindows.horizontalMove) * Time.deltaTime;
			    cameras [index]._camera.transform.position += cameras [index]._camera.transform.forward * speedCamFly * Input.GetAxis(cameraSettings.FlyCamera_OnlyWindows.verticalMove) * Time.deltaTime;
			    //
			    if(Input.GetKey(cameraSettings.FlyCamera_OnlyWindows.moveUp)){
				    cameras [index]._camera.transform.position += Vector3.up * speedCamFly * Time.deltaTime;
			    }
			    if(Input.GetKey(cameraSettings.FlyCamera_OnlyWindows.moveDown)){
				    cameras [index]._camera.transform.position -= Vector3.up * speedCamFly * Time.deltaTime;
			    }
			    break;
            case MSACC_CameraType.TipoRotac.PlaneX_Z:
                // Get parameters
                float xHeight = cameraSettings.PlaneX_Z.normalYPosition;
                float ZHeight = cameraSettings.PlaneX_Z.normalYPosition;
                float rateOfChange = ((cameraSettings.PlaneX_Z.normalYPosition - cameraSettings.PlaneX_Z.limitYPosition) / cameraSettings.PlaneX_Z.edgeDistanceToStartDescending);
                float clampXPos = Mathf.Clamp(targetTransform.position.x, cameraSettings.PlaneX_Z.minXPosition, cameraSettings.PlaneX_Z.maxXPosition);
                float clampZPos = Mathf.Clamp(targetTransform.position.z, cameraSettings.PlaneX_Z.minZPosition, cameraSettings.PlaneX_Z.maxZPosition);
                Vector3 newPos = new Vector3(clampXPos, cameraSettings.PlaneX_Z.normalYPosition, clampZPos);
                
                //discover xHeight
                if (newPos.x < (cameraSettings.PlaneX_Z.minXPosition + cameraSettings.PlaneX_Z.edgeDistanceToStartDescending)) {
                    xHeight = cameraSettings.PlaneX_Z.limitYPosition + (newPos.x - cameraSettings.PlaneX_Z.minXPosition) * rateOfChange;
                }
                if (newPos.x > (cameraSettings.PlaneX_Z.maxXPosition - cameraSettings.PlaneX_Z.edgeDistanceToStartDescending)) {
                    xHeight = cameraSettings.PlaneX_Z.limitYPosition + (cameraSettings.PlaneX_Z.maxXPosition - newPos.x) * rateOfChange;
                }

                //discover yHeight
                if (newPos.z < (cameraSettings.PlaneX_Z.minZPosition + cameraSettings.PlaneX_Z.edgeDistanceToStartDescending)) {
                    ZHeight = cameraSettings.PlaneX_Z.limitYPosition + (newPos.z - cameraSettings.PlaneX_Z.minZPosition) * rateOfChange;
                }
                if (newPos.z > (cameraSettings.PlaneX_Z.maxZPosition - cameraSettings.PlaneX_Z.edgeDistanceToStartDescending)) {
                    ZHeight = cameraSettings.PlaneX_Z.limitYPosition + (cameraSettings.PlaneX_Z.maxZPosition - newPos.z) * rateOfChange;
                }

                //finally, get yHeight and Apply new position
                float minHeight = Mathf.Min(xHeight, ZHeight, newPos.y);
                newPos = new Vector3(clampXPos, minHeight, clampZPos);
                cameras[index]._camera.transform.position = Vector3.Lerp(cameras[index]._camera.transform.position, newPos, Time.deltaTime * cameraSettings.PlaneX_Z.displacementSpeed);

                //rotation
                switch (cameraSettings.PlaneX_Z.SelectRotation) {
                    case MSACC_SettingsCameraPlaneX_Z.PlaneXZRotationType.KeepTheRotationFixed:
                        Vector3 newRot = new Vector3(90, 0, 0);
                        cameras[index]._camera.transform.rotation = Quaternion.Euler(newRot);
                        break;
                    case MSACC_SettingsCameraPlaneX_Z.PlaneXZRotationType.LookAt:
                        cameras[index]._camera.transform.LookAt(targetTransform.position);
                        break;
                    case MSACC_SettingsCameraPlaneX_Z.PlaneXZRotationType.OptimizedLookAt:
                        Quaternion nextRotation = Quaternion.LookRotation(targetTransform.position - cameras[index]._camera.transform.position, Vector3.up);
                        cameras[index]._camera.transform.rotation = Quaternion.Slerp(cameras[index]._camera.transform.rotation, nextRotation, Time.deltaTime * cameraSettings.PlaneX_Z.spinSpeedCustomLookAt);
                        break;
                }
                break;
		}
	}

	public static float MSADCCClampAngle (float angle, float min, float max){
		if (angle < -360F) { angle += 360F; }
		if (angle > 360F) { angle -= 360F; }
		return Mathf.Clamp (angle, min, max);
	}
	 
	public void MSADCCChangeCameras(){ //use this void to change cameras using buttons
		if (Time.timeScale > 0) {
			if (index < (cameras.Length - 1)) {
				lastIndex = index;
				index++;
				EnableCameras (index);
			} else if (index >= (cameras.Length - 1)) {
				lastIndex = index;
				index = 0;
				EnableCameras (index);
			}
		}
	}

	void Update(){
		
		// Get inputs
		if (!_enableMobileInputs) {
			_horizontalInputMSACC = Input.GetAxis (cameraSettings.inputMouseX);
			_verticalInputMSACC = Input.GetAxis (cameraSettings.inputMouseY);
			_scrollInputMSACC = Input.GetAxis (cameraSettings.inputMouseScrollWheel);
		} else {
			//GetComponent<MSCameraController>()._horizontalInputMSACC = InputX;
			//GetComponent<MSCameraController>()._verticalInputMSACC = InputY;
			//GetComponent<MSCameraController>()._scrollInputMSACC = InputScroll;
		}
		_horizontalInputMSACC = Mathf.Clamp (_horizontalInputMSACC, -1, 1);
		_verticalInputMSACC = Mathf.Clamp (_verticalInputMSACC, -1, 1);
		_scrollInputMSACC = Mathf.Clamp (_scrollInputMSACC, -1, 1);
		//


		//camera switch key
		if (!_enableMobileInputs) {
			if (Time.timeScale > 0) {
				if (Input.GetKeyDown (cameraSettings.cameraSwitchKey) && index < (cameras.Length - 1)) {
					lastIndex = index;
					index++;
					EnableCameras (index);
				} else if (Input.GetKeyDown (cameraSettings.cameraSwitchKey) && index >= (cameras.Length - 1)) {
					lastIndex = index;
					index = 0;
					EnableCameras (index);
				}
			}
		}


		//update cameras
		if (cameraSettings.camerasUpdateMode == MSACC_CameraSetting.UpdateMode.Update) {
			if (cameras.Length > 0 && Time.timeScale > 0) {
				if (cameras [index]._camera) {
					ManageCameras ();
				}
			}
		}
	}

	void LateUpdate(){
		//update cameras
		if (cameraSettings.camerasUpdateMode == MSACC_CameraSetting.UpdateMode.LateUpdate) {
			if (cameras.Length > 0 && Time.timeScale > 0) {
				if (cameras [index]._camera) {
					ManageCameras ();
				}
			}
		}
	}

	void FixedUpdate(){
		//update cameras
		if (cameraSettings.camerasUpdateMode == MSACC_CameraSetting.UpdateMode.FixedUpdate) {
			if (cameras.Length > 0 && Time.timeScale > 0) {
				if (cameras [index]._camera) {
					ManageCameras ();
				}
			}
		}
	}
}