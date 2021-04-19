using System.Collections.Generic;
using System.Collections;
using System.IO;
using UnityEngine.UI;
using UnityEngine;

public class ScreenCapturer : MonoBehaviour
{
    [Header("UI Elements")]
    public GameObject weatherDropdown;
    public GameObject rainSlider;
    public GameObject leftTurnButton;
    public GameObject rightTurnButton;
    public GameObject lowBeamButton;
    public GameObject longBeamButton;
    public GameObject resetButton;

    [Header("Stuff to be deactivated")]
    public List<Camera> cameras = new List<Camera>();
    public GameObject visualDashboardUI;
    public GameObject cameraParentObject;

    [Header("Objects for generating the dataset")]
    public Camera cameraSensor;
    public GameObject pinkSigns;
    public GameObject normalSigns;

    private float timeStamp;
    private int secondStamp;
    private int captureIdx;
    private Texture2D screenShot;
    private Texture2D scaledScreenShot;
    private int weatherIdx;
    public static float rainIntensity;
    public GameObject carLights;
    public GameObject sun;

    // Start is called before the first frame update
    private void Start()
    {
        if(UI.widthValue == "")
        {
            UI.widthValue = "1920";
        }
        if(UI.heightValue == "")
        {
            UI.heightValue = "1080";
        }
        if(UI.captureRate == "")
        {
            UI.captureRate = "1";
        }

        weatherIdx = 0;
        rainIntensity = 0f;

        // Generate the directory for storing the dataset
        if (!Directory.Exists(UI.storeDatasetPath + "/Sample Data"))
        {
            Directory.CreateDirectory(UI.storeDatasetPath + "/Sample Data");
        }
        if (!Directory.Exists(UI.storeDatasetPath + "/Target Data"))
        {
            Directory.CreateDirectory(UI.storeDatasetPath + "/Target Data");
        }

        // Deactivate all active cameras
        foreach (Camera camera in cameras)
        {
            camera.gameObject.SetActive(false);
        }
        // Activate the camera responsible for capturing the dataset images
        cameras[4].gameObject.SetActive(true);

        // Deactivate visual UI and the camera controll script
        visualDashboardUI.GetComponent<Canvas>().enabled = false;
        cameraParentObject.GetComponent<MSCameraController>().enabled = false;

        timeStamp = Time.time; // Used to track the time, when to take a new screenshot
        secondStamp = -1; // Used to track the time, between the sample and target screenshot
        captureIdx = 0;
        screenShot = new Texture2D(Screen.width, Screen.height, TextureFormat.RGB24, false);
    }

    // Update is called once per frame
    private void Update()
    {
        getUserInput();
        rainSlider.GetComponent<Slider>().value = 0.5f;

        if (Time.time - timeStamp > float.Parse(UI.captureRate) || 
            (secondStamp != -1 && Mathf.Abs(System.DateTime.Now.Second - secondStamp) > 1))
        {
            if (Time.timeScale == 1)
            {
                // Stops the game to make the screenshots
                Time.timeScale = 0.0f;
                StartCoroutine(TakeScreenshotTarget());
            }
            else
            {
                // Resume the game
                Time.timeScale = 1;
            }

            weatherDropdown.GetComponent<Dropdown>().value = weatherIdx;
            // Reactivate normal signs
            QualitySettings.shadows = ShadowQuality.All;
            pinkSigns.SetActive(false);
            normalSigns.SetActive(true);
            carLights.SetActive(true);

            timeStamp = Time.time;

            if (secondStamp == -1)
            {
                secondStamp = System.DateTime.Now.Second;
            }
            else
            {
                secondStamp = -1;
            }
        }
    }

    private IEnumerator TakeScreenshotSample()
    {
        rainSlider.GetComponent<Slider>().value = 0.5f;
        yield return new WaitForEndOfFrame();
        //rainSlider.GetComponent<Slider>().value = 0.5f;
        screenShot.ReadPixels(new Rect(0, 0, Screen.width, Screen.height), 0, 0);
        screenShot.Apply();
        scaledScreenShot = ScaleTexture(screenShot, int.Parse(UI.widthValue), int.Parse(UI.heightValue));
        byte[] bytes = scaledScreenShot.EncodeToPNG();
        int index = captureIdx + 1;
        string fileName = UI.storeDatasetPath + "/Sample Data/sample" + captureIdx.ToString() + ".png";
        System.IO.File.WriteAllBytes(fileName, bytes);
        //rainSlider.GetComponent<Slider>().value = 0.5f;
    }

    IEnumerator TakeScreenshotTarget()
    {
        yield return new WaitForEndOfFrame();
        //rainSlider.GetComponent<Slider>().value = 0.5f;
        yield return TakeScreenshotSample();
        weatherDropdown.GetComponent<Dropdown>().value = 0;
        // Activate target signs
        pinkSigns.SetActive(true);
        normalSigns.SetActive(false);
        QualitySettings.shadows = ShadowQuality.Disable;
        carLights.SetActive(false);
        sun.GetComponent<Light>().intensity = 5;
        //cameras[4].Render();
        yield return new WaitForEndOfFrame();
        ///cameras[4].Render();
        // Activate target signs
        pinkSigns.SetActive(true);
        normalSigns.SetActive(false);

        screenShot.ReadPixels(new Rect(0, 0, Screen.width, Screen.height), 0, 0);
        screenShot.Apply();
        scaledScreenShot = ScaleTexture(screenShot, int.Parse(UI.widthValue), int.Parse(UI.heightValue));
        byte[] bytes = scaledScreenShot.EncodeToPNG();
        int index = captureIdx + 1;
        string fileName = UI.storeDatasetPath + "/Target Data/sample" + captureIdx.ToString() + ".png";
        System.IO.File.WriteAllBytes(fileName, bytes);
        captureIdx++;
        //rainSlider.GetComponent<Slider>().value = 0.5f;
    }

    /// <summary>
    /// Scales the image to a specific resolution by using Bilinear interpolation.
    /// </summary>
    /// <param name="source">Image which has to be scaled</param>
    /// <param name="targetWidth">New Width value for the image</param>
    /// <param name="targetHeight">New Height value for the image</param>
    /// <returns></returns>
    private Texture2D ScaleTexture(Texture2D source, int targetWidth, int targetHeight)
    {
        Texture2D result = new Texture2D(targetWidth, targetHeight, source.format, true);
        UnityEngine.Color[] rpixels = result.GetPixels(0);
        float incX = ((float)1 / source.width) * ((float)source.width / targetWidth);
        float incY = ((float)1 / source.height) * ((float)source.height / targetHeight);
        for (int px = 0; px < rpixels.Length; px++)
        {
            rpixels[px] = source.GetPixelBilinear(incX * ((float)px % targetWidth),
                              incY * ((float)Mathf.Floor(px / targetWidth)));
        }
        result.SetPixels(rpixels, 0);
        result.Apply();
        scaledScreenShot = result;
        UnityEngine.Object.Destroy(result);
        return scaledScreenShot;
    }

    /// <summary>
    /// Controlls the visul UI via keyboard input, since the UI will not be visible.
    /// </summary>
    private void getUserInput()
    {
        // Checks if any key has beenn pressed
        if (Input.anyKeyDown)
        {
            if (Input.GetKeyDown(KeyCode.Alpha1))
            {
                // Sunny weather
                weatherDropdown.GetComponent<Dropdown>().value = 0;
                weatherIdx = 0;
            }
            else if (Input.GetKeyDown(KeyCode.Alpha2))
            {
                // Rainy weather
                weatherDropdown.GetComponent<Dropdown>().value = 1;
                weatherIdx = 1;
            }
            else if (Input.GetKeyDown(KeyCode.Alpha3))
            {
                // Sunset
                weatherDropdown.GetComponent<Dropdown>().value = 2;
                weatherIdx = 2;
            }
            else if (Input.GetKeyDown(KeyCode.Alpha4))
            {
                // Bright night 
                weatherDropdown.GetComponent<Dropdown>().value = 3;
                weatherIdx = 3;
            }
            else if (Input.GetKeyDown(KeyCode.Alpha5))
            {
                // Dark night
                weatherDropdown.GetComponent<Dropdown>().value = 4;
                weatherIdx = 4;
            }
            else if (Input.GetKeyDown(KeyCode.KeypadPlus))
            {
                // Increases the rain intensity on the slider
                rainSlider.GetComponent<Slider>().value += 0.1f;
                rainIntensity = rainSlider.GetComponent<Slider>().value;
            }
            else if (Input.GetKeyDown(KeyCode.KeypadMinus))
            {
                // Decreases the rain intensity on the slider
                rainSlider.GetComponent<Slider>().value -= 0.1f;
                rainIntensity = rainSlider.GetComponent<Slider>().value;
            }
            else if (Input.GetKeyDown(KeyCode.LeftArrow))
            {
                // Activates the left turn light button
                leftTurnButton.GetComponent<Button>().onClick.Invoke();
            }
            else if (Input.GetKeyDown(KeyCode.RightArrow))
            {
                // Activates the right turn light button
                rightTurnButton.GetComponent<Button>().onClick.Invoke();
            }
            else if (Input.GetKeyDown(KeyCode.UpArrow))
            {
                // Activates the long beam lights button
                longBeamButton.GetComponent<Button>().onClick.Invoke();
            }
            else if (Input.GetKeyDown(KeyCode.DownArrow))
            {
                // Activates the low beam lights button
                lowBeamButton.GetComponent<Button>().onClick.Invoke();
            }
            else if (Input.GetKeyDown(KeyCode.R))
            {
                // Activates the reset button
                resetButton.GetComponent<Button>().onClick.Invoke();
                cameraParentObject.GetComponent<MSCameraController>().enabled = true;
            }
            else if (Input.GetKeyDown(KeyCode.Escape))
            {
                // Closes the program
                Application.Quit();
            }
        }
    }
}